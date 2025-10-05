# -*- coding: utf-8 -*-
# ШАГ 04: Энергия с реальным u(r,z) из Бринкмана (радиальный профиль).
# Используем ранее написанные модули + новую утилиту u_to_face_velocities.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.energy import EnergySolver, EnergySettings
from solvers.momentum import MomentumBrinkman, MomentumSettings
from solvers.momentum import BrinkmanRadialSolver, RadialSettings
from solvers.momentum import u_to_face_velocities  # <— новая утилита

# >>> директории вывода <<<
OUT_DIR = Path("results") / "solvers" / "step04_energy_with_brinkman"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# >>> геометрия <<<
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

# >>> физсвойства / режим <<<
vr     = VacuumResidue(1)
T_in   = 370.0 + 273.15
T_wall = 510.0 + 273.15
m_dot  = 5.0/1000.0/60.0   # кг/с
rho_in = vr.density(T_in)

gamma  = np.ones((NR, NZ))   # на этом шаге «пустой барабан»

# >>> шаг 2: осевой dp/dz по Brinkman+Ergun (для старта радиального шага)
MB = MomentumBrinkman(grid=grid, fluid=vr, settings=MomentumSettings(
    p_outlet=0.0, diameter=D, dp_particle=1e-3, gamma_min=1e-3, rho_ref_mode="inlet"
))
p_ax, v_ax, dpdz_ax, info_ax = MB.compute(T=np.full((NR, NZ), T_in), gamma=gamma, m_dot=m_dot)

# >>> шаг 3: радиальный профиль u(r,z) с точным расходом
MR = BrinkmanRadialSolver(grid=grid, fluid=vr, settings=RadialSettings(
    dp_particle=1e-3, gamma_min=1e-3, max_outer=80, max_inner=80,
    tol_u=1e-10, tol_mdot=1e-8
))
u, p, dpdz, info_rad = MR.compute(
    T=np.full((NR, NZ), T_in), gamma=gamma, m_dot=m_dot, dpdz_init=dpdz_ax
)

# >>> переводим u(r,z) на грани (узлы энергии ожидают face-скорости)
ur_face, uz_face = u_to_face_velocities(u, grid, gamma, rho_ref=rho_in, m_dot=m_dot)

# контроль массового расхода через ВСЕ z-грани:
r = grid.r_centers.astype(float); dr = float(grid.dr)
ring_area = 2.0*np.pi*r*dr
mdot_faces = []
for jf in range(NZ+1):
    # берём «среднюю» пористость на грани
    jL = max(jf-1, 0); jR = min(jf, NZ-1)
    g_face = 0.5*(gamma[:, jL] + gamma[:, jR])
    Q = float(np.sum(g_face * uz_face[:, jf] * ring_area))
    mdot_faces.append(rho_in * Q)
mdot_faces = np.array(mdot_faces)
imb = 100.0 * np.max(np.abs(mdot_faces - m_dot))/m_dot

# >>> решаем энергию с реальными скоростями
bc_type  = {'axis':'neumann','wall':'dirichlet','inlet':'dirichlet','outlet':'neumann'}
bc_value = {'axis':0.0,       'wall':T_wall,     'inlet':T_in,       'outlet':0.0}
T0 = np.ones((NR, NZ)) * T_in

E = EnergySolver(grid, fvm, vr, EnergySettings(
    dt=1.0, max_iters=3000, min_iters=200, tol=5e-4, print_every=100
))
T, hist_all, hist_int = E.solve_pseudo_transient(T0, ur_face, uz_face, bc_type, bc_value)

print("="*70)
print("ШАГ 04: Энергия с u(r,z) из Бринкмана (конвекция реальной скоростью)")
print(f"Проверка расхода на z-гранях: max|mdot_face - m_dot|/m_dot = {imb:.3e} (д.б. ~ машинный ноль)")
print("="*70)

# -------------------- графики --------------------
# 1) сходимость энергии
plt.figure(figsize=(7,4))
plt.semilogy(hist_all,  label="max|ΔT| (всё поле)")
plt.semilogy(hist_int,  label="max|ΔT| (внутр.)")
plt.xlabel("итерация"); plt.ylabel("K"); plt.title("Сходимость энергии (шаг 4)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/"energy_residuals.png", dpi=140); plt.show()

# 2) поле T(r,z)
z_plot = grid.z_centers; r_plot = grid.r_centers
plt.figure(figsize=(9,5))
levels = 40
cs = plt.contourf(z_plot, r_plot, T, levels=levels, cmap="hot")
cb = plt.colorbar(cs); cb.set_label("T, K")
plt.xlabel("z, м"); plt.ylabel("r, м"); plt.title("Температура после сходимости (шаг 4)")
plt.tight_layout(); plt.savefig(OUT_DIR/"T_field.png", dpi=140); plt.show()

# 3) профиль T̄(z)
T_mean = T.mean(axis=0)
plt.figure(figsize=(7,4))
plt.plot(T_mean, z_plot, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel("T̄(z), K"); plt.ylabel("z, м"); plt.title("Средняя температура по радиусу")
plt.tight_layout(); plt.savefig(OUT_DIR/"T_profile_axial.png", dpi=140); plt.show()

print(f"Сохранено в: {OUT_DIR}")
