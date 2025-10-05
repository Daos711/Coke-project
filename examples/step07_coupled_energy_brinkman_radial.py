# -*- coding: utf-8 -*-
# ШАГ 07: Сопряжение Энергии с радиальным Бринкманом u(r,z)
# - Внешний цикл: [ u(r,z) из Brinkman ] -> [ Energy (конвекция u) ] -> повтор
# - Точный массовый расход на каждой z-грани (перенормировка uz_face)
# - Используются уже написанные модули проекта.

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

# --------------------- результаты ---------------------
OUT = Path("results") / "solvers" / "step07_coupled_energy_brinkman_radial"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------- геометрия ----------------------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

r = grid.r_centers
z = grid.z_centers
R = D * 0.5

# радиальные "лица" и площадь осевых граней ячеек (для mdot на z-гранях)
r_faces = np.zeros(NR + 1)
r_faces[0] = 0.0
for i in range(1, NR):
    r_faces[i] = 0.5 * (r[i-1] + r[i])
r_faces[-1] = R
dr_i = r_faces[1:] - r_faces[:-1]         # (NR,)
Az_band = 2.0 * np.pi * r * dr_i          # (NR,) площадь осевой грани «кольца»
# ЭТО ЖЕ — площадь кольца в сечении, используем для осесимм. усреднения ⟨u⟩

# --------------------- физика -------------------------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15       # K
m_dot = 5.0/1000.0/60.0                             # кг/с
gamma = np.ones((NR, NZ))                           # пустой барабан (ε≈1) -> Δп ~ 0, это норма

# --------------------- энергия ------------------------
E = EnergySolver(grid, fvm, vr, EnergySettings(
    dt=1.0, max_iters=3000, min_iters=50, tol=2e-4, print_every=100
))

# --------------------- бринкман (радиальный) ----------
M = MomentumBrinkman(
    grid=grid, fluid=vr,
    settings=MomentumSettings(
        p_outlet=0.0, diameter=D, dp_particle=1e-3,
        gamma_min=1e-3, rho_ref_mode="inlet"
    )
)

# --------------------- вспом. функции -----------------
def uz_faces_from_u(u_rz, T):
    """
    Интерполяция u(r,z) на z-грани: среднее между соседними колонками.
    Потом точная перенормировка под целевой m_dot на каждой z-грани.
    """
    nr, nz = u_rz.shape
    uz_face = np.zeros((nr, nz + 1))

    # плотность на колонках и на гранях
    T_col  = T.mean(axis=0)                     # (nz,)
    rho_col= vr.density(T_col)                  # (nz,)
    rho_face = np.zeros(nz + 1)
    rho_face[0]  = rho_col[0]
    rho_face[-1] = rho_col[-1]
    if nz > 1:
        rho_face[1:-1] = 0.5 * (rho_col[:-1] + rho_col[1:])

    # интерполяция скорости на z-гранях
    uz_face[:, 0]  = u_rz[:, 0]
    uz_face[:, -1] = u_rz[:, -1]
    if nz > 1:
        uz_face[:, 1:-1] = 0.5 * (u_rz[:, :-1] + u_rz[:, 1:])

    # точная перенормировка по m_dot для каждой грани
    # mdot_face = rho_face * sum( uz_face[:,j] * Az_band * eps_j )
    eps_col = np.clip(gamma.mean(axis=0), 1e-3, 1.0)
    eps_face = np.zeros(nz + 1)
    eps_face[0]  = eps_col[0]
    eps_face[-1] = eps_col[-1]
    if nz > 1:
        eps_face[1:-1] = 0.5 * (eps_col[:-1] + eps_col[1:])

    for j in range(nz + 1):
        mdot_j = rho_face[j] * np.sum(uz_face[:, j] * Az_band * eps_face[j])
        if mdot_j != 0.0:
            uz_face[:, j] *= (m_dot / mdot_j)
        else:
            # защита: если профиль нулевой (не должно быть), подставим plug
            A_free = eps_face[j] * np.pi * R * R
            uz_face[:, j] = (m_dot / (rho_face[j] * A_free)) * np.ones(nr)

    ur_face = np.zeros((NR + 1, NZ))  # радиальный поток не используем на шаге 7
    return ur_face, uz_face

# --------------------- внешний цикл -------------------
T = np.ones((NR, NZ)) * T_in
outer_max = 6
hist_dT = []

print("======================================================================")
print("ШАГ 07: Энергия ⟷ радиальный Бринкман u(r,z) (внешние итерации)")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м; ṁ={m_dot*1e3*60:.2f} г/мин")
print("======================================================================")

for it in range(1, outer_max + 1):
    # 1) Радиальный Бринкман: u(r,z), dp/dz, p(z)
    p_z, u_rz, dpdz_z, infoM = M.compute_radial_analytic(T=T, gamma=gamma, m_dot=m_dot)

    # 2) Узлы для энергии: скорости на z-гранях с точным m_dot
    ur_face, uz_face = uz_faces_from_u(u_rz, T)

    # 3) Энергия: один прогон (как шаг 4/6) до сходимости
    bc_type  = {'axis': 'neumann', 'wall': 'dirichlet', 'inlet': 'dirichlet', 'outlet': 'neumann'}
    bc_value = {'axis': 0.0,       'wall': T_wall,       'inlet': T_in,       'outlet': 0.0}

    Told = T.copy()
    T, res, _ = E.solve_pseudo_transient(T, ur_face, uz_face, bc_type, bc_value)
    dT = float(np.max(np.abs(T - Told)))
    hist_dT.append(dT)
    print(f"[Outer {it}/{outer_max}] max|ΔT|={dT:.3e} K,  ⟨-dp/dz⟩={-dpdz_z.mean():.3e} Pa/m")

# --------------------- графики ------------------------
# сходимость внешнего цикла
plt.figure(figsize=(6.5,4))
plt.semilogy(range(1, len(hist_dT)+1), hist_dT, 'o-')
plt.grid(alpha=0.3); plt.xlabel("Внешняя итерация"); plt.ylabel("max|ΔT|, K")
plt.title("Сходимость: энергия ⟷ u(r,z) (Бринкман)")
plt.tight_layout(); plt.savefig(OUT/"outer_convergence.png", dpi=150)

# T поле
plt.figure(figsize=(9,4.8))
im = plt.imshow(T, origin='lower', aspect='auto',
                extent=[z.min(), z.max(), r.min(), r.max()], cmap='hot')
plt.colorbar(im, label='T, K')
plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('T(r,z) после сопряжённого расчёта (шаг 07)')
plt.tight_layout(); plt.savefig(OUT/"T_field.png", dpi=150)

# T(z) среднее по r
plt.figure(figsize=(6.5,4))
plt.plot(T.mean(axis=0), z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('T̄(z), K'); plt.ylabel('z, м')
plt.title('Средняя по радиусу температура')
plt.tight_layout(); plt.savefig(OUT/"T_axial_mean.png", dpi=150)

# u_mean vs plug  (ИСПРАВЛЕНО: осесимметрическое среднее по площади кольца)
rho_z = infoM["rho_z"]
eps_z = infoM["eps_z"]
A_free_z = eps_z * np.pi * R * R
v_plug = m_dot / (rho_z * A_free_z)                 # plug из того же ρ, ε

ring_area = Az_band                                  # 2π r Δr, площадь кольца
v_mean = (u_rz * ring_area[:, None]).sum(axis=0) / ring_area.sum()

plt.figure(figsize=(6.5,4))
plt.plot(v_mean, z, lw=2, label='⟨u⟩(z) — площадь')
plt.plot(v_plug, z, '--', lw=1.8, label='plug(z)')
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel('скорость, м/с'); plt.ylabel('z, м')
plt.title('Средняя скорость vs plug')
plt.tight_layout(); plt.savefig(OUT/"u_mean_vs_plug.png", dpi=150)

# несколько радиальных сечений u(r)
idxs = [0, NZ//2, NZ-1]
plt.figure(figsize=(6.5,4))
for j in idxs:
    plt.plot(r, u_rz[:, j], lw=2, label=f"z={z[j]:.3f} м")
plt.grid(alpha=0.3); plt.legend(); plt.xlabel('r, м'); plt.ylabel('u(r), м/с')
plt.title('Радиальные профили скорости u(r) (несколько z)')
plt.tight_layout(); plt.savefig(OUT/"u_radial_profiles.png", dpi=150)

# dp/dz профиль  (ИСПРАВЛЕНО: единицы — Па/м, без деления на 1e3)
plt.figure(figsize=(6.5,4))
plt.plot(-dpdz_z, z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('-dp/dz, Па/м'); plt.ylabel('z, м')
plt.title('Профиль -dp/dz (рад. Бринкман)')
plt.tight_layout(); plt.savefig(OUT/"dpdz_profile.png", dpi=150)

plt.show()

# текстовый итог
delta_p = np.trapezoid(dpdz_z, z)
print("======================================================================")
print("ШАГ 07: Энергия ⟷ радиальный Бринкман u(r,z)")
print(f"Итог: ⟨-dp/dz⟩={(-dpdz_z).mean():.3e} Па/м,  Δp≈{delta_p:.3e} Па")
print(f"Графики и файлы: {OUT}")
print("======================================================================")
