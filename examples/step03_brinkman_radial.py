# -*- coding: utf-8 -*-
# ШАГ 03: радиальный профиль Бринкмана u(r,z) с точным массовым балансом

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.energy import EnergySolver, EnergySettings
from solvers.momentum import (
    MomentumBrinkman, MomentumSettings,
    BrinkmanRadialSolver, RadialSettings
)

# --- папка результатов ---
OUT_DIR = Path("results") / "solvers" / "step03_brinkman_radial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- геометрия/сетка ---
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

# --- физика/режим ---
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15
m_dot = 5.0/1000.0/60.0
gamma = np.ones((NR, NZ)) * 0.95

# --- служебные функции ---
def axial_free_area(gamma_col):
    g = float(np.clip(np.mean(gamma_col), 1e-3, 1.0))
    return g * np.pi * (D/2)**2

def build_v_faces(mdot, rho_ref, gamma_field):
    """Плаг-распределение v(z) на межъячеечных гранях (для шага энергии)."""
    ur_face = np.zeros((NR+1, NZ))
    uz_face = np.zeros((NR,   NZ+1))
    for j in range(NZ+1):
        jj = min(max(j-1, 0), NZ-1)
        A_free = axial_free_area(gamma_field[:, jj])
        v = mdot/(rho_ref*A_free)
        uz_face[:, j] = v
    return ur_face, uz_face

# --- шаг A: энергия (как на шаге 1) ---
rho_in = vr.density(T_in)
ur_face, uz_face = build_v_faces(m_dot, rho_in, gamma)

bc_type  = {'axis':'neumann','wall':'dirichlet','inlet':'dirichlet','outlet':'neumann'}
bc_value = {'axis':0.0,       'wall':T_wall,     'inlet':T_in,       'outlet':0.0}

T0 = np.ones((NR, NZ)) * T_in
E = EnergySolver(grid, fvm, vr, EnergySettings(dt=1.0, max_iters=1500, min_iters=50, tol=2e-4, print_every=100))
T, _, _ = E.solve_pseudo_transient(T0, ur_face, uz_face, bc_type, bc_value)

# --- шаг B: осевой dp/dz (как на шаге 2) ---
mom = MomentumBrinkman(
    grid, vr,
    MomentumSettings(diameter=D, rho_ref_mode="local")   # локальная ρ → нулевой дисбаланс
)
p_ax, v_ax, dpdz0, info_ax = mom.compute(T=T, gamma=gamma, m_dot=m_dot)

# --- шаг C: радиальный профиль Бринкмана с подбором dp/dz ---
rad = BrinkmanRadialSolver(grid, vr, RadialSettings())
u, p, dpdz, info = rad.compute(T=T, gamma=gamma, m_dot=m_dot, dpdz_init=dpdz0)

print("="*70)
print("ШАГ 03: радиальный Бринкман — u(r,z) с точным ṁ")
print(f"max|ошибка расхода| = {info['mdot_rel_error_max']:.4f} %, внеш. итераций = {info['outer_iters_max']}")
print(f"u_min..u_max = {info['u_min']:.4e} .. {info['u_max']:.4e} м/с")
print("="*70)

# --- подготовка осреднений по площади (чтобы 2D → 1D по z) ---
r = grid.r_centers
z = grid.z_centers
area_r = 2*np.pi*r*grid.dr  # площади кольцевых ячеек в осесимм. постановке

def area_avg(field_2d_or_1d):
    """Если подали 2D (nr,nz) — вернём осевой профиль, осреднённый по площади.
       Если уже 1D (nz) — вернём как есть."""
    if np.ndim(field_2d_or_1d) == 2:
        return (field_2d_or_1d * area_r[:, None]).sum(axis=0) / area_r.sum()
    return field_2d_or_1d

# --- графики ---
# 1) Несколько профилей u(r) на разных высотах
z_ids = [int(0.05*NZ), int(0.50*NZ), int(0.95*NZ)]
plt.figure(figsize=(6,4))
for j in z_ids:
    plt.plot(u[:, j], r, lw=2, label=f"z={z[j]:.2f} м")
plt.grid(alpha=0.3); plt.legend()
plt.xlabel("u(r), м/с"); plt.ylabel("r, м")
plt.title("Радиальные профили скорости (Бринкман)")
plt.tight_layout(); plt.savefig(OUT_DIR/"u_radial_profiles.png", dpi=140); plt.show()

# 2) Средняя по сечению ⟨u⟩(z) и plug v(z) для сравнения
u_avg_z = area_avg(u)          # (nz,)
v_ax_z  = area_avg(v_ax)       # гарантия 1D по z
plt.figure(figsize=(6,4))
plt.plot(v_ax_z,  z, lw=2, label="plug (шаг 2)")
plt.plot(u_avg_z, z, lw=2, label="⟨u⟩ (шаг 3)")
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel("скорость, м/с"); plt.ylabel("z, м")
plt.title("Средняя скорость по высоте")
plt.tight_layout(); plt.savefig(OUT_DIR/"u_mean_vs_plug.png", dpi=140); plt.show()

# 3) Профиль dp/dz: стартовый (шаг 2) vs после подбора (шаг 3)
plt.figure(figsize=(6,4))
plt.plot(area_avg(dpdz)/1e3,  z, lw=2,   label="после подбора (шаг 3)")
plt.plot(area_avg(dpdz0)/1e3, z, lw=1.5, label="старт (шаг 2)")
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel("dp/dz, кПа/м"); plt.ylabel("z, м")
plt.title("Градиент давления по высоте")
plt.tight_layout(); plt.savefig(OUT_DIR/"dpdz_compare.png", dpi=140); plt.show()

print(f"Сохранено в: {OUT_DIR}")
