# -*- coding: utf-8 -*-
# ШАГ 09: Двухтемпературная энергия (fluid↔coke) + u(r,z) Бринкман + кинетика γ(t)
# -----------------------------------------------------------------------------
# Цикл:
#   [ u(r,z | γ) ] -> [ 2T Energy (T_f, T_c | u, γ) ] -> [ γ^+(T_f) ] -> ...
# Использует:
#  - BrinkmanRadialSolver (импульс)
#  - EnergyTwoTemperature (двухтемпературная энергия)
#  - PorosityKinetics (рост кокса → γ)
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.momentum import BrinkmanRadialSolver, RadialSettings, u_to_face_velocities
from solvers.energy import EnergyTwoTemperature, TwoTempSettings
from solvers.porosity import PorosityKinetics, PorositySettings

# --------------------- результаты ---------------------
OUT = Path("results") / "solvers" / "step09_two_temp_energy"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------- геометрия ----------------------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

r = grid.r_centers
z = grid.z_centers

# --------------------- физика -------------------------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15     # K
m_dot = 5.0/1000.0/60.0                           # кг/с

# начальные поля
T_f = np.full((NR, NZ), T_in, dtype=float)        # жидкость
T_c = np.full((NR, NZ), T_in, dtype=float)        # «кокс/скелет»
alpha_c = np.zeros((NR, NZ), dtype=float)
gamma   = 1.0 - alpha_c

# --------------------- импульс ------------------------
mom = BrinkmanRadialSolver(
    grid, vr,
    RadialSettings(dp_particle=1e-3, gamma_min=1e-3,
                   max_outer=80, max_inner=80, tol_u=1e-10, tol_mdot=1e-6)
)

dpdz0 = np.zeros(NZ)
u_rz, p_rz, dpdz_z, infoM = mom.compute(T=T_f, gamma=gamma, m_dot=m_dot, dpdz_init=dpdz0)
rho_ref = float(infoM["rho_z"][0])   # плотность на входной колонке
ur_face, uz_face = u_to_face_velocities(u_rz, grid, gamma, rho_ref, m_dot)

# --------------------- энергия 2T ---------------------
# ВАЖНО: TwoTempSettings не принимает H_vol. Теплообмен U_v считается внутри,
# а параметр h_mult позволяет его масштабировать (по умолчанию 1.0).
e2t = EnergyTwoTemperature(
    grid, fvm, vr,
    TwoTempSettings(
        dt=1.0, max_iters=5000, min_iters=50, tol=1e-4, print_every=200,
        dp_particle=1e-3, gamma_min=1e-3,
        k_solid=1.5, rho_solid=1400.0, cp_solid=1000.0,
        h_mult=1.0,                 # можно усилить обмен, например 1.5–2.0
        coupling_tol=1e-4, max_coupling_iters=800
    )
)

# ГУ для жидкости
bc_fluid_type = {'axis': 'neumann', 'wall': 'dirichlet', 'inlet': 'dirichlet', 'outlet': 'neumann'}
bc_fluid_val  = {'axis': 0.0,       'wall': T_wall,       'inlet': T_in,       'outlet': 0.0}

# ГУ для «кокса»
bc_coke_type  = {'axis': 'neumann', 'wall': 'dirichlet', 'inlet': 'neumann', 'outlet': 'neumann'}
bc_coke_val   = {'axis': 0.0,       'wall': T_wall,       'inlet': 0.0,       'outlet': 0.0}

# --------------------- кинетика γ ---------------------
PK = PorosityKinetics(
    fluid=vr,
    settings=PorositySettings(
        rho_coke=1400.0, k0=1.0, E=60_000.0, order=1.0,
        gamma_min=1e-3, alpha_max=0.999
    )
)

# --------------------- временной цикл -----------------
n_steps = 60
dt_rxn  = 1.0

hist_alpha_mean = []
hist_dpdz_mean  = []

print("="*70)
print("ШАГ 09: Двухтемпературная энергия (fluid↔coke) + Бринкман u(r,z) + γ-кинетика")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м; ṁ={m_dot*60*1000:.2f} г/мин")
print("="*70)

for it in range(1, n_steps + 1):
    # 1) энергия 2T (сопряжённо для T_f и T_c)
    T_f, T_c, histT = e2t.solve_coupled(
        T_f, T_c, ur_face, uz_face, gamma,
        bc_fluid_type, bc_fluid_val,
        bc_coke_type,  bc_coke_val
    )

    # 2) кинетика пористости (по температуре жидкости)
    alpha_c, gamma, infoP = PK.advance(T=T_f, alpha_c=alpha_c, dt=dt_rxn)

    # 3) импульс с новой γ
    u_rz, p_rz, dpdz_z, infoM = mom.compute(T=T_f, gamma=gamma, m_dot=m_dot, dpdz_init=dpdz_z)
    ur_face, uz_face = u_to_face_velocities(u_rz, grid, gamma, rho_ref, m_dot)

    print(f"[t={it:03d}] ⟨α_C⟩={alpha_c.mean():.3e}, γ_min={gamma.min():.3f}, "
          f"⟨-dp/dz⟩={(-dpdz_z).mean():.3e} Па/м, max|ΔT|={histT[-1]:.2e}")

    hist_alpha_mean.append(float(alpha_c.mean()))
    hist_dpdz_mean.append(float((-dpdz_z).mean()))

print("="*70)
print(f"Итог: ⟨α_C⟩={alpha_c.mean():.4f}, γ_min={gamma.min():.4f}, "
      f"⟨-dp/dz⟩={np.mean(hist_dpdz_mean[-10:]):.3e} Па/м")
print(f"Графики и файлы: {OUT}")
print("="*70)

# --------------------- графики ------------------------
# 1) осевые средние температур
plt.figure(figsize=(6.6,4.2))
plt.plot(T_f.mean(axis=0), z, label='⟨T_f⟩(z)')
plt.plot(T_c.mean(axis=0), z, label='⟨T_c⟩(z)')
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel('T, K'); plt.ylabel('z, м')
plt.title('Осевая средняя температура (fluid vs coke)')
plt.tight_layout(); plt.savefig(OUT/"axial_T_f_T_c.png", dpi=150)

# 2) профиль -dp/dz
plt.figure(figsize=(6.6,4.2))
plt.plot(-dpdz_z, z)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('-dp/dz, Па/м'); plt.ylabel('z, м')
plt.title('Профиль -dp/dz (после шага 09)')
plt.tight_layout(); plt.savefig(OUT/"dpdz_profile.png", dpi=150)

# 3) эволюция ⟨α_C⟩ и ⟨-dp/dz⟩
t = np.arange(1, n_steps+1)
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(t, hist_alpha_mean, 'o-'); ax[0].grid(alpha=0.3)
ax[0].set_xlabel('шаг'); ax[0].set_ylabel('⟨α_C⟩'); ax[0].set_title('Рост средней доли кокса')
ax[1].plot(t, hist_dpdz_mean, 'o-'); ax[1].grid(alpha=0.3)
ax[1].set_xlabel('шаг'); ax[1].set_ylabel('⟨-dp/dz⟩, Па/м'); ax[1].set_title('Эволюция среднего -dp/dz')
fig.tight_layout(); fig.savefig(OUT/"evolution_alpha_dpdz.png", dpi=150)

# 4) поля T_f, T_c, γ
extent = [z.min(), z.max(), r.min(), r.max()]

plt.figure(figsize=(9,4.6))
im = plt.imshow(T_f, origin='lower', aspect='auto', extent=extent, cmap='hot')
plt.colorbar(im, label='T_f, K'); plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('T_f(r,z) после шага 09')
plt.tight_layout(); plt.savefig(OUT/"T_f_field.png", dpi=150)

plt.figure(figsize=(9,4.6))
im = plt.imshow(T_c, origin='lower', aspect='auto', extent=extent, cmap='hot')
plt.colorbar(im, label='T_c, K'); plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('T_c(r,z) после шага 09')
plt.tight_layout(); plt.savefig(OUT/"T_c_field.png", dpi=150)

plt.figure(figsize=(9,4.6))
im = plt.imshow(gamma, origin='lower', aspect='auto', extent=extent, cmap='viridis')
plt.colorbar(im, label='γ'); plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('γ(r,z) после шага 09')
plt.tight_layout(); plt.savefig(OUT/"gamma_field.png", dpi=150)

plt.close("all")
