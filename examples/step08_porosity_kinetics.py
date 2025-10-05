# -*- coding: utf-8 -*-
# ШАГ 08: Эволюция пористости γ(t) из кинетики кокса + сопряжение u↔T
# -----------------------------------------------------------------------------
# Цикл по "времени":  [ u(r,z|γ) ] -> [ Energy (T|у,γ) ] -> [ γ^+(T) ] -> ...
# Используем:
#  - MomentumBrinkman.compute_radial_analytic (как на шаге 07)
#  - EnergySolver (как на шаге 04/06/07)
#  - НОВОЕ: PorosityKinetics.advance(T, α_C, dt)
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.energy import EnergySolver, EnergySettings
from solvers.momentum import MomentumBrinkman, MomentumSettings, u_to_face_velocities
from solvers.porosity import PorosityKinetics, PorositySettings

# --------------------- результаты ---------------------
OUT = Path("results") / "solvers" / "step08_porosity_kinetics"
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

# --------------------- физика -------------------------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15       # K
m_dot = 5.0/1000.0/60.0                             # кг/с

# начальная пористость (без кокса)
alpha_c = np.zeros((NR, NZ))        # объёмная доля кокса
gamma   = 1.0 - alpha_c             # пористость

# настройки кинетики: подобрать, чтобы был заметный рост α за десятки–сотни секунд
kin_set = PorositySettings(
    rho_coke=1400.0,     # кг/м^3
    k0=1.5,              # 1/с — интенсивность (демо-значение)
    E=60_000.0,          # Дж/моль
    order=1.0,
    gamma_min=1e-3,
    alpha_max=0.95
)
PK = PorosityKinetics(fluid=vr, settings=kin_set)

# --------------------- энергия/моментум ----------------
E = EnergySolver(grid, fvm, vr, EnergySettings(
    dt=1.0, max_iters=3000, min_iters=50, tol=2e-4, print_every=100
))

M = MomentumBrinkman(
    grid=grid, fluid=vr,
    settings=MomentumSettings(
        p_outlet=0.0, diameter=D, dp_particle=1e-3,
        gamma_min=1e-3, rho_ref_mode="inlet"
    )
)

# --------------------- временной цикл ------------------
T = np.ones((NR, NZ)) * T_in

n_steps = 60       # кол-во макрошагов (условных секунд)
dt_rxn  = 1.0      # с — шаг по кинетике на один макрошаг (можно субциклировать)
hist_alpha_mean = []
hist_gamma_min  = []
hist_dpdz_mean  = []
hist_deltaT     = []

print("======================================================================")
print("ШАГ 08: Энергия ⟷ u(r,z) ⟷ Пористость γ(t) (временной цикл)")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м; ṁ={m_dot*1e3*60:.2f} г/мин")
print("======================================================================")

for it in range(1, n_steps + 1):
    # 1) Радиальный Бринкман: u(r,z), dp/dz, p(z)
    p_z, u_rz, dpdz_z, infoM = M.compute_radial_analytic(T=T, gamma=gamma, m_dot=m_dot)

    # 2) Энергия: скорости на z-гранях -> solve до сходимости
    rho_ref = infoM["rho_z"][0]  # для нормировки на гранях используем входную колонку
    ur_face, uz_face = u_to_face_velocities(u_rz, grid, gamma, rho_ref, m_dot)

    bc_type  = {'axis': 'neumann', 'wall': 'dirichlet', 'inlet': 'dirichlet', 'outlet': 'neumann'}
    bc_value = {'axis': 0.0,       'wall': T_wall,       'inlet': T_in,       'outlet': 0.0}

    Told = T.copy()
    T, _, _ = E.solve_pseudo_transient(T, ur_face, uz_face, bc_type, bc_value)
    dT = float(np.max(np.abs(T - Told)))

    # 3) Кинетика пористости: α_C, γ
    alpha_c, gamma, infoP = PK.advance(T=T, alpha_c=alpha_c, dt=dt_rxn)

    # 4) Лог
    hist_alpha_mean.append(float(alpha_c.mean()))
    hist_gamma_min.append(float(gamma.min()))
    hist_dpdz_mean.append(float((-dpdz_z).mean()))
    hist_deltaT.append(dT)
    print(f"[t={it:03d}] ⟨α_C⟩={alpha_c.mean():.3e}, γ_min={gamma.min():.3e}, "
          f"⟨-dp/dz⟩={(-dpdz_z).mean():.3e} Па/м, max|ΔT|={dT:.2e} K")

# --------------------- графики ------------------------

# 1) поле α_C (или γ)
plt.figure(figsize=(9, 4.8))
im = plt.imshow(alpha_c, origin='lower', aspect='auto',
                extent=[z.min(), z.max(), r.min(), r.max()], cmap='magma')
plt.colorbar(im, label='α_C')
plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('α_C(r,z) после шага 08')
plt.tight_layout(); plt.savefig(OUT/"alpha_field.png", dpi=150)

plt.figure(figsize=(9, 4.8))
im2 = plt.imshow(gamma, origin='lower', aspect='auto',
                 extent=[z.min(), z.max(), r.min(), r.max()], cmap='viridis')
plt.colorbar(im2, label='γ')
plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('γ(r,z) после шага 08')
plt.tight_layout(); plt.savefig(OUT/"gamma_field.png", dpi=150)

# 2) осевой профиль средних по радиусу α_C и γ
plt.figure(figsize=(6.5,4))
plt.plot(alpha_c.mean(axis=0), z, lw=2, label='ᾱ_C(z)')
plt.plot(gamma.mean(axis=0), z, lw=2, label='γ̄(z)')
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel('значение'); plt.ylabel('z, м')
plt.title('Осевые профили ᾱ_C и γ̄')
plt.tight_layout(); plt.savefig(OUT/"axial_alpha_gamma.png", dpi=150)

# 3) профиль -dp/dz (финальный)
plt.figure(figsize=(6.5,4))
plt.plot(-dpdz_z, z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('-dp/dz, Па/м'); plt.ylabel('z, м')
plt.title('Профиль -dp/dz (после шага 08)')
plt.tight_layout(); plt.savefig(OUT/"dpdz_profile.png", dpi=150)

# 4) эволюция ⟨α_C⟩ и ⟨-dp/dz⟩
t = np.arange(1, n_steps+1, dtype=float)
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(t, hist_alpha_mean, 'o-')
ax[0].grid(alpha=0.3); ax[0].set_xlabel('шаг'); ax[0].set_ylabel('⟨α_C⟩')
ax[0].set_title('Рост средней доли кокса')
ax[1].plot(t, hist_dpdz_mean, 'o-')
ax[1].grid(alpha=0.3); ax[1].set_xlabel('шаг'); ax[1].set_ylabel('⟨-dp/dz⟩, Па/м')
ax[1].set_title('Эволюция среднего -dp/dz')
plt.tight_layout(); plt.savefig(OUT/"evolution_alpha_dpdz.png", dpi=150)

# 5) температура, как на шаге 07
plt.figure(figsize=(6.5,4))
plt.plot(T.mean(axis=0), z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('T̄(z), K'); plt.ylabel('z, м')
plt.title('Средняя по радиусу температура (после шага 08)')
plt.tight_layout(); plt.savefig(OUT/"T_axial_mean.png", dpi=150)

plt.figure(figsize=(9, 4.8))
imT = plt.imshow(T, origin='lower', aspect='auto',
                 extent=[z.min(), z.max(), r.min(), r.max()], cmap='hot')
plt.colorbar(imT, label='T, K')
plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('T(r,z) после шага 08')
plt.tight_layout(); plt.savefig(OUT/"T_field.png", dpi=150)

plt.show()

# --------------------- сводка -------------------------
print("======================================================================")
print("ШАГ 08: Энергия ⟷ u ⟷ γ(t)")
print(f"Итог: ⟨α_C⟩={alpha_c.mean():.4f}, γ_min={gamma.min():.4f}, "
      f"⟨-dp/dz⟩={np.mean(-dpdz_z):.3e} Па/м")
print(f"Графики и файлы: {OUT}")
print("======================================================================")
