# -*- coding: utf-8 -*-
# ШАГ 02 (PATCH по Eq.5): Momentum без коррекции давления — Brinkman + Ergun (интринзик-скорость v)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from physics.correlations import PorousDrag
from solvers.energy import EnergySolver, EnergySettings
from solvers.momentum import MomentumBrinkman, MomentumSettings

# -------- директория результатов --------
OUT_DIR = Path("results") / "solvers" / "step02_momentum_brinkman"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- геометрия / сетка --------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

# -------- физика / режим --------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15
m_dot = 5.0/1000.0/60.0  # кг/с
gamma_const = 0.95
gamma = np.ones((NR, NZ)) * gamma_const

# -------- утилита: усреднить по радиусу, если нужно --------
def axialize(x):
    x = np.asarray(x)
    return x.mean(axis=0) if x.ndim == 2 else x

# -------- вспом. функции для шага энергии --------
def axial_free_area(gamma_col):
    g = float(np.clip(np.mean(gamma_col), 1e-3, 1.0))
    return g * np.pi * (D/2.0)**2

def build_v_faces(mdot, rho_ref, gamma_field):
    ur_face = np.zeros((NR+1, NZ))
    uz_face = np.zeros((NR, NZ+1))
    for j in range(NZ+1):
        jj = min(max(j-1, 0), NZ-1)
        A_free = axial_free_area(gamma_field[:, jj])
        v = mdot/(rho_ref*A_free)
        uz_face[:, j] = v
    return ur_face, uz_face

# -------- ШАГ A: энергия --------
rho_in = vr.density(T_in)
ur_face, uz_face = build_v_faces(m_dot, rho_in, gamma)

bc_type  = {'axis':'neumann','wall':'dirichlet','inlet':'dirichlet','outlet':'neumann'}
bc_value = {'axis':0.0,       'wall':T_wall,     'inlet':T_in,       'outlet':0.0}

T0 = np.ones((NR, NZ)) * T_in
E = EnergySolver(grid, fvm, vr, EnergySettings(dt=1.0, max_iters=1500, min_iters=50, tol=2e-4, print_every=100))
T, _, _ = E.solve_pseudo_transient(T0, ur_face, uz_face, bc_type, bc_value)

# -------- ШАГ B: моментум (Eq.5: v-интринзик, γ²/γ³) --------
mom = MomentumBrinkman(
    grid=grid,
    fluid=vr,
    settings=MomentumSettings(
        p_outlet=0.0,
        diameter=D,
        dp_particle=1e-3,
        gamma_min=1e-3,
        rho_ref_mode="local",   # чтобы не было масс. дисбаланса от нормировки
    ),
)
p, v, dpdz, info = mom.compute(T=T, gamma=gamma, m_dot=m_dot)

# Приводим всё к векторам по z:
p_z    = axialize(p)
v_z    = axialize(v)
dpdz_z = axialize(dpdz)

# ----- бэко-совместимые метрики -----
delta_p_calc = float(p_z[0] - p_z[-1])
delta_p = float(info.get('delta_p', delta_p_calc))

v_min = float(info.get('v_min', np.min(v_z)))
v_max = float(info.get('v_max', np.max(v_z)))

dpdz_vis_mean = float(info.get('dpdz_vis_mean',
                        np.mean(info.get('dpdz_vis', np.zeros_like(dpdz_z)))))
dpdz_in_mean  = float(info.get('dpdz_in_mean',
                        np.mean(info.get('dpdz_in',  np.zeros_like(dpdz_z)))))

# массовый дисбаланс: m(z)=ρ(z)·v̄(z)·A_free(z)
rho_z = np.asarray(info.get('rho_z', vr.density(T.mean(axis=0))))
gamma_bar = np.clip(gamma.mean(axis=0), 1e-3, 0.999999)
A_free_z = gamma_bar * np.pi * (D/2.0)**2
m_z = rho_z * v_z * A_free_z
imbalance_percent = info.get('imbalance_percent', None)
if imbalance_percent is None:
    m_mean = float(np.mean(m_z))
    imbalance_percent = float(100.0*(np.max(m_z)-np.min(m_z))/m_mean) if m_mean > 0 else 0.0

print("="*70)
print("ШАГ 02 (PATCH): MOMENTUM по Eq.5 — v-интринзик, γ²/γ³ в пористых терминах")
print(f"v_min..v_max={v_min:.4e}..{v_max:.4e} м/с")
print(f"⟨dp/dz⟩_vis={dpdz_vis_mean:.3e} Па/м, ⟨dp/dz⟩_in={dpdz_in_mean:.3e} Па/м")
print(f"Δp={delta_p/1e3:.6f} кПа, дисбаланс массы={imbalance_percent:.4f} %")
print("="*70)

# -------- графики --------
z = grid.z_centers

# 1) Давление p(z)
plt.figure(figsize=(6,4))
plt.plot(p_z/1e3, z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel("p, кПа"); plt.ylabel("z, м")
plt.title("Профиль давления (Brinkman + Ergun, Eq.5)")
plt.tight_layout(); plt.savefig(OUT_DIR/"p_profile.png", dpi=140); plt.show()

# 2) Скорость v(z) (интринзик, осреднён по радиусу)
plt.figure(figsize=(6,4))
plt.plot(v_z, z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel("v, м/с"); plt.ylabel("z, м")
plt.title("Интринзик-скорость по высоте (v̄_r)")
plt.tight_layout(); plt.savefig(OUT_DIR/"v_profile.png", dpi=140); plt.show()

# 3) Компоненты dp/dz (пересчёт по Eq.5, проверка)
mu_z  = vr.viscosity(T.mean(axis=0))
rho_z = vr.density(T.mean(axis=0))
g_bar = gamma_bar
K_z   = np.array([PorousDrag.ergun_permeability(float(g), 1e-3) for g in g_bar])
C2_z  = np.array([PorousDrag.ergun_inertial   (float(g), 1e-3) for g in g_bar])

dpdz_vis_chk = (g_bar**2) * (mu_z/np.maximum(K_z,1e-30)) * v_z
dpdz_in_chk  = 0.5 * (g_bar**3) * C2_z * rho_z * np.abs(v_z) * v_z

plt.figure(figsize=(6,4))
plt.plot(dpdz_vis_chk/1e3, z, lw=2, label="Brinkman (вязк.)")
plt.plot(dpdz_in_chk/1e3,  z, lw=1.8, label="Ergun (инерц.)")
plt.plot(dpdz_z/1e3,       z, lw=2.2, label="Сумма из солвера")
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel("dp/dz, кПа/м"); plt.ylabel("z, м")
plt.title("Градиент давления (Eq.5)")
plt.tight_layout(); plt.savefig(OUT_DIR/"dpdz_components.png", dpi=140); plt.show()

print(f"Сохранено в: {OUT_DIR}")
