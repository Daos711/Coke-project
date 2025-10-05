# -*- coding: utf-8 -*-
# ШАГ 05: 1D Stokes–Brinkman (без итераций SIMPLEC)
# Берём скорость из расхода и считаем -dp/dz по Eq.(5), затем p(z) интегрированием.
# K и C2 поставлены такими же, как печатались в шаге 2 — для сопоставимости.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.energy import EnergySolver, EnergySettings

# ---------- директория результатов ----------
OUT_DIR = Path("results") / "solvers" / "step05_simplec_stokes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- геометрия / сетка ----------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D / 2, height=H)
grid = AxiSymmetricGrid(geom)
fvm = FiniteVolumeDiscretization(grid)

# ---------- физика / режим ----------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15
m_dot = 5.0 / 1000.0 / 60.0   # кг/с
gamma = np.ones((NR, NZ))     # пористость ≈ 1 (на данном шаге без кокс-скелета)

# ---------- подсобные ----------
def free_area_column(gamma_col: np.ndarray) -> float:
    g = float(np.clip(np.mean(gamma_col), 1e-6, 1.0))
    return g * np.pi * (D / 2) ** 2

def build_faces_from_mdot(mdot: float, rho_ref: float, gamma_field: np.ndarray):
    ur = np.zeros((NR + 1, NZ))
    uz = np.zeros((NR, NZ + 1))
    for j in range(NZ + 1):
        jj = min(max(j - 1, 0), NZ - 1)
        A = free_area_column(gamma_field[:, jj])
        uz[:, j] = mdot / (rho_ref * A)
    return ur, uz

# ---------- ШАГ A: температура для μ(T), ρ(T) ----------
rho_in = vr.density(T_in)
ur_face, uz_face = build_faces_from_mdot(m_dot, rho_in, gamma)

bc_type = {"axis": "neumann", "wall": "dirichlet", "inlet": "dirichlet", "outlet": "neumann"}
bc_val  = {"axis": 0.0,        "wall": T_wall,    "inlet": T_in,        "outlet": 0.0}

T0 = np.full((NR, NZ), T_in)
E = EnergySolver(grid, fvm, vr, EnergySettings(dt=1.0, max_iters=3000, min_iters=50, tol=2e-4, print_every=100))
T, res_hist, _ = E.solve_pseudo_transient(T0, ur_face, uz_face, bc_type, bc_val)

# усреднения вдоль радиуса
T_z = T.mean(axis=0)                # (NZ,)
mu_z = vr.viscosity(T_z)            # (NZ,)
rho_z = vr.density(T_z)             # (NZ,)

# скорость из расхода (интринзик)
A_free = free_area_column(gamma[:, 0])  # тут γ≈const, поэтому A_free по столбцу 0
v_z = m_dot / (rho_z * A_free)          # (NZ,)

# ---------- ШАГ B: -dp/dz и p(z) по Eq.(5) ----------
# Значения K и C2 из шага 2 (для 1-в-1 сопоставимости):
K  = 1.905e-06    # м^2
C2 = 2.041e+02    # 1/м

dpdz_vis = mu_z * v_z / K                          # Па/м
dpdz_in  = 0.5 * C2 * rho_z * np.abs(v_z) * v_z    # Па/м
dpdz = dpdz_vis + dpdz_in

# Интегрируем сверху вниз: p(H)=0
z_faces = grid.z_faces              # (NZ+1,)
dz = np.diff(z_faces)               # (NZ,)
p = np.zeros(NZ)
p[-1] = 0.0
for j in range(NZ - 2, -1, -1):
    p[j] = p[j + 1] + 0.5 * (dpdz[j] + dpdz[j + 1]) * dz[j]  # трапеции

# контроль расхода
mdot_z = rho_z * v_z * A_free
max_rel_m_err = float(np.max(np.abs(mdot_z - m_dot)) / m_dot)

# печать
print("=" * 70)
print("ШАГ 05: SIMPLEC 1D (Stokes–Brinkman без итераций)")
print(f"A = {A_free: .6e} м²,  K = {K: .3e} м²,  C2 = {C2: .3e} 1/м")
print(f"⟨dp/dz⟩ = {np.mean(dpdz): .3e} Па/м,   Δp ≈ {np.trapezoid(dpdz, grid.z_centers): .3e} Па")
print(f"max относит. ошибка расхода = {max_rel_m_err: .3e}")
print("=" * 70)

# ---------- графики ----------
# 1) -dp/dz
plt.figure(figsize=(7.0, 4.6))
plt.plot(dpdz/1e3, grid.z_centers, lw=2.0, label="-dp/dz (Eq.5)")
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel("dp/dz, кПа/м"); plt.ylabel("z, м")
plt.title("Профиль -dp/dz (Brinkman + Ergun, 1D)")
plt.tight_layout(); plt.savefig(OUT_DIR/"dpdz.png", dpi=140)

# 2) p(z)
plt.figure(figsize=(7.0, 4.6))
plt.plot(p/1e3, grid.z_centers, lw=2.0)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel("p, кПа"); plt.ylabel("z, м")
plt.title("Давление p(z) (интеграл от dp/dz, p(H)=0)")
plt.tight_layout(); plt.savefig(OUT_DIR/"p_profile.png", dpi=140)

# 3) «сходимость» (фиктивная ось, т.к. без итераций)
plt.figure(figsize=(7.0, 4.6))
plt.semilogy([0, 1], [max_rel_m_err, max_rel_m_err], lw=2.0)
plt.grid(alpha=0.3, which="both")
plt.xlabel("итерация (фиктивная шкала)"); plt.ylabel("max|ошибка ṁ| / ṁ")
plt.title("Сходимость SIMPLEC 1D (без итераций)")
plt.tight_layout(); plt.savefig(OUT_DIR/"residuals.png", dpi=140)
plt.show()

print(f"Сохранено в: {OUT_DIR}")
