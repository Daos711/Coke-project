# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue
from solvers.energy import EnergySolver, EnergySettings

OUT_DIR = Path("results") / "solvers" / "step01_energy_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# геометрия/сетка
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

# свойства и режим
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15
m_dot = 5.0/1000.0/60.0
gamma = np.ones((NR, NZ))

def axial_free_area(gamma_col):
    g = float(np.clip(np.mean(gamma_col), 1e-3, 1.0))
    return g * np.pi * (D/2)**2

def build_v_faces(mdot, rho_ref, gamma_field):
    ur_face = np.zeros((NR+1, NZ))
    uz_face = np.zeros((NR, NZ+1))
    for j in range(NZ+1):
        jj = min(max(j-1, 0), NZ-1)
        A_free = axial_free_area(gamma_field[:, jj])
        v = mdot/(rho_ref*A_free)
        uz_face[:, j] = v
    return ur_face, uz_face

rho_in = vr.density(T_in)
ur_face, uz_face = build_v_faces(m_dot, rho_in, gamma)

bc_type  = {'axis': 'neumann', 'wall': 'dirichlet', 'inlet': 'dirichlet', 'outlet': 'neumann'}
bc_value = {'axis': 0.0,        'wall': T_wall,     'inlet': T_in,        'outlet': 0.0}

T0 = np.ones((NR, NZ)) * T_in

settings = EnergySettings(dt=1.0, max_iters=5000, min_iters=50, tol=1e-4, print_every=50)
solver = EnergySolver(grid, fvm, vr, settings)

print("="*70)
print("ШАГ 01: ЭНЕРГИЯ (без импульса/давления)")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м; m_dot=5 г/мин, ρ_in={rho_in:.1f} кг/м³")
print("="*70)

T, res_all, res_int = solver.solve_pseudo_transient(T0, ur_face, uz_face, bc_type, bc_value)

# 1) Сходимость (две метрики)
plt.figure(figsize=(6,4))
plt.semilogy(res_all, lw=1.5, label="max|ΔT| (всё поле)")
plt.semilogy(res_int, lw=2.0, label="max|ΔT| (внутр.)")
plt.xlabel("Итерация"); plt.ylabel("K")
plt.title("Сходимость солвера энергии (псевдо-временной)")
plt.grid(True, which='both', alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "energy_residuals.png", dpi=140); plt.show()

# 2) Поле T(r,z)
Z, R = np.meshgrid(grid.z_centers, grid.r_centers)
plt.figure(figsize=(7.2,4.8))
im = plt.contourf(Z*100, R*1000, T-273.15, 22, cmap="hot")
plt.colorbar(im, label="T, °C")
plt.xlabel("z, см"); plt.ylabel("r, мм")
plt.title("Температурное поле после сходимости")
plt.tight_layout(); plt.savefig(OUT_DIR / "energy_T_field.png", dpi=140); plt.show()

# 3) Профиль средней по r температуры
T_mean_r = T.mean(axis=0)
plt.figure(figsize=(6,4))
plt.plot(T_mean_r-273.15, grid.z_centers, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel("T̄(r), °C"); plt.ylabel("z, м")
plt.title("Профиль температуры (среднее по r)")
plt.tight_layout(); plt.savefig(OUT_DIR / "energy_T_profile.png", dpi=140); plt.show()

print(f"Сохранено в: {OUT_DIR}")
