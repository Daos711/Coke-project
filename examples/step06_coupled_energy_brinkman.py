# -*- coding: utf-8 -*-
# ШАГ 06: СВЯЗКА Energy ↔ Brinkman+Ergun (осевые p(z), u(z), внешний цикл)

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

# ---------------- Папка результатов ----------------
OUT = Path("results") / "solvers" / "step06_coupled_energy_brinkman"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Геометрия/сетка ------------------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D/2, height=H)
grid = AxiSymmetricGrid(geom)
fvm  = FiniteVolumeDiscretization(grid)

r = grid.r_centers            # (NR,)
z = grid.z_centers            # (NZ,)
A_total = np.pi * (D/2)**2

# ---------------- Физика/режим ---------------------
vr = VacuumResidue(1)
T_in, T_wall = 370.0 + 273.15, 510.0 + 273.15     # K
m_dot = 5.0/1000.0/60.0                           # кг/с
gamma = np.ones((NR, NZ))                         # пока пустой барабан (ε≈1)

# ---------------- Вспом. функции -------------------
def free_area_col(gamma_col: np.ndarray) -> float:
    g = float(np.clip(np.mean(gamma_col), 1e-3, 1.0))
    return g * A_total

def build_faces_from_plug(mdot: float, rho_col: float, gamma_col: np.ndarray):
    """Собираем скорости на гранях из осевой plug-скорости для конвекции энергии."""
    uz_face = np.zeros((NR, NZ+1))
    ur_face = np.zeros((NR+1, NZ))
    v = mdot / (rho_col * free_area_col(gamma_col))
    uz_face[:] = v
    return ur_face, uz_face, v

# ---------------- Инициализация --------------------
T = np.ones((NR, NZ)) * T_in
bc_type  = {'axis':'neumann','wall':'dirichlet','inlet':'dirichlet','outlet':'neumann'}
bc_value = {'axis':0.0,       'wall':T_wall,     'inlet':T_in,       'outlet':0.0}

E = EnergySolver(grid, fvm, vr, EnergySettings(
    dt=1.0, max_iters=3000, min_iters=50, tol=2e-4, print_every=100
))

M = MomentumBrinkman(
    grid=grid, fluid=vr,
    settings=MomentumSettings(
        p_outlet=0.0,
        diameter=D,
        dp_particle=1e-3,   # характерный размер частиц для Ergun
        gamma_min=1e-3,
        rho_ref_mode="inlet"
    )
)

# ---------------- Внешние итерации -----------------
max_outer = 6
hist_T, hist_dp = [], []
dpdz_prev_mean = None

for it in range(1, max_outer+1):
    # плотность для задания конвекции на текущем шаге
    rho_col = vr.density(T.mean(axis=0)).mean()
    ur_face, uz_face, v_plug = build_faces_from_plug(m_dot, rho_col, gamma)

    # (1) Энергия — псевдовременной прогон (как в шаге 4)
    T_new, resid, _ = E.solve_pseudo_transient(T, ur_face, uz_face, bc_type, bc_value)
    dT = float(np.max(np.abs(T_new - T)))
    T = T_new.copy()
    hist_T.append(dT)

    # (2) Моментум вдоль оси (Eq.5)
    p, v_col, dpdz, info = M.compute(T=T, gamma=gamma, m_dot=m_dot)

    # приведём p и dpdz к осевым профилям (если solver вернул 2D-поля)
    p_z    = p.mean(axis=0)    if isinstance(p, np.ndarray) and p.ndim    == 2 else np.asarray(p)
    dpdz_z = dpdz.mean(axis=0) if isinstance(dpdz, np.ndarray) and dpdz.ndim == 2 else np.asarray(dpdz)

    dpdz_mean = float(np.mean(dpdz_z))
    hist_dp.append(abs(dpdz_mean - (dpdz_prev_mean if dpdz_prev_mean is not None else dpdz_mean)))
    dpdz_prev_mean = dpdz_mean

    print(f"[Outer {it}/{max_outer}] max|ΔT|={dT:.3e} K, ⟨dp/dz⟩={dpdz_mean:.3e} Pa/m, v_plug≈{v_plug:.3e} m/s")

# ---------------- Графики/итоги --------------------
# 1) Сходимость внешних итераций (энергия)
plt.figure(figsize=(7,4))
plt.semilogy(range(1, len(hist_T)+1), hist_T, 'o-')
plt.grid(alpha=0.3); plt.xlabel('внешняя итерация'); plt.ylabel('max|ΔT|, K')
plt.title('Сходимость «энергия ↔ Brinkman»')
plt.tight_layout(); plt.savefig(OUT/'outer_convergence_T.png', dpi=150)

# 2) Профиль -dp/dz
plt.figure(figsize=(7,4))
plt.plot(-dpdz_z/1e3, z, lw=2, label='-dp/dz (Eq.5)')
plt.gca().invert_yaxis(); plt.grid(alpha=0.3); plt.legend()
plt.xlabel('dp/dz, кПа/м'); plt.ylabel('z, м')
plt.title('Профиль -dp/dz (Brinkman + Ergun, 1D)')
plt.tight_layout(); plt.savefig(OUT/'dpdz_profile.png', dpi=150)

# 3) Давление p(z)
plt.figure(figsize=(7,4))
plt.plot(p_z/1e3, z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('p, кПа'); plt.ylabel('z, м')
plt.title('Давление p(z) (p(H)=0)')
plt.tight_layout(); plt.savefig(OUT/'p_profile.png', dpi=150)

# 4) Средняя по радиусу T(z)
plt.figure(figsize=(7,4))
plt.plot(T.mean(axis=0), z, lw=2)
plt.gca().invert_yaxis(); plt.grid(alpha=0.3)
plt.xlabel('T(z), K'); plt.ylabel('z, м')
plt.title('Средняя температура по радиусу')
plt.tight_layout(); plt.savefig(OUT/'T_axial_mean.png', dpi=150)

# 5) Поле температуры
plt.figure(figsize=(10,5.5))
im = plt.imshow(T, origin='lower', aspect='auto',
                extent=[z.min(), z.max(), r.min(), r.max()], cmap='hot')
plt.colorbar(im, label='T, K')
plt.xlabel('z, м'); plt.ylabel('r, м')
plt.title('Температура после сопряжённого расчёта (шаг 6)')
plt.tight_layout(); plt.savefig(OUT/'T_field.png', dpi=150)

plt.show()

# краткое резюме
delta_p = np.trapezoid(dpdz_z, z)  # заменить trapz на trapezoid
print("="*70)
print("ШАГ 06: сопряжение Energy ↔ Brinkman+Ergun (осевая модель)")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м; ṁ={m_dot*1e3*60:.2f} г/мин")
print(f"Итог: ⟨dp/dz⟩={np.mean(dpdz_z):.3e} Па/м,  Δp≈{delta_p:.3e} Па")
print(f"Рисунки в: {OUT}")
print("="*70)
