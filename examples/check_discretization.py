"""Проверка дискретизации."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from utils.helpers import ensure_directory

ensure_directory('results/discretization')

# Создаём сетку
geom = GridGeometry(nr=20, nz=40, radius=0.03, height=0.1)
grid = AxiSymmetricGrid(geom)
fvm = FiniteVolumeDiscretization(grid)

# Тестовые поля
r, z = grid.R, grid.Z

# 1. Гауссово распределение температуры
T = 500 + 200 * np.exp(-((r-0.015)**2/(0.01)**2 + (z-0.05)**2/(0.02)**2))
k_thermal = np.ones_like(T) * 0.1  # Вт/(м·К)

# 2. Поле скорости (параболический профиль)
vz_max = 0.01  # м/с
vr = np.zeros((grid.nr+1, grid.nz))
vz = np.zeros((grid.nr, grid.nz+1))
for i in range(grid.nr):
    vz[i, :] = vz_max * (1 - (grid.r_centers[i]/grid.radius)**2)

# Граничные условия
bc_type = {'axis': 'neumann', 'wall': 'dirichlet',
           'inlet': 'dirichlet', 'outlet': 'neumann'}
bc_value = {'axis': 0, 'wall': 500, 'inlet': 500, 'outlet': 0}

# Расчёт потоков
diff_flux = fvm.diffusion_term(T, k_thermal, bc_type, bc_value)
conv_flux = fvm.convection_term(T, vr, vz)

# Визуализация
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Температура (не транспонируем!)
im1 = ax1.contourf(z*100, r*1000, T, levels=20, cmap='hot')
ax1.set_ylabel('r (мм)')
ax1.set_xlabel('z (см)')
ax1.set_title('Температурное поле')
plt.colorbar(im1, ax=ax1, label='T (K)')

# Диффузионный поток
im2 = ax2.contourf(z*100, r*1000, diff_flux, levels=20, cmap='RdBu_r')
ax2.set_ylabel('r (мм)')
ax2.set_xlabel('z (см)')
ax2.set_title('Диффузионный поток')
plt.colorbar(im2, ax=ax2, label='∇·(k∇T)')

# Конвективный поток
im3 = ax3.contourf(z*100, r*1000, conv_flux, levels=20, cmap='RdBu_r')
ax3.set_ylabel('r (мм)')
ax3.set_xlabel('z (см)')
ax3.set_title('Конвективный поток')
plt.colorbar(im3, ax=ax3, label='∇·(vT)')

# Профиль скорости
ax4.plot(grid.r_centers*1000, vz[:, grid.nz//2], 'b-', linewidth=2)
ax4.set_xlabel('r (мм)')
ax4.set_ylabel('vz (м/с)')
ax4.set_title('Профиль скорости')
ax4.grid(True)

plt.suptitle('Проверка конечно-объёмной дискретизации')
plt.tight_layout()
plt.savefig('results/discretization/fvm_test.png', dpi=150)
plt.show()

# Проверка баланса
print(f"\n{'='*50}")
print("Проверка дискретизации:")
print(f"{'='*50}")
print(f"Макс. диффузионный поток: {np.abs(diff_flux).max():.2e}")
print(f"Макс. конвективный поток: {np.abs(conv_flux).max():.2e}")

# Интегральный баланс
volumes = grid.get_cell_volumes()
total_diff = np.sum(diff_flux * volumes)
total_conv = np.sum(conv_flux * volumes)
print(f"Интегральный диффузионный поток: {total_diff:.2e}")
print(f"Интегральный конвективный поток: {total_conv:.2e}")
print(f"{'='*50}")