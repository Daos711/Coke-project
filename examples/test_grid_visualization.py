"""Проверка сетки."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import matplotlib.pyplot as plt
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from utils.helpers import ensure_directory

# Создаём папку для результатов
ensure_directory('results/mesh')

# Создаём сетку с параметрами из статьи
geom = GridGeometry(nr=7, nz=126, radius=0.0301, height=0.5692)
grid = AxiSymmetricGrid(geom)

print("\n" + "="*40)
print(grid.info())
print("="*40 + "\n")

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Сетка
for i in range(0, grid.nr + 1, 2):
    ax1.axvline(grid.r_faces[i] * 1000, color='k', linewidth=0.5)
for j in range(0, grid.nz + 1, 20):
    ax1.axhline(grid.z_faces[j], color='k', linewidth=0.5)
ax1.set_xlabel('r (мм)')
ax1.set_ylabel('z (м)')
ax1.set_title('Сетка 7×126')

# Объёмы
im = ax2.contourf(grid.R * 1000, grid.Z, grid.get_cell_volumes() * 1e6, levels=20)
plt.colorbar(im, ax=ax2, label='Объём (см³)')
ax2.set_xlabel('r (мм)')
ax2.set_ylabel('z (м)')
ax2.set_title('Объёмы ячеек')

plt.tight_layout()
plt.savefig('results/mesh/grid_final.png', dpi=150)
plt.show()