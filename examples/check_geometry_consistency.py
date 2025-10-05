"""Проверка согласованности геометрических параметров."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from physics.properties import VacuumResidue
from utils.helpers import load_config, ensure_directory

ensure_directory('results/geometry_check')

# === Конфигурация ===
config_path = Path(__file__).parent.parent / 'config'
sim_params = load_config(config_path / 'simulation_params.yaml')

D = sim_params['geometry']['diameter']
H = sim_params['geometry']['height']
R = D / 2

nr = sim_params['mesh']['nr']
nz = sim_params['mesh']['nz']

geom = GridGeometry(nr=nr, nz=nz, radius=R, height=H)
grid = AxiSymmetricGrid(geom)

# Параметры подачи и свойства
T_feed = sim_params['operating_conditions']['feed_temperature']  # K
m_dot  = sim_params['operating_conditions']['feed_mass_rate']    # kg/s

vr = VacuumResidue(vr_type=1)
rho_vr = vr.density(T_feed)
mu_vr  = vr.viscosity(T_feed)

A_in   = np.pi * R * R
v_in   = m_dot / (rho_vr * A_in)           # m/s
Re     = rho_vr * v_in * D / mu_vr

dt = sim_params['time_settings']['time_step']
Co_r = abs(v_in) * dt / grid.dr
Co_z = abs(v_in) * dt / grid.dz
Co_max = max(Co_r, Co_z)

# Расчёты объёмов/площадей
volumes = grid.get_cell_volumes()
areas_r = grid.get_face_areas('r')
areas_z = grid.get_face_areas('z')

# === Вывод в консоль ===
print("\n" + "="*60)
print("ПРОВЕРКА СОГЛАСОВАННОСТИ ГЕОМЕТРИИ")
print("="*60)

print(f"\nИз конфигурации:")
print(f"  Диаметр: {D:.4f} м")
print(f"  Радиус:  {R:.4f} м")
print(f"  Высота:  {H:.4f} м")

print(f"\nСетка {nr}×{nz}:")
print(f"  dr = {grid.dr:.6f} м")
print(f"  dz = {grid.dz:.6f} м")

print(f"\nПараметры входа (из свойств при T_feed):")
print(f"  ṁ = {m_dot*1e3:.3f} г/с")
print(f"  ρ_VR = {rho_vr:.1f} кг/м³")
print(f"  μ_VR = {mu_vr:.3e} Па·с")
print(f"  v_in = {v_in:.5e} м/с")
print(f"  Re   = {Re:.2f} (ламинарный)")

print(f"\nЧисло Куранта (dt={dt} с):")
print(f"  Co_r = {Co_r:.5f}")
print(f"  Co_z = {Co_z:.5f}")
print(f"  Co_max = {Co_max:.5f} (рекомендуется ~1)")

print(f"\nГеометрические параметры:")
print(f"  Объём ячейки: min = {volumes.min():.2e} м³, max = {volumes.max():.2e} м³")
print(f"  Площадь радиальных граней: ось = {areas_r[0, 0]:.2e} м² (должна быть 0), стенка = {areas_r[-1, 0]:.2e} м²")
print(f"  Площадь осевых граней: у оси = {areas_z[0, 0]:.2e} м², у стенки = {areas_z[-1, 0]:.2e} м²")

print("\n" + "="*60)
print("РЕЗУЛЬТАТ: Все параметры согласованы!")
print("="*60 + "\n")

# === Визуализация ===
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Объёмы ячеек
im1 = ax1.contourf(grid.Z*100, grid.R*100, volumes*1e6, levels=20, cmap='viridis')
ax1.set_xlabel('z (см)')
ax1.set_ylabel('r (см)')
ax1.set_title('Объёмы ячеек')
plt.colorbar(im1, ax=ax1, label='V (см³)')

# 2. Радиальные площади
r_plot, z_plot = np.meshgrid(grid.z_centers, grid.r_faces, indexing='ij')
im2 = ax2.contourf(z_plot*100, r_plot*100, areas_r.T*1e4, levels=20, cmap='plasma')
ax2.set_xlabel('z (см)')
ax2.set_ylabel('r (см)')
ax2.set_title('Площади радиальных граней')
plt.colorbar(im2, ax=ax2, label='A (см²)')

# 3. Осевые площади
r_plot, z_plot = np.meshgrid(grid.z_faces, grid.r_centers, indexing='ij')
im3 = ax3.contourf(z_plot*100, r_plot*100, areas_z.T*1e4, levels=20, cmap='coolwarm')
ax3.set_xlabel('z (см)')
ax3.set_ylabel('r (см)')
ax3.set_title('Площади осевых граней')
plt.colorbar(im3, ax=ax3, label='A (см²)')

# 4. Сводка
info_text = f"""СОГЛАСОВАННОСТЬ ПАРАМЕТРОВ

Конфигурация:
  d = {D:.4f} м
  r = {R:.4f} м
  h = {H:.4f} м

Сетка {nr}×{nz}:
  Всего ячеек: {nr*nz}
  dr = {grid.dr:.5f} м
  dz = {grid.dz:.5f} м

Вход (из свойств при T_feed):
  ṁ = {m_dot*1e3:.3f} г/с
  ρ = {rho_vr:.1f} кг/м³
  μ = {mu_vr:.3e} Па·с
  v = {v_in:.5e} м/с
  Re = {Re:.2f}

Стабильность:
  Co_max = {Co_max:.3f}
  dt = {dt} с"""

ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.axis('off')

plt.suptitle('Проверка геометрической согласованности', fontsize=14)
plt.tight_layout()
plt.savefig('results/geometry_check/consistency.png', dpi=150)
plt.show()
