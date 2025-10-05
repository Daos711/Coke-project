# -*- coding: utf-8 -*-
"""Визуализация граничных условий (полный скрипт)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt

from boundary_conditions.bc_handler import BoundaryConditionHandler
from utils.helpers import ensure_directory, load_config
from physics.properties import VacuumResidue

# -----------------------------------------------------------------------------
# Настройка путей/каталогов
# -----------------------------------------------------------------------------
ensure_directory('results/boundary_conditions')

# -----------------------------------------------------------------------------
# Конфигурация (чтение YAML)
# -----------------------------------------------------------------------------
cfg_path = Path(__file__).parent.parent / 'config' / 'simulation_params.yaml'
cfg = load_config(cfg_path)

D = float(cfg['geometry']['diameter'])
R = D / 2.0
A_in = float(np.pi * R * R)

T_feed = float(cfg['operating_conditions']['feed_temperature'])  # K
m_dot  = float(cfg['operating_conditions']['feed_mass_rate'])   # kg/s
T_wall = float(cfg['operating_conditions']['wall_temperature']) # K

Q_N2   = float(cfg['cooling']['nitrogen_flow_rate'])            # m^3/s
T_N2   = float(cfg['cooling']['nitrogen_temperature'])          # K
P_atm  = float(cfg['operating_conditions']['pressure'])         # Pa

# -----------------------------------------------------------------------------
# Свойства фаз, скорости на входе
# -----------------------------------------------------------------------------
# Вакуумный остаток (VR1) — ВСЁ берём из класса свойств (без хардкода)
vr_props = VacuumResidue(vr_type=1)
rho_vr = float(vr_props.density(T_feed))    # kg/m^3
mu_vr  = float(vr_props.viscosity(T_feed))  # Pa·s

# Скорость подачи VR из массового расхода
v_in = m_dot / (rho_vr * A_in + 1e-30)      # m/s

# Азот: плотность из идеального газа (R_specific = R_u / M)
R_u = 8.314462618          # J/(mol·K)
M_N2 = 0.0280134           # kg/mol
R_N2 = R_u / M_N2          # J/(kg·K)
rho_N2 = P_atm / (R_N2 * T_N2 + 1e-30)  # kg/m^3

# Скорость N2 задаём из объёмного расхода
v_N2 = Q_N2 / (A_in + 1e-30)              # m/s

# -----------------------------------------------------------------------------
# Обработчик ГУ и рабочие поля
# -----------------------------------------------------------------------------
nr, nz = 7, 20  # компактная визуализация
bc = BoundaryConditionHandler(nr=nr, nz=nz)

T = np.ones((nr, nz)) * 500.0
vr = np.zeros((nr+1, nz))      # staggered по r
vz = np.zeros((nr,   nz+1))    # staggered по z
alpha_vr = np.zeros((nr, nz))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# -----------------------------------------------------------------------------
# 1) Режим коксования (VR)
# -----------------------------------------------------------------------------
bc.set_coking_mode(feed_velocity=float(v_in), wall_temperature=T_wall)

# Применяем ГУ для коксования
bc.apply_temperature_bc(T, 'wall')
bc.apply_temperature_bc(T, 'inlet')
bc.apply_velocity_bc(vr, vz, 'inlet', 'liquid')
bc.apply_volume_fraction_bc(alpha_vr, 'inlet', 'liquid')

# Массовый и объёмный расходы для КОКСОВАНИЯ считаем до переключения режима
mass_flux_coking = bc.get_inlet_mass_flux(density=rho_vr, area=A_in)  # kg/s
vol_flow_coking  = v_in * A_in                                        # m^3/s

# Визуализация температуры (коксование)
im1 = ax1.imshow(T.T, origin='lower', aspect='auto', cmap='hot')
ax1.set_xlabel('r (ячейки)')
ax1.set_ylabel('z (ячейки)')
ax1.set_title('Температура в режиме коксования')
plt.colorbar(im1, ax=ax1, label='T (K)')

# Отметки границ
ax1.text(-0.5, nz/2, 'Ось', rotation=90, va='center')
ax1.text(nr-0.5, nz/2, f'Стенка\n{T_wall:.0f}K', rotation=90, va='center')
ax1.text(nr/2, -0.5, f'Вход {T_feed:.0f}K', ha='center')
ax1.text(nr/2, nz-0.5, 'Выход', ha='center')

# Профиль скорости на входе (из коксования)
ax2.plot(vz[:, 0], 'b-o')
ax2.set_xlabel('r (ячейки)')
ax2.set_ylabel('vz (м/с)')
ax2.set_title('Профиль скорости на входе (коксование)')
ax2.grid(True)

# -----------------------------------------------------------------------------
# 2) Режим охлаждения (N2)
# -----------------------------------------------------------------------------
bc.set_cooling_mode(nitrogen_velocity=float(v_N2), nitrogen_temperature=T_N2)

T_cool = np.ones((nr, nz)) * 600.0  # горячий кокс
bc.apply_temperature_bc(T_cool, 'wall')   # адиабата стенки
bc.apply_temperature_bc(T_cool, 'inlet')  # холодный азот на входе

# Массовый и объёмный расходы для ОХЛАЖДЕНИЯ (после переключения режима)
mass_flux_cooling = bc.get_inlet_mass_flux(density=rho_N2, area=A_in)  # kg/s
vol_flow_cooling  = v_N2 * A_in                                       # m^3/s (= Q_N2)

# Визуализация температуры (охлаждение)
im3 = ax3.imshow(T_cool.T, origin='lower', aspect='auto', cmap='hot')
ax3.set_xlabel('r (ячейки)')
ax3.set_ylabel('z (ячейки)')
ax3.set_title('Температура в режиме охлаждения')
plt.colorbar(im3, ax=ax3, label='T (K)')

ax3.text(-0.5, nz/2, 'Ось', rotation=90, va='center')
ax3.text(nr-0.5, nz/2, 'Стенка\n(адиабата)', rotation=90, va='center')
ax3.text(nr/2, -0.5, f'N2 {T_N2:.0f}K', ha='center')

# Сводка (оба режима) — оставляем Unicode в рисунке (PNG), это не консоль
info_text = f"""ГРАНИЧНЫЕ УСЛОВИЯ

Режим коксования:
- Вход: VR, T={T_feed-273.15:.0f}°C, v={v_in:.5e} м/с
- Стенка: T={T_wall-273.15:.0f}°C (фикс.)
- Выход: ∂p/∂z=0, ∂T/∂z=0
- Ось: симметрия

Режим охлаждения:
- Вход: N2, T={T_N2-273.15:.0f}°C, v={v_N2:.3f} м/с
- Стенка: ∂T/∂r=0 (адиабата)
- Выход: без изменений
- Ось: без изменений

Для всех режимов:
- P на входе: 1 атм
- No-slip на стенке"""

ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace')
ax4.axis('off')

plt.suptitle('Граничные условия реактора замедленного коксования')
plt.tight_layout()
plt.savefig('results/boundary_conditions/bc_overview.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# Консольный вывод — два режима отдельно и корректно (ASCII в единицах)
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("Коксование (из BC и свойств):")
print("="*50)
print(f"Температура подачи: {T_feed:.2f} K")
print(f"Плотность VR:       {rho_vr:.1f} kg/m^3")
print(f"Вязкость VR:        {mu_vr:.6e} Pa·s")
print(f"Диаметр входа:      {D:.4f} m")
print(f"Площадь входа:      {A_in:.6f} m^2")
print(f"Скорость подачи:    {v_in:.5e} m/s")
print(f"Массовый расход:    {mass_flux_coking*1e3:.3f} g/s")
print(f"Объемный расход:    {vol_flow_coking*1e6:.3f} cm^3/s")
print("="*50)

print("\n" + "="*50)
print("Охлаждение (из BC и свойств N2):")
print("="*50)
print(f"Температура N2:     {T_N2:.2f} K")
print(f"Плотность N2:       {rho_N2:.3f} kg/m^3")
print(f"Скорость N2:        {v_N2:.3f} m/s")
print(f"Массовый расход:    {mass_flux_cooling*1e3:.3f} g/s")
print(f"Объемный расход:    {vol_flow_cooling*1e6:.3f} cm^3/s  (из YAML: {Q_N2*1e6:.3f})")
print("="*50 + "\n")
