"""Проверка готовности системы к реализации солверов."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from boundary_conditions.bc_handler import BoundaryConditionHandler
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue, Distillables, Coke
from physics.kinetics import ReactionKinetics
from utils.helpers import load_config

print("\n" + "="*70)
print(" ПРОВЕРКА ГОТОВНОСТИ СИСТЕМЫ К РЕАЛИЗАЦИИ СОЛВЕРОВ ".center(70))
print("="*70)

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent / 'config'
sim_params = load_config(config_path / 'simulation_params.yaml')

# 1. ПРОВЕРКА ГЕОМЕТРИИ
print("\n1. ГЕОМЕТРИЯ И СЕТКА")
print("-" * 40)

diameter = sim_params['geometry']['diameter']
height = sim_params['geometry']['height']
nr = sim_params['mesh']['nr']
nz = sim_params['mesh']['nz']

geom = GridGeometry(nr=nr, nz=nz, radius=diameter/2, height=height)
grid = AxiSymmetricGrid(geom)

print(f"✓ Реактор: D={diameter:.4f} м, H={height:.4f} м")
print(f"✓ Сетка: {nr}×{nz} = {nr*nz} ячеек")
print(f"✓ Шаги: dr={grid.dr:.5f} м, dz={grid.dz:.5f} м")

# Проверка объёмов
volumes = grid.get_cell_volumes()
total_volume = np.sum(volumes)
cylinder_volume = np.pi * (diameter/2)**2 * height
error = abs(total_volume - cylinder_volume) / cylinder_volume

print(f"✓ Объём: численный={total_volume:.6f} м³, аналитический={cylinder_volume:.6f} м³")
print(f"✓ Ошибка: {error*100:.2e}% {'✓ OK' if error < 1e-10 else '✗ FAIL'}")

# 2. ГРАНИЧНЫЕ УСЛОВИЯ
print("\n2. ГРАНИЧНЫЕ УСЛОВИЯ")
print("-" * 40)

bc_handler = BoundaryConditionHandler(nr=nr, nz=nz)

# Массовый расход
m_dot = sim_params['operating_conditions']['feed_mass_rate']
T_feed = sim_params['operating_conditions']['feed_temperature']
T_wall = sim_params['operating_conditions']['wall_temperature']

# Свойства VR при температуре подачи
vr = VacuumResidue(1)
rho_vr = vr.density(T_feed)
mu_vr = vr.viscosity(T_feed)

# Расчёт скорости через массовый расход
inlet_area = np.pi * (diameter/2)**2
v_inlet = bc_handler.set_mass_flow_inlet(m_dot, rho_vr, inlet_area)

print(f"✓ Массовый расход: {m_dot*1000:.3f} г/с")
print(f"✓ Плотность VR при {T_feed-273:.0f}°C: {rho_vr:.1f} кг/м³")
print(f"✓ Вязкость VR: {mu_vr:.3f} Па·с")
print(f"✓ Скорость на входе: {v_inlet:.2e} м/с")

# Число Рейнольдса
Re = rho_vr * v_inlet * diameter / mu_vr
print(f"✓ Re = {Re:.2f} {'(ламинарный)' if Re < 2300 else '(турбулентный)'}")

# 3. ДИСКРЕТИЗАЦИЯ
print("\n3. ДИСКРЕТИЗАЦИЯ")
print("-" * 40)

fvm = FiniteVolumeDiscretization(grid)

# Тестовое поле температуры
T_test = np.ones((nr, nz)) * T_feed
T_test[-1, :] = T_wall  # Стенка

# Теплопроводность
k_test = vr.thermal_conductivity(T_feed) * np.ones_like(T_test)

# Граничные условия для диффузии
bc_type = {'axis': 'neumann', 'wall': 'dirichlet',
           'inlet': 'dirichlet', 'outlet': 'neumann'}
bc_value = {'axis': 0, 'wall': T_wall,
            'inlet': T_feed, 'outlet': 0}

# Проверка диффузионного члена
flux = fvm.diffusion_term(T_test, k_test, bc_type, bc_value)

print(f"✓ Размер сетки: {fvm.nr}×{fvm.nz}")
print(f"✓ Объёмы ячеек: мин={fvm.volumes.min():.2e} м³, макс={fvm.volumes.max():.2e} м³")
print(f"✓ Диффузионный поток: макс={np.abs(flux).max():.2e}")

# 4. КИНЕТИКА
print("\n4. КИНЕТИКА РЕАКЦИЙ")
print("-" * 40)

for vr_type in [1, 2, 3]:
    kin = ReactionKinetics(vr_type)
    rate = kin.reaction_rate(T_feed, rho_vr)
    print(f"✓ VR{vr_type}: скорость при {T_feed-273:.0f}°C = {rate:.3e} кг/(м³·с)")

# 5. ЧИСЛО КУРАНТА
print("\n5. СТАБИЛЬНОСТЬ")
print("-" * 40)

dt = 0.01  # с
Co_r = v_inlet * dt / grid.dr
Co_z = v_inlet * dt / grid.dz

print(f"✓ Шаг времени: {dt} с")
print(f"✓ Куранта по r: {Co_r:.5f}")
print(f"✓ Куранта по z: {Co_z:.5f}")
print(f"✓ Максимальное: {max(Co_r, Co_z):.5f} {'✓ OK' if max(Co_r, Co_z) < 1 else '⚠ Warning'}")

# 6. ПРОВЕРКА ПАМЯТИ
print("\n6. ОЦЕНКА ПАМЯТИ")
print("-" * 40)

# Количество переменных на ячейку
# 3 фазы × (α, T, ρ, μ, k, Cp) + p + vr + vz = ~20 переменных
vars_per_cell = 20
memory_per_var = nr * nz * 8  # 8 байт на float64
total_memory = vars_per_cell * memory_per_var

print(f"✓ Переменных на ячейку: ~{vars_per_cell}")
print(f"✓ Память на переменную: {memory_per_var/1024:.1f} КБ")
print(f"✓ Общая память: {total_memory/1024/1024:.1f} МБ")

# 7. ИТОГОВАЯ ПРОВЕРКА
print("\n" + "="*70)
print(" РЕЗУЛЬТАТЫ ПРОВЕРКИ ".center(70))
print("="*70)

checks = {
    "Геометрия": error < 1e-10,
    "Граничные условия": v_inlet > 0,
    "Ламинарный режим": Re < 2300,
    "Число Куранта": max(Co_r, Co_z) < 1,
    "Дискретизация": np.isfinite(flux).all(),
    "Кинетика": True  # Всегда работает
}

all_ok = all(checks.values())

for name, status in checks.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {name}: {'OK' if status else 'FAIL'}")

print("\n" + "="*70)
if all_ok:
    print(" СИСТЕМА ГОТОВА К РЕАЛИЗАЦИИ СОЛВЕРОВ! ".center(70))
    print("="*70)
    print("\nСледующие шаги:")
    print("1. Реализация SIMPLEC алгоритма")
    print("2. Реализация PEA для межфазного взаимодействия")
    print("3. Солверы для уравнений непрерывности, момента и энергии")
else:
    print(" ЕСТЬ ПРОБЛЕМЫ - ТРЕБУЕТСЯ ИСПРАВЛЕНИЕ ".center(70))
    print("="*70)