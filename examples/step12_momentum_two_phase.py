# -*- coding: utf-8 -*-
"""
Шаг 12: Уравнения импульса для двух фаз в пористой среде.
Тестирование SIMPLEC солвера с членами Эргуна и межфазного обмена.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from solvers.momentum_two_phase_porous import (
    SimpleC2PSettings,
    simplec_two_phase_step,
    apply_momentum_bc
)
from physics.properties import VacuumResidue
from physics.correlations import PorousDrag

# Пути для загрузки и сохранения
OUT = Path(__file__).parent.parent / 'results' / 'solvers' / 'step12_momentum'
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ШАГ 12: УРАВНЕНИЯ ИМПУЛЬСА ДЛЯ ДВУХ ФАЗ В ПОРИСТОЙ СРЕДЕ")
print("=" * 70)

# ==================== 1. ЗАГРУЗКА РЕЗУЛЬТАТОВ ШАГА 11 ====================
print("\n1. ЗАГРУЗКА ПОЛЕЙ ИЗ ШАГА 11")
print("-" * 40)

# Загружаем финальное состояние из шага 11
data_file = Path(__file__).parent.parent / 'results' / 'solvers' / 'step11_transport' / 'transport_final.npz'

if not data_file.exists():
    print("⚠ Файл transport_final.npz не найден!")
    print("Используем тестовые начальные условия...")

    # Создаём тестовые поля
    NR, NZ = 7, 126

    # Инициализация полей
    alpha_R = 0.7 * np.ones((NR, NZ))
    alpha_D = 0.2 * np.ones((NR, NZ))
    alpha_C = 0.1 * np.ones((NR, NZ))
    gamma = 1.0 - alpha_C

    # Добавим градиент для интереса
    z = np.linspace(0, 0.5692, NZ)
    for j in range(NZ):
        alpha_C[:, j] = 0.05 + 0.15 * z[j] / 0.5692  # Кокс растёт вверх
        gamma[:, j] = 1.0 - alpha_C[:, j]
        # Перенормируем флюиды
        alpha_sum = alpha_R[:, j] + alpha_D[:, j]
        alpha_R[:, j] = alpha_R[:, j] / alpha_sum * gamma[:, j] * 0.8
        alpha_D[:, j] = alpha_D[:, j] / alpha_sum * gamma[:, j] * 0.2
else:
    data = np.load(data_file)
    alpha_R = data['alpha_R']
    alpha_D = data['alpha_D']
    alpha_C = data['alpha_C']
    gamma = data['gamma']
    NR, NZ = alpha_R.shape
    print(f"Загружена сетка {NR}×{NZ}")

print(f"Средние значения:")
print(f"  α_R = {np.mean(alpha_R):.3f}")
print(f"  α_D = {np.mean(alpha_D):.3f}")
print(f"  α_C = {np.mean(alpha_C):.3f}")
print(f"  γ   = {np.mean(gamma):.3f}")

# ==================== 2. СОЗДАНИЕ СЕТКИ ====================
print("\n2. СОЗДАНИЕ СЕТКИ")
print("-" * 40)

# Геометрия реактора из статьи
D = 0.0602  # м (диаметр)
H = 0.5692  # м (высота)


# Создаём простую сетку
class SimpleGrid:
    def __init__(self, nr, nz, radius, height):
        self.nr = nr
        self.nz = nz
        self.radius = radius
        self.height = height
        self.dr = radius / nr
        self.dz = height / nz
        self.r_centers = np.linspace(self.dr / 2, radius - self.dr / 2, nr)
        self.z_centers = np.linspace(self.dz / 2, height - self.dz / 2, nz)


grid = SimpleGrid(nr=NR, nz=NZ, radius=D / 2, height=H)

print(f"Размер сетки: {NR} × {NZ}")
print(f"dr = {grid.dr:.4f} м")
print(f"dz = {grid.dz:.4f} м")

# ==================== 3. ИНИЦИАЛИЗАЦИЯ СВОЙСТВ ====================
print("\n3. ИНИЦИАЛИЗАЦИЯ ФИЗИЧЕСКИХ СВОЙСТВ")
print("-" * 40)

T = 370.0 + 273.15  # К

# Создаём объект для свойств VR
vr_props = VacuumResidue(vr_type=1)  # VR тип 1 из статьи

# Свойства VR (тяжелые остатки)
rho_R = vr_props.density(T)
mu_R = vr_props.viscosity(T)
print(f"VR: ρ = {rho_R:.1f} кг/м³, μ = {mu_R:.3e} Па·с")

# Свойства дистиллятов (газовая фаза - упрощенно)
# Для газа используем простые оценки
rho_D = 0.663  # кг/м³ при 370°C (из шага 11)
mu_D = 1.6e-5  # Па·с (типичная вязкость газа)
print(f"Дистилляты: ρ = {rho_D:.3f} кг/м³, μ = {mu_D:.3e} Па·с")

props = {
    'rho_R': rho_R,
    'rho_D': rho_D,
    'rho_C': 1500.0,  # Плотность кокса
    'mu_R': mu_R,
    'mu_D': mu_D
}

# ==================== 4. ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ СКОРОСТИ ====================
print("\n4. ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ СКОРОСТИ")
print("-" * 40)

# Инициализация скоростей
v_R_r = np.zeros((NR, NZ))
v_R_z = np.ones((NR, NZ)) * 1e-5  # Малая начальная скорость
v_D_r = np.zeros((NR, NZ))
v_D_z = np.ones((NR, NZ)) * 1e-4  # Дистилляты движутся быстрее

# Массовый расход VR на входе = 5 г/мин
inlet_mass_rate = 5e-3 / 60  # кг/с
inlet_area = np.pi * (D / 2) ** 2
inlet_velocity = inlet_mass_rate / (rho_R * inlet_area)
print(f"Скорость VR на входе: {inlet_velocity * 1000:.2f} мм/с")

# Собираем поля в словарь
fields = {
    'alpha_R': alpha_R,
    'alpha_D': alpha_D,
    'alpha_C': alpha_C,
    'gamma': gamma,
    'v_R_r': v_R_r,
    'v_R_z': v_R_z,
    'v_D_r': v_D_r,
    'v_D_z': v_D_z,
    'p': np.zeros((NR, NZ))  # Начальное давление
}

# Применяем граничные условия
fields = apply_momentum_bc(fields, grid, inlet_vr_z=inlet_velocity)

# ==================== 5. НАСТРОЙКА SIMPLEC ====================
print("\n5. НАСТРОЙКА SIMPLEC СОЛВЕРА")
print("-" * 40)

cfg = SimpleC2PSettings(
    dp_particle=1e-3,  # Диаметр частиц кокса 1 мм
    gamma_cut=1e-3,  # Минимальная пористость
    alpha_cut=1e-6,  # Минимальная объёмная доля
    max_outer=100,  # Макс. итераций
    tol_m=1e-6,  # Критерий сходимости
    relax_p=0.5,  # Релаксация давления
    relax_u=0.7,  # Релаксация скорости
    use_pea=True  # Использовать PEA
)

print(f"Диаметр частиц: {cfg.dp_particle * 1000:.1f} мм")
print(f"Релаксация: p={cfg.relax_p}, u={cfg.relax_u}")
print(f"PEA: {'включен' if cfg.use_pea else 'выключен'}")

# ==================== 6. РЕШЕНИЕ УРАВНЕНИЙ ИМПУЛЬСА ====================
print("\n6. РЕШЕНИЕ УРАВНЕНИЙ ИМПУЛЬСА")
print("-" * 40)

result = simplec_two_phase_step(grid, fields, props, cfg)

print(f"\nРезультаты SIMPLEC:")
print(f"  Сошлось: {'Да' if result['converged'] else 'Нет'}")
print(f"  Итераций: {result['iterations']}")
print(f"  Невязка массы: {result['mass_residual']:.3e}")
print(f"  Перепад давления: {result['dp_total']:.1f} Па")

# Обновляем поля
fields['p'] = result['p']
fields['v_R_r'] = result['v_R_r']
fields['v_R_z'] = result['v_R_z']
fields['v_D_r'] = result['v_D_r']
fields['v_D_z'] = result['v_D_z']

# ==================== 7. АНАЛИЗ РЕЗУЛЬТАТОВ ====================
print("\n7. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Скорости
v_R_mag = np.sqrt(fields['v_R_r'] ** 2 + fields['v_R_z'] ** 2)
v_D_mag = np.sqrt(fields['v_D_r'] ** 2 + fields['v_D_z'] ** 2)

print(f"Скорость VR: max={np.max(v_R_mag) * 1000:.3f} мм/с")
print(f"Скорость дистиллятов: max={np.max(v_D_mag) * 1000:.3f} мм/с")

# Проверка баланса массы
# Правильный расчет для осесимметричной геометрии
# Площади на входе/выходе для каждого радиального кольца
A_rings = 2 * np.pi * grid.r_centers * grid.dr  # площади колец

# Вход (j=0)
inlet_flux_R = np.sum(fields['gamma'][:, 0] * fields['alpha_R'][:, 0] *
                      fields['v_R_z'][:, 0] * A_rings) * rho_R
inlet_flux_D = np.sum(fields['gamma'][:, 0] * fields['alpha_D'][:, 0] *
                      fields['v_D_z'][:, 0] * A_rings) * rho_D

# Выход (j=-1)
outlet_flux_R = np.sum(fields['gamma'][:, -1] * fields['alpha_R'][:, -1] *
                       fields['v_R_z'][:, -1] * A_rings) * rho_R
outlet_flux_D = np.sum(fields['gamma'][:, -1] * fields['alpha_D'][:, -1] *
                       fields['v_D_z'][:, -1] * A_rings) * rho_D

print(f"\nБаланс массы:")
print(f"  VR: вход={inlet_flux_R * 1e6:.3f} г/с, выход={outlet_flux_R * 1e6:.3f} г/с")
print(f"  Дистилляты: вход={inlet_flux_D * 1e6:.3f} г/с, выход={outlet_flux_D * 1e6:.3f} г/с")

# Диагностика направления потоков
if inlet_flux_R < 0:
    print("  ⚠ Входной поток VR отрицательный - проверьте знаки скоростей!")
if outlet_flux_R < 0:
    print("  ⚠ Выходной поток VR отрицательный - возможен обратный поток")

# ==================== 8. ПОСТРОЕНИЕ ГРАФИКОВ ====================
print("\n8. ПОСТРОЕНИЕ ГРАФИКОВ")
print("-" * 40)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Поле давления
im1 = axes[0, 0].contourf(grid.r_centers * 100, grid.z_centers * 100,
                          fields['p'].T, levels=20, cmap='RdBu_r')
axes[0, 0].set_title('Давление, Па')
axes[0, 0].set_xlabel('r, см')
axes[0, 0].set_ylabel('z, см')
plt.colorbar(im1, ax=axes[0, 0])

# Скорость VR
im2 = axes[0, 1].contourf(grid.r_centers * 100, grid.z_centers * 100,
                          v_R_mag.T * 1000, levels=20, cmap='viridis')
axes[0, 1].set_title('|v_R|, мм/с')
axes[0, 1].set_xlabel('r, см')
axes[0, 1].set_ylabel('z, см')
plt.colorbar(im2, ax=axes[0, 1])

# Скорость дистиллятов
im3 = axes[0, 2].contourf(grid.r_centers * 100, grid.z_centers * 100,
                          v_D_mag.T * 1000, levels=20, cmap='plasma')
axes[0, 2].set_title('|v_D|, мм/с')
axes[0, 2].set_xlabel('r, см')
axes[0, 2].set_ylabel('z, см')
plt.colorbar(im3, ax=axes[0, 2])

# Пористость
im4 = axes[0, 3].contourf(grid.r_centers * 100, grid.z_centers * 100,
                          fields['gamma'].T, levels=20, cmap='YlGn')
axes[0, 3].set_title('Пористость γ')
axes[0, 3].set_xlabel('r, см')
axes[0, 3].set_ylabel('z, см')
plt.colorbar(im4, ax=axes[0, 3])

# Профили вдоль оси
r_axis = 0  # Индекс на оси

# Профиль давления
axes[1, 0].plot(grid.z_centers * 100, fields['p'][r_axis, :])
axes[1, 0].set_xlabel('z, см')
axes[1, 0].set_ylabel('Давление, Па')
axes[1, 0].set_title('Профиль давления вдоль оси')
axes[1, 0].grid(True)

# Профили скоростей
axes[1, 1].plot(grid.z_centers * 100, fields['v_R_z'][r_axis, :] * 1000, 'b-', label='v_R_z')
axes[1, 1].plot(grid.z_centers * 100, fields['v_D_z'][r_axis, :] * 1000, 'r-', label='v_D_z')
axes[1, 1].set_xlabel('z, см')
axes[1, 1].set_ylabel('Скорость, мм/с')
axes[1, 1].set_title('Вертикальные скорости вдоль оси')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Профили объёмных долей
axes[1, 2].plot(grid.z_centers * 100, fields['alpha_R'][r_axis, :], 'b-', label='α_R')
axes[1, 2].plot(grid.z_centers * 100, fields['alpha_D'][r_axis, :], 'r-', label='α_D')
axes[1, 2].plot(grid.z_centers * 100, fields['alpha_C'][r_axis, :], 'k-', label='α_C')
axes[1, 2].set_xlabel('z, см')
axes[1, 2].set_ylabel('Объёмная доля')
axes[1, 2].set_title('Профили фаз вдоль оси')
axes[1, 2].legend()
axes[1, 2].grid(True)

# Градиент давления vs Эргун
dp_dz = np.gradient(fields['p'][r_axis, :], grid.dz)
axes[1, 3].plot(grid.z_centers * 100, -dp_dz, 'b-', label='CFD')

# Теоретический градиент по Эргуну
k_ergun = PorousDrag.ergun_permeability(fields['gamma'][r_axis, :], cfg.dp_particle)
C2_ergun = PorousDrag.ergun_inertial_c2(fields['gamma'][r_axis, :], cfg.dp_particle)
v_avg = (fields['alpha_R'][r_axis, :] * fields['v_R_z'][r_axis, :] +
         fields['alpha_D'][r_axis, :] * fields['v_D_z'][r_axis, :]) / \
        (fields['alpha_R'][r_axis, :] + fields['alpha_D'][r_axis, :] + 1e-10)
rho_avg = (fields['alpha_R'][r_axis, :] * rho_R + fields['alpha_D'][r_axis, :] * rho_D) / \
          (fields['alpha_R'][r_axis, :] + fields['alpha_D'][r_axis, :] + 1e-10)
mu_avg = (fields['alpha_R'][r_axis, :] * mu_R + fields['alpha_D'][r_axis, :] * mu_D) / \
         (fields['alpha_R'][r_axis, :] + fields['alpha_D'][r_axis, :] + 1e-10)

dp_dz_ergun = (mu_avg * v_avg / k_ergun +
               0.5 * C2_ergun * rho_avg * v_avg * np.abs(v_avg))
axes[1, 3].plot(grid.z_centers * 100, dp_dz_ergun, 'r--', label='Эргун')
axes[1, 3].set_xlabel('z, см')
axes[1, 3].set_ylabel('-dp/dz, Па/м')
axes[1, 3].set_title('Градиент давления')
axes[1, 3].legend()
axes[1, 3].grid(True)

plt.tight_layout()
plt.savefig(OUT / 'momentum_fields.png', dpi=150)
plt.show()

# ==================== 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
print("\n9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Сохраняем поля
np.savez(OUT / "momentum_final.npz",
         alpha_R=fields['alpha_R'],
         alpha_D=fields['alpha_D'],
         alpha_C=fields['alpha_C'],
         gamma=fields['gamma'],
         v_R_r=fields['v_R_r'],
         v_R_z=fields['v_R_z'],
         v_D_r=fields['v_D_r'],
         v_D_z=fields['v_D_z'],
         p=fields['p'],
         r=grid.r_centers,
         z=grid.z_centers,
         NR=NR,
         NZ=NZ)

print(f"Результаты сохранены в: {OUT}")

# ==================== 10. ИТОГОВАЯ ПРОВЕРКА ====================
print("\n" + "=" * 70)
print("ИТОГИ ШАГА 12")
print("=" * 70)

# Критерии приёмки
checks = {
    "Невязка массы < 1e-6": result['mass_residual'] < 1e-6,
    "Перепад давления > 0": result['dp_total'] > 0,
    "Баланс массы VR": abs(inlet_flux_R - outlet_flux_R) / inlet_flux_R < 0.1,
    "min(γ) > γ_cut": np.min(fields['gamma']) > cfg.gamma_cut
}

all_passed = all(checks.values())

for criterion, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {criterion}")

if all_passed:
    print("\n✅ ШАГ 12 УСПЕШНО ЗАВЕРШЕН")
else:
    print("\n⚠ Требуется дополнительная отладка")

print("=" * 70)