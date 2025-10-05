# -*- coding: utf-8 -*-
# ШАГ 11: Транспортные уравнения для объёмных долей
# -----------------------------------------------------------------------------
# Реализация уравнений (2)-(4) из статьи Díaz et al.
# - Уравнение переноса для R и D с химическими источниками
# - Накопление кокса
# - Обновление пористости
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Импорт модулей проекта
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue, Distillables, Coke
from physics.kinetics import ReactionKinetics
from solvers.phase_transport import (
    TransportSettings,
    advance_one_timestep,
    check_conservation,
    apply_boundary_conditions
)

# Результаты
OUT = Path("results") / "solvers" / "step11_transport"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ШАГ 11: ТРАНСПОРТНЫЕ УРАВНЕНИЯ ДЛЯ ОБЪЁМНЫХ ДОЛЕЙ")
print("=" * 70)

# ==================== 1. ЗАГРУЗКА ДАННЫХ ИЗ ШАГА 10 ====================
print("\n1. ЗАГРУЗКА НАЧАЛЬНЫХ УСЛОВИЙ")
print("-" * 40)

# Загружаем результаты шага 10
step10_data = np.load(Path("results") / "solvers" / "step10_distillates_phase" / "phase_fractions.npz")
S_R_init = step10_data['S_R']
S_D_init = step10_data['S_D']
alpha_R_init = step10_data['alpha_R']
alpha_D_init = step10_data['alpha_D']
alpha_C_init = step10_data['alpha_C']
gamma_init = step10_data['gamma']
NR = int(step10_data['NR'])
NZ = int(step10_data['NZ'])

print(f"Загружена сетка {NR}×{NZ}")
print(f"Средние начальные значения:")
print(f"  α_R = {alpha_R_init.mean():.3f}")
print(f"  α_D = {alpha_D_init.mean():.3f}")
print(f"  α_C = {alpha_C_init.mean():.3f}")
print(f"  γ   = {gamma_init.mean():.3f}")

# ==================== 2. СОЗДАНИЕ СЕТКИ ====================
D, H = 0.0602, 0.5692
geom = GridGeometry(nr=NR, nz=NZ, radius=D / 2, height=H)
grid = AxiSymmetricGrid(geom)
fvm = FiniteVolumeDiscretization(grid)

r = grid.r_centers
z = grid.z_centers

# ==================== 3. ФИЗИЧЕСКИЕ СВОЙСТВА ====================
print("\n2. ИНИЦИАЛИЗАЦИЯ ФИЗИЧЕСКИХ СВОЙСТВ")
print("-" * 40)

# Температура (постоянная для простоты, потом из шага энергии)
T_operating = 643.15  # K (370°C)
P_operating = 101325  # Па

# Свойства фаз
vr_props = VacuumResidue(1)
dist_props = Distillables()
coke_props = Coke()

# Плотности при рабочей температуре
rho_R = vr_props.density(T_operating)
rho_D = dist_props.density(T_operating, P_operating)
rho_C = coke_props.density(T_operating)

props = {
    'rho_R': rho_R,
    'rho_D': rho_D,
    'rho_C': rho_C
}

print(f"Плотности при T={T_operating - 273.15:.0f}°C:")
print(f"  ρ_R = {rho_R:.1f} кг/м³")
print(f"  ρ_D = {rho_D:.3f} кг/м³")
print(f"  ρ_C = {rho_C:.0f} кг/м³")

# ==================== 4. КИНЕТИКА ====================
print("\n3. НАСТРОЙКА КИНЕТИКИ")
print("-" * 40)

kinetics = ReactionKinetics(vr_type=1)  # Используем VR type 1
print(f"Модель кинетики: переменный порядок")
print(f"VR тип: {kinetics.vr_type}")

# Получаем доли продуктов из метода source_terms
# В методе source_terms используется f_coke=0.3 и f_dist=0.7
xi_coke = 0.3  # Из kinetics.py
xi_dist = 0.7  # Из kinetics.py

# ==================== 5. ПОЛЯ СКОРОСТЕЙ ====================
print("\n4. ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ СКОРОСТЕЙ")
print("-" * 40)

# Простое поле скоростей для начала (малые значения)
# Позже возьмем из SIMPLEC решателя
v_R_r = np.zeros((NR, NZ))
v_R_z = np.ones((NR, NZ)) * 0.001  # 1 мм/с вверх
v_D_r = np.zeros((NR, NZ))
v_D_z = np.ones((NR, NZ)) * 0.002  # 2 мм/с вверх (газ быстрее)

# Граничное условие: вход VR снизу
inlet_mass_rate = 5e-3 / 60  # 5 г/мин = 5e-3/60 кг/с
inlet_area = np.pi * (D / 2) ** 2
inlet_velocity = inlet_mass_rate / (rho_R * inlet_area)
v_R_z[:, 0] = inlet_velocity

print(f"Скорость на входе: {inlet_velocity * 1000:.2f} мм/с")
print(f"Массовый расход VR: {inlet_mass_rate * 60 * 1000:.1f} г/мин")

# ==================== 6. НАЧАЛЬНЫЕ ПОЛЯ ====================
# Температурные поля (упрощенно - линейный градиент)
T_R = np.ones((NR, NZ)) * T_operating
T_D = np.ones((NR, NZ)) * T_operating

# Небольшой градиент температуры по высоте
for j in range(NZ):
    T_factor = 1.0 - 0.05 * (j / NZ)  # 5% падение температуры
    T_R[:, j] *= T_factor
    T_D[:, j] *= T_factor

# Собираем все поля
fields = {
    'alpha_R': alpha_R_init.copy(),
    'alpha_D': alpha_D_init.copy(),
    'alpha_C': alpha_C_init.copy(),
    'S_R': S_R_init.copy(),
    'S_D': S_D_init.copy(),
    'gamma': gamma_init.copy(),
    'T_R': T_R,
    'T_D': T_D,
    'v_R_r': v_R_r,
    'v_R_z': v_R_z,
    'v_D_r': v_D_r,
    'v_D_z': v_D_z,
}

# ==================== 7. НАСТРОЙКИ СОЛВЕРА ====================
cfg = TransportSettings(
    dt=0.01,  # 0.01 с как в статье
    xi_coke=xi_coke,  # Используем значение из статьи
)

print(f"\n5. ПАРАМЕТРЫ СИМУЛЯЦИИ")
print("-" * 40)
print(f"Шаг по времени: {cfg.dt} с")
print(f"Доля кокса: {cfg.xi_coke:.2f}")

# ==================== 8. ВРЕМЕННОЙ ЦИКЛ ====================
print("\n6. ЗАПУСК СИМУЛЯЦИИ")
print("-" * 40)

# Параметры симуляции
n_steps = 100  # Количество шагов
save_interval = 10  # Сохранять каждые N шагов

# История для анализа
history = {
    'time': [],
    'mass_R': [],
    'mass_D': [],
    'mass_C': [],
    'mass_total': [],
    'balance_error': [],
    'courant': [],
    'alpha_C_max': [],
    'gamma_min': []
}

# Сохраняем начальное состояние
snapshots = []
snapshots.append({
    'time': 0,
    'alpha_R': fields['alpha_R'].copy(),
    'alpha_D': fields['alpha_D'].copy(),
    'alpha_C': fields['alpha_C'].copy(),
    'gamma': fields['gamma'].copy()
})

# Начальная масса
conservation = check_conservation(fields, fields, props, grid, cfg.dt)
initial_mass = conservation['mass_total']
print(f"Начальная общая масса: {initial_mass:.6f} кг")

# Временной цикл
for step in range(1, n_steps + 1):
    time = step * cfg.dt

    # Сохраняем старые значения для проверки
    fields_old = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in fields.items()}

    # Граничные условия
    fields['rho_R'] = rho_R  # Добавляем плотность в fields для граничных условий
    fields = apply_boundary_conditions(fields, grid, inlet_mass_rate)

    # Продвижение на один шаг
    result = advance_one_timestep(fields, props, kinetics, grid, cfg)

    # Обновляем поля
    for key in ['alpha_R', 'alpha_D', 'alpha_C', 'S_R', 'S_D', 'gamma']:
        fields[key] = result[key]

    # Проверка сохранения
    conservation = check_conservation(fields_old, fields, props, grid, cfg.dt)

    # Сохраняем историю
    history['time'].append(time)
    history['mass_R'].append(conservation['mass_R'])
    history['mass_D'].append(conservation['mass_D'])
    history['mass_C'].append(conservation['mass_C'])
    history['mass_total'].append(conservation['mass_total'])
    history['balance_error'].append(result['balance_error'])
    history['courant'].append(result['courant'])
    history['alpha_C_max'].append(fields['alpha_C'].max())
    history['gamma_min'].append(fields['gamma'].min())

    # Сохраняем снимок
    if step % save_interval == 0:
        snapshots.append({
            'time': time,
            'alpha_R': fields['alpha_R'].copy(),
            'alpha_D': fields['alpha_D'].copy(),
            'alpha_C': fields['alpha_C'].copy(),
            'gamma': fields['gamma'].copy()
        })

    # Вывод прогресса
    if step % 10 == 0:
        sum_alpha = fields['alpha_R'] + fields['alpha_D'] + fields['alpha_C']
        sum_S = fields['S_R'] + fields['S_D']

        print(f"Шаг {step:4d}, t={time:.2f}с:")
        print(f"  Баланс: max|Σα-1|={np.max(np.abs(sum_alpha - 1)):.3e}, "
              f"max|ΣS-1|={np.max(np.abs(sum_S - 1)):.3e}")
        print(f"  Кокс: max={fields['alpha_C'].max():.3f}, "
              f"Пористость: min={fields['gamma'].min():.3f}")
        print(f"  Масса: ΔM/M₀={(conservation['mass_total'] - initial_mass) / initial_mass:.3e}, "
              f"Курант={result['courant']:.3f}")

print("\n" + "=" * 70)
print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")

# ==================== 9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
print("\n7. ПОСТРОЕНИЕ ГРАФИКОВ")
print("-" * 40)

# График 1: История масс
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Массы компонентов
ax = axes[0, 0]
ax.plot(history['time'], history['mass_R'], 'b-', label='VR', lw=2)
ax.plot(history['time'], history['mass_D'], 'orange', label='Дист.', lw=2)
ax.plot(history['time'], history['mass_C'], 'brown', label='Кокс', lw=2)
ax.plot(history['time'], history['mass_total'], 'k--', label='Общая', lw=1)
ax.set_xlabel('Время, с')
ax.set_ylabel('Масса, кг')
ax.set_title('Эволюция масс фаз')
ax.legend()
ax.grid(alpha=0.3)

# Ошибка баланса
ax = axes[0, 1]
ax.semilogy(history['time'], history['balance_error'], 'r-', lw=2)
ax.set_xlabel('Время, с')
ax.set_ylabel('max|Σα - 1|')
ax.set_title('Ошибка баланса объёмных долей')
ax.grid(alpha=0.3)

# Число Куранта
ax = axes[0, 2]
ax.plot(history['time'], history['courant'], 'g-', lw=2)
ax.axhline(1.0, color='r', ls='--', alpha=0.5)
ax.set_xlabel('Время, с')
ax.set_ylabel('Число Куранта')
ax.set_title('Courant = v·dt/dx')
ax.grid(alpha=0.3)

# Максимум кокса
ax = axes[1, 0]
ax.plot(history['time'], history['alpha_C_max'], 'brown', lw=2)
ax.set_xlabel('Время, с')
ax.set_ylabel('max(α_C)')
ax.set_title('Максимальная доля кокса')
ax.grid(alpha=0.3)

# Минимум пористости
ax = axes[1, 1]
ax.plot(history['time'], history['gamma_min'], 'g-', lw=2)
ax.set_xlabel('Время, с')
ax.set_ylabel('min(γ)')
ax.set_title('Минимальная пористость')
ax.grid(alpha=0.3)

# Относительное изменение массы
ax = axes[1, 2]
mass_change = [(m - initial_mass) / initial_mass * 100 for m in history['mass_total']]
ax.plot(history['time'], mass_change, 'k-', lw=2)
ax.set_xlabel('Время, с')
ax.set_ylabel('ΔM/M₀, %')
ax.set_title('Относительное изменение массы')
ax.grid(alpha=0.3)

plt.suptitle('Шаг 11: Транспорт объёмных долей - История', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "transport_history.png", dpi=150, bbox_inches='tight')

# График 2: Поля в конечный момент
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# Конечные поля
final = snapshots[-1]

# α_R
im = axes[0, 0].contourf(r * 100, z * 100, final['alpha_R'].T, levels=20, cmap='Blues')
axes[0, 0].set_title('α_R (конечное)')
axes[0, 0].set_xlabel('r, см')
axes[0, 0].set_ylabel('z, см')
plt.colorbar(im, ax=axes[0, 0])

# α_D
im = axes[0, 1].contourf(r * 100, z * 100, final['alpha_D'].T, levels=20, cmap='Oranges')
axes[0, 1].set_title('α_D (конечное)')
axes[0, 1].set_xlabel('r, см')
axes[0, 1].set_ylabel('z, см')
plt.colorbar(im, ax=axes[0, 1])

# α_C
im = axes[0, 2].contourf(r * 100, z * 100, final['alpha_C'].T, levels=20, cmap='YlOrBr')
axes[0, 2].set_title('α_C (конечное)')
axes[0, 2].set_xlabel('r, см')
axes[0, 2].set_ylabel('z, см')
plt.colorbar(im, ax=axes[0, 2])

# γ
im = axes[0, 3].contourf(r * 100, z * 100, final['gamma'].T, levels=20, cmap='Greens')
axes[0, 3].set_title('γ (конечное)')
axes[0, 3].set_xlabel('r, см')
axes[0, 3].set_ylabel('z, см')
plt.colorbar(im, ax=axes[0, 3])

# Изменения относительно начального состояния
initial = snapshots[0]

# Δα_R
delta_R = final['alpha_R'] - initial['alpha_R']
im = axes[1, 0].contourf(r * 100, z * 100, delta_R.T, levels=20, cmap='RdBu_r')
axes[1, 0].set_title('Δα_R')
axes[1, 0].set_xlabel('r, см')
axes[1, 0].set_ylabel('z, см')
plt.colorbar(im, ax=axes[1, 0])

# Δα_D
delta_D = final['alpha_D'] - initial['alpha_D']
im = axes[1, 1].contourf(r * 100, z * 100, delta_D.T, levels=20, cmap='RdBu_r')
axes[1, 1].set_title('Δα_D')
axes[1, 1].set_xlabel('r, см')
axes[1, 1].set_ylabel('z, см')
plt.colorbar(im, ax=axes[1, 1])

# Δα_C
delta_C = final['alpha_C'] - initial['alpha_C']
im = axes[1, 2].contourf(r * 100, z * 100, delta_C.T, levels=20, cmap='RdBu_r')
axes[1, 2].set_title('Δα_C')
axes[1, 2].set_xlabel('r, см')
axes[1, 2].set_ylabel('z, см')
plt.colorbar(im, ax=axes[1, 2])

# Сумма долей
sum_alpha = final['alpha_R'] + final['alpha_D'] + final['alpha_C']
im = axes[1, 3].contourf(r * 100, z * 100, sum_alpha.T, levels=20,
                         cmap='coolwarm', vmin=0.99, vmax=1.01)
axes[1, 3].set_title('Σα (должно = 1)')
axes[1, 3].set_xlabel('r, см')
axes[1, 3].set_ylabel('z, см')
plt.colorbar(im, ax=axes[1, 3])

plt.suptitle(f'Шаг 11: Поля после {final["time"]:.1f} с симуляции', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "transport_fields.png", dpi=150, bbox_inches='tight')

# График 3: Профили вдоль оси
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

center_idx = NR // 2

# Эволюция профилей α_R
ax = axes[0]
for i, snap in enumerate(snapshots[::2]):  # Каждый второй снимок
    alpha = 1.0 - i / (len(snapshots[::2]) - 1)
    ax.plot(snap['alpha_R'][center_idx, :], z * 100,
            color='blue', alpha=alpha, label=f't={snap["time"]:.1f}s' if i % 2 == 0 else '')
ax.set_xlabel('α_R')
ax.set_ylabel('Высота z, см')
ax.set_title('Эволюция профиля VR')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# Эволюция профилей α_D
ax = axes[1]
for i, snap in enumerate(snapshots[::2]):
    alpha = 1.0 - i / (len(snapshots[::2]) - 1)
    ax.plot(snap['alpha_D'][center_idx, :], z * 100,
            color='orange', alpha=alpha, label=f't={snap["time"]:.1f}s' if i % 2 == 0 else '')
ax.set_xlabel('α_D')
ax.set_ylabel('Высота z, см')
ax.set_title('Эволюция профиля дистиллятов')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# Эволюция профилей α_C
ax = axes[2]
for i, snap in enumerate(snapshots[::2]):
    alpha = 1.0 - i / (len(snapshots[::2]) - 1)
    ax.plot(snap['alpha_C'][center_idx, :], z * 100,
            color='brown', alpha=alpha, label=f't={snap["time"]:.1f}s' if i % 2 == 0 else '')
ax.set_xlabel('α_C')
ax.set_ylabel('Высота z, см')
ax.set_title('Эволюция профиля кокса')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle('Эволюция профилей вдоль оси реактора', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "transport_profiles.png", dpi=150, bbox_inches='tight')

# ==================== 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Сохраняем конечное состояние
final_data = {
    'alpha_R': final['alpha_R'],
    'alpha_D': final['alpha_D'],
    'alpha_C': final['alpha_C'],
    'gamma': final['gamma'],
    'r': r,
    'z': z,
    'NR': NR,
    'NZ': NZ,
    'final_time': final['time'],  # Переименовали time в final_time
    'history_time': history['time'],
    'history_mass_total': history['mass_total'],
    'history_balance_error': history['balance_error']
}

np.savez(OUT / "transport_final.npz", **final_data)

print(f"Результаты сохранены в: {OUT}")

# ==================== ИТОГОВАЯ СТАТИСТИКА ====================
print("\n" + "=" * 70)
print("ИТОГИ ШАГА 11")
print("=" * 70)

final_sum = final['alpha_R'] + final['alpha_D'] + final['alpha_C']
print(f"Время симуляции: {final['time']:.2f} с")
print(f"Финальный баланс: max|Σα-1| = {np.max(np.abs(final_sum - 1)):.3e}")
print(f"Изменение массы: ΔM/M₀ = {(history['mass_total'][-1] - initial_mass) / initial_mass:.3e}")
print(f"Максимальная доля кокса: {final['alpha_C'].max():.3f}")
print(f"Минимальная пористость: {final['gamma'].min():.3f}")
print(f"Среднее число Куранта: {np.mean(history['courant']):.3f}")

if np.max(np.abs(final_sum - 1)) < 1e-6:
    print("\n✅ ШАГ 11 УСПЕШНО ЗАВЕРШЕН")
else:
    print("\n⚠ Есть проблемы с балансом массы, требуется отладка")

print("=" * 70)

plt.show()