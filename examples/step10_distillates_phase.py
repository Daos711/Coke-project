# -*- coding: utf-8 -*-
# ШАГ 10: Введение эйлеровой газовой фазы (Distillates, D)
# -----------------------------------------------------------------------------
# Цель: Добавить газовую фазу дистиллятов с полями α_D, v_D, ρ_D(T), μ_D(T)
# Исправлено:
#   - Плотность газа теперь реалистична (~0.6 кг/м³ при 643K)
#   - Используются насыщенности S_R, S_D с условием S_R + S_D = 1
#   - Правильные проверки баланса: α_R + α_D + α_C = 1
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization
from physics.properties import VacuumResidue, Distillables, Coke

# --------------------- результаты ---------------------
OUT = Path("results") / "solvers" / "step10_distillates_phase"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------- геометрия ----------------------
D, H = 0.0602, 0.5692
NR, NZ = 7, 126
geom = GridGeometry(nr=NR, nz=NZ, radius=D / 2, height=H)
grid = AxiSymmetricGrid(geom)
fvm = FiniteVolumeDiscretization(grid)

r = grid.r_centers
z = grid.z_centers

# --------------------- физические свойства -------------------------
vr = VacuumResidue(1)  # Жидкая фаза
dist = Distillables()  # Газовая фаза
coke = Coke()  # Твердая фаза

print("=" * 70)
print("ШАГ 10: Эйлерова газовая фаза дистиллятов (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
print(f"Сетка {NR}×{NZ}, D={D:.4f} м, H={H:.4f} м")
print("=" * 70)

# --------------------- 1. Тестирование свойств дистиллятов -----------------
print("\n1. СВОЙСТВА ФАЗЫ ДИСТИЛЛЯТОВ")
print("-" * 40)

# Диапазон температур
T_test = np.linspace(300, 800, 100)

# Свойства при разных температурах
P_atm = 101325  # Па
rho_D = dist.density(T_test, P=P_atm)
mu_D = dist.viscosity(T_test)
cp_D = dist.heat_capacity(T_test)
k_D = dist.thermal_conductivity(T_test)

# Свойства при рабочей температуре 643K
T_ref = 643.15
rho_ref = dist.density(T_ref, P_atm)
print(f"При T = {T_ref:.1f} K (370°C) и P = 1 атм:")
print(f"  Молярная масса:   {dist.molecular_weight * 1000:.1f} г/моль")
print(f"  Плотность:        {rho_ref:.3f} кг/м³")
print(f"  Вязкость:         {dist.viscosity(T_ref) * 1e6:.2f} мкПа·с")
print(f"  Теплоёмкость:     {dist.heat_capacity(T_ref):.0f} Дж/(кг·К)")
print(f"  Теплопроводность: {dist.thermal_conductivity(T_ref):.4f} Вт/(м·К)")

# Проверка реалистичности плотности
if 0.2 < rho_ref < 1.2:
    print(f"  ✓ Плотность в реалистичном диапазоне (0.2-1.2 кг/м³)")
else:
    print(f"  ✗ ВНИМАНИЕ: Плотность вне ожидаемого диапазона!")

# График свойств дистиллятов
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(T_test - 273.15, rho_D, 'b-', lw=2)
ax1.set_xlabel('Температура, °C')
ax1.set_ylabel('Плотность, кг/м³')
ax1.set_title(f'Плотность дистиллятов (M={dist.molecular_weight * 1000:.1f} г/моль)')
ax1.grid(alpha=0.3)
ax1.axvline(370, color='r', ls='--', alpha=0.5, label='T_работы')
ax1.axhline(rho_ref, color='g', ls=':', alpha=0.5, label=f'ρ={rho_ref:.3f}')
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(T_test - 273.15, mu_D * 1e6, 'g-', lw=2)
ax2.set_xlabel('Температура, °C')
ax2.set_ylabel('Вязкость, мкПа·с')
ax2.set_title('Вязкость дистиллятов (Sutherland)')
ax2.grid(alpha=0.3)
ax2.axvline(370, color='r', ls='--', alpha=0.5)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(T_test - 273.15, cp_D, 'r-', lw=2)
ax3.set_xlabel('Температура, °C')
ax3.set_ylabel('Теплоёмкость, Дж/(кг·К)')
ax3.set_title('Теплоёмкость дистиллятов')
ax3.grid(alpha=0.3)
ax3.axvline(370, color='r', ls='--', alpha=0.5)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(T_test - 273.15, k_D, 'm-', lw=2)
ax4.set_xlabel('Температура, °C')
ax4.set_ylabel('Теплопроводность, Вт/(м·К)')
ax4.set_title('Теплопроводность дистиллятов')
ax4.grid(alpha=0.3)
ax4.axvline(370, color='r', ls='--', alpha=0.5)

plt.suptitle('Физические свойства газовой фазы дистиллятов', fontsize=14)
plt.savefig(OUT / "distillates_properties.png", dpi=150, bbox_inches='tight')

# --------------------- 2. Инициализация полей с насыщенностями -----------------
print("\n2. ПОЛЯ ОБЪЁМНЫХ ДОЛЕЙ (С НАСЫЩЕННОСТЯМИ)")
print("-" * 40)

# Начальное состояние: чистый VR
S_R = np.ones((NR, NZ), dtype=float)  # Насыщенность VR = 1
S_D = np.zeros((NR, NZ), dtype=float)  # Насыщенность дистиллятов = 0
alpha_C = np.zeros((NR, NZ), dtype=float)  # Нет кокса вначале

# Моделируем частичное образование продуктов (для визуализации)
for j in range(NZ):
    progress = j / NZ

    # Кокс накапливается внизу (до 30% внизу, убывает с высотой)
    if j < NZ // 3:
        alpha_C[:, j] = 0.3 * (1 - 3 * j / NZ)

    # Дистилляты образуются и поднимаются (увеличиваются с высотой)
    if j > NZ // 4:
        # Насыщенность дистиллятов в порах
        S_D[:, j] = 0.1 + 0.3 * (j - NZ // 4) / (NZ - NZ // 4)

# Правильная нормализация насыщенностей
S_D = np.clip(S_D, 0.0, 1.0 - 1e-12)  # Ограничиваем S_D
S_R = 1.0 - S_D  # S_R + S_D = 1 всегда

# Пористость и объёмные доли
gamma = 1.0 - alpha_C  # Eq. (4): γ = 1 - α_C
alpha_R = gamma * S_R  # Объёмная доля VR
alpha_D = gamma * S_D  # Объёмная доля дистиллятов

# Проверка балансов
sum_alpha = alpha_R + alpha_D + alpha_C
sum_saturations = S_R + S_D

# Ошибки балансов
err_bulk = np.abs(sum_alpha - 1.0)
err_sat = np.abs(sum_saturations - 1.0)

print(f"ПРОВЕРКА БАЛАНСОВ:")
print(f"  max|α_R + α_D + α_C - 1| = {err_bulk.max():.3e}  (должно → 0)")
print(f"  max|S_R + S_D - 1|       = {err_sat.max():.3e}  (должно → 0)")
print(f"  Средняя ошибка объёмных долей: {err_bulk.mean():.3e}")
print(f"  Средняя ошибка насыщенностей:  {err_sat.mean():.3e}")

# Статистика
print(f"\nСредние значения по области:")
print(f"  S_R (насыщ. VR):       {S_R.mean():.3f}")
print(f"  S_D (насыщ. дист.):    {S_D.mean():.3f}")
print(f"  α_R (объём. доля VR):  {alpha_R.mean():.3f}")
print(f"  α_D (объём. доля дист.):{alpha_D.mean():.3f}")
print(f"  α_C (объём. доля кокса):{alpha_C.mean():.3f}")
print(f"  γ (пористость):        {gamma.mean():.3f}")

# --------------------- 3. Визуализация полей -----------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# Насыщенность VR
im1 = axes[0, 0].contourf(r * 100, z * 100, S_R.T, levels=20, cmap='Blues')
axes[0, 0].set_title('S_R (насыщенность VR)')
axes[0, 0].set_xlabel('r, см')
axes[0, 0].set_ylabel('z, см')
plt.colorbar(im1, ax=axes[0, 0])

# Насыщенность дистиллятов
im2 = axes[0, 1].contourf(r * 100, z * 100, S_D.T, levels=20, cmap='Oranges')
axes[0, 1].set_title('S_D (насыщенность дист.)')
axes[0, 1].set_xlabel('r, см')
axes[0, 1].set_ylabel('z, см')
plt.colorbar(im2, ax=axes[0, 1])

# Объёмная доля VR
im3 = axes[0, 2].contourf(r * 100, z * 100, alpha_R.T, levels=20, cmap='Blues')
axes[0, 2].set_title('α_R (объёмная доля VR)')
axes[0, 2].set_xlabel('r, см')
axes[0, 2].set_ylabel('z, см')
plt.colorbar(im3, ax=axes[0, 2])

# Объёмная доля дистиллятов
im4 = axes[0, 3].contourf(r * 100, z * 100, alpha_D.T, levels=20, cmap='Oranges')
axes[0, 3].set_title('α_D (объёмная доля дист.)')
axes[0, 3].set_xlabel('r, см')
axes[0, 3].set_ylabel('z, см')
plt.colorbar(im4, ax=axes[0, 3])

# Объёмная доля кокса
im5 = axes[1, 0].contourf(r * 100, z * 100, alpha_C.T, levels=20, cmap='YlOrBr')
axes[1, 0].set_title('α_C (объёмная доля кокса)')
axes[1, 0].set_xlabel('r, см')
axes[1, 0].set_ylabel('z, см')
plt.colorbar(im5, ax=axes[1, 0])

# Пористость
im6 = axes[1, 1].contourf(r * 100, z * 100, gamma.T, levels=20, cmap='Greens')
axes[1, 1].set_title('γ = 1 - α_C (пористость)')
axes[1, 1].set_xlabel('r, см')
axes[1, 1].set_ylabel('z, см')
plt.colorbar(im6, ax=axes[1, 1])

# Проверка баланса объёмных долей
im7 = axes[1, 2].contourf(r * 100, z * 100, sum_alpha.T, levels=20, cmap='coolwarm',
                          vmin=0.99, vmax=1.01)
axes[1, 2].set_title('Σα = α_R + α_D + α_C')
axes[1, 2].set_xlabel('r, см')
axes[1, 2].set_ylabel('z, см')
cbar = plt.colorbar(im7, ax=axes[1, 2])
cbar.set_label('Должно = 1')

# Ошибка баланса
im8 = axes[1, 3].contourf(r * 100, z * 100, np.log10(err_bulk.T + 1e-16),
                          levels=20, cmap='Reds')
axes[1, 3].set_title('log₁₀|Σα - 1| (ошибка баланса)')
axes[1, 3].set_xlabel('r, см')
axes[1, 3].set_ylabel('z, см')
plt.colorbar(im8, ax=axes[1, 3])

plt.suptitle('Поля насыщенностей и объёмных долей фаз', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "volume_fractions_fields.png", dpi=150, bbox_inches='tight')

# --------------------- 4. Профили вдоль оси -----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Профили насыщенностей в центре
center_idx = NR // 2
axes[0].plot(S_R[center_idx, :], z * 100, 'b-', label='S_R', lw=2)
axes[0].plot(S_D[center_idx, :], z * 100, 'orange', label='S_D', lw=2)
axes[0].plot(S_R[center_idx, :] + S_D[center_idx, :], z * 100, 'k:',
             label='S_R+S_D', lw=1)
axes[0].set_xlabel('Насыщенность')
axes[0].set_ylabel('Высота z, см')
axes[0].set_title('Насыщенности вдоль оси')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 1.1])

# Профили объёмных долей
axes[1].plot(alpha_R[center_idx, :], z * 100, 'b-', label='α_R (VR)', lw=2)
axes[1].plot(alpha_D[center_idx, :], z * 100, 'orange', label='α_D (дист.)', lw=2)
axes[1].plot(alpha_C[center_idx, :], z * 100, 'brown', label='α_C (кокс)', lw=2)
axes[1].plot(gamma[center_idx, :], z * 100, 'g--', label='γ (порист.)', lw=2)
axes[1].plot(sum_alpha[center_idx, :], z * 100, 'k:', label='Σα', lw=1)
axes[1].set_xlabel('Объёмная доля')
axes[1].set_ylabel('Высота z, см')
axes[1].set_title('Объёмные доли вдоль оси')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1.1])

# Радиальные профили α_D на разных высотах
z_positions = [NZ // 6, NZ // 3, NZ // 2, 2 * NZ // 3]
colors = ['blue', 'green', 'orange', 'red']
labels = ['z=H/6', 'z=H/3', 'z=H/2', 'z=2H/3']

for idx, (z_idx, color, label) in enumerate(zip(z_positions, colors, labels)):
    axes[2].plot(r * 100, alpha_D[:, z_idx], color=color, label=label, lw=2)

axes[2].set_xlabel('Радиус r, см')
axes[2].set_ylabel('α_D (объёмная доля дистиллятов)')
axes[2].set_title('Радиальные профили α_D')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "profiles.png", dpi=150, bbox_inches='tight')

# --------------------- 5. Сравнение свойств трёх фаз -----------------
print("\n3. СРАВНЕНИЕ СВОЙСТВ ТРЁХ ФАЗ ПРИ T=643K, P=1атм")
print("-" * 60)
print(f"{'Свойство':<30} {'VR':>10} {'Дист.':>10} {'Кокс':>10}")
print("-" * 60)
print(
    f"{'Плотность, кг/м³':<30} {vr.density(T_ref):>10.1f} {dist.density(T_ref, P_atm):>10.3f} {coke.density(T_ref):>10.0f}")
print(f"{'Вязкость, мПа·с':<30} {vr.viscosity(T_ref) * 1000:>10.3f} {dist.viscosity(T_ref) * 1000:>10.3f} {'∞':>10}")
print(
    f"{'Теплоёмкость, Дж/(кг·К)':<30} {vr.heat_capacity(T_ref):>10.0f} {dist.heat_capacity(T_ref):>10.0f} {coke.heat_capacity(T_ref):>10.0f}")
print(
    f"{'Теплопровод., Вт/(м·К)':<30} {vr.thermal_conductivity(T_ref):>10.3f} {dist.thermal_conductivity(T_ref):>10.3f} {coke.thermal_conductivity(T_ref):>10.3f}")

# --------------------- 6. Тесты корректности -----------------
print("\n4. ТЕСТЫ КОРРЕКТНОСТИ")
print("-" * 40)

# Тест 1: Баланс объёмных долей
test_bulk = err_bulk.max() < 1e-10
print(f"✓ Тест балансa Σα=1: {'ПРОЙДЕН' if test_bulk else 'ПРОВАЛЕН'} (max err={err_bulk.max():.3e})")

# Тест 2: Баланс насыщенностей
test_sat = err_sat.max() < 1e-10
print(f"✓ Тест насыщенностей S_R+S_D=1: {'ПРОЙДЕН' if test_sat else 'ПРОВАЛЕН'} (max err={err_sat.max():.3e})")

# Тест 3: Реалистичность плотности газа
test_rho = 0.2 < rho_ref < 1.2
print(f"✓ Тест плотности газа: {'ПРОЙДЕН' if test_rho else 'ПРОВАЛЕН'} (ρ={rho_ref:.3f} кг/м³)")

# Тест 4: Неотрицательность
test_positive = (alpha_R.min() >= 0) and (alpha_D.min() >= 0) and (alpha_C.min() >= 0)
print(f"✓ Тест неотрицательности: {'ПРОЙДЕН' if test_positive else 'ПРОВАЛЕН'}")

# Тест 5: Пористость в диапазоне [0,1]
test_gamma = (gamma.min() >= 0) and (gamma.max() <= 1)
print(f"✓ Тест пористости: {'ПРОЙДЕН' if test_gamma else 'ПРОВАЛЕН'} (γ∈[{gamma.min():.3f}, {gamma.max():.3f}])")

all_tests = test_bulk and test_sat and test_rho and test_positive and test_gamma
print(f"\n{'ВСЕ ТЕСТЫ ПРОЙДЕНЫ!' if all_tests else 'ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ!'}")

# Исправляем финальное сообщение - плотность реалистична только если тест пройден
if test_rho:
    density_msg = f"Плотность газа при 643K: {rho_ref:.3f} кг/м³ (реалистично)"
else:
    density_msg = f"Плотность газа при 643K: {rho_ref:.3f} кг/м³ (ТРЕБУЕТ ИСПРАВЛЕНИЯ!)"

print("\n" + "=" * 70)
print("ШАГ 10: ЗАВЕРШЁН (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
print("Газовая фаза дистиллятов успешно добавлена")
print(density_msg)
print(f"Графики сохранены в: {OUT}")
print("=" * 70)

# --------------------- 7. Сохранение данных для следующих шагов -----------------
# Сохраняем насыщенности и объёмные доли для использования в шаге 11
np.savez(OUT / "phase_fractions.npz",
         S_R=S_R, S_D=S_D,
         alpha_R=alpha_R, alpha_D=alpha_D, alpha_C=alpha_C,
         gamma=gamma,
         r=r, z=z,
         NR=NR, NZ=NZ)

print("\nДанные сохранены в phase_fractions.npz для использования в следующих шагах")

plt.show()