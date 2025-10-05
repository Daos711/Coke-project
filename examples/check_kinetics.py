"""Визуализация кинетики реакций."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from physics.kinetics import ReactionKinetics
from utils.helpers import ensure_directory

# Создаём папку
ensure_directory('results/kinetics')

# Температурный диапазон
T = np.linspace(400, 900, 500)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

# Для всех трёх типов VR
for vr_type in [1, 2, 3]:
    kin = ReactionKinetics(vr_type)

    # 1. Порядок реакции
    order = kin.reaction_order(T)
    ax1.plot(T - 273, order, linewidth=2, label=f'VR{vr_type}')

    # 2. Константы скорости
    k = np.zeros_like(T)
    for i, temp in enumerate(T):
        ord = kin.reaction_order(temp)
        k[i] = kin.rate_constant(temp, ord)
    ax2.semilogy(1000 / T, k, linewidth=2, label=f'VR{vr_type}')

    # 3. Скорость реакции при C_vr = 800 кг/м³
    C_vr = 800
    rate = kin.reaction_rate(T, C_vr)
    ax3.plot(T - 273, rate, linewidth=2, label=f'VR{vr_type}')

    # 4. Конверсия за 1 час
    conversion = 1 - np.exp(-rate / C_vr * 3600)
    ax4.plot(T - 273, conversion * 100, linewidth=2, label=f'VR{vr_type}')

# Оформление
ax1.set_xlabel('T (°C)');
ax1.set_ylabel('Порядок реакции')
ax1.grid(True);
ax1.legend()
ax1.set_ylim(0.8, 2.2)

ax2.set_xlabel('1000/T (1/K)');
ax2.set_ylabel('k (1/с или др.)')
ax2.grid(True);
ax2.legend()
ax2.invert_xaxis()

ax3.set_xlabel('T (°C)');
ax3.set_ylabel('Скорость (кг/м³·с)')
ax3.grid(True);
ax3.legend()

ax4.set_xlabel('T (°C)');
ax4.set_ylabel('Конверсия за 1 час (%)')
ax4.grid(True);
ax4.legend()
ax4.set_ylim(0, 100)

plt.suptitle('Кинетика реакций замедленного коксования')
plt.tight_layout()
plt.savefig('results/kinetics/reaction_kinetics.png', dpi=150)
plt.show()

# Вывод параметров при рабочей температуре
print(f"\n{'=' * 50}")
print("Кинетика при 370°C (643K), C_vr = 800 кг/м³:")
print(f"{'=' * 50}")
for vr_type in [1, 2, 3]:
    kin = ReactionKinetics(vr_type)
    rate = kin.reaction_rate(643, 800)
    order = kin.reaction_order(643)
    print(f"VR{vr_type}: порядок = {order}, скорость = {rate:.3e} кг/м³·с")
print(f"{'=' * 50}")