"""Визуализация корреляций."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from physics.correlations import DragCoefficients, PorousDrag, HeatTransfer
from utils.helpers import ensure_directory

ensure_directory('results/correlations')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

# 1. Drag vs относительная скорость
v_rel = np.logspace(-3, 0, 50)  # 0.001 - 1 м/с
K = [DragCoefficients.symmetric_model(
    0.3, 0.7, 900, 5, 0.1, 1e-5, v
) for v in v_rel]

ax1.loglog(v_rel, K, 'b-', linewidth=2)
ax1.set_xlabel('Относительная скорость (м/с)')
ax1.set_ylabel('K (кг/м³·с)')
ax1.set_title('Межфазное сопротивление')
ax1.grid(True)

# 2. Эргун vs пористость
porosity = np.linspace(0.2, 0.8, 50)
perm = PorousDrag.ergun_permeability(porosity)
C2 = PorousDrag.ergun_inertial(porosity)

ax2_twin = ax2.twinx()
ax2.semilogy(porosity, perm*1e9, 'b-', linewidth=2, label='Проницаемость')
ax2_twin.semilogy(porosity, C2, 'r-', linewidth=2, label='C2')
ax2.set_xlabel('Пористость')
ax2.set_ylabel('Проницаемость (нм²)', color='b')
ax2_twin.set_ylabel('C2 (1/м)', color='r')
ax2.set_title('Параметры Эргуна')
ax2.grid(True)

# 3. Теплообмен жидкость-газ vs скорость
v_rel = np.linspace(0, 0.5, 50)
H_fg = [HeatTransfer.fluid_fluid_tomiyama(
    0.3, 0.7, 0.15, 0.05, 900, 5, 2000, 1000, 1e-5, v
) for v in v_rel]

ax3.plot(v_rel, np.array(H_fg)/1000, 'g-', linewidth=2)
ax3.set_xlabel('Относительная скорость (м/с)')
ax3.set_ylabel('H (кВт/м³·К)')
ax3.set_title('Теплообмен жидкость-газ')
ax3.grid(True)

# 4. Теплообмен флюид-твёрдое vs пористость
porosity = np.linspace(0.2, 0.8, 50)
H_fs = [HeatTransfer.fluid_solid_wakao(
    0.5, 0.15, 900, 2000, 0.1, 0.01, p
) for p in porosity]

ax4.plot(porosity, np.array(H_fs)/1000, 'r-', linewidth=2)
ax4.set_xlabel('Пористость')
ax4.set_ylabel('H (кВт/м³·К)')
ax4.set_title('Теплообмен флюид-кокс')
ax4.grid(True)

plt.suptitle('Корреляции для межфазного взаимодействия')
plt.tight_layout()
plt.savefig('results/correlations/correlations.png', dpi=150)
plt.show()

# Типичные значения
print("\n" + "="*50)
print("Типичные значения корреляций:")
print("="*50)
print(f"Drag (v=0.1 м/с): {DragCoefficients.symmetric_model(0.3,0.7,900,5,0.1,1e-5,0.1):.1f} кг/м³·с")
print(f"Проницаемость (γ=0.4): {PorousDrag.ergun_permeability(0.4)*1e9:.2f} нм²")
print(f"C2 (γ=0.4): {PorousDrag.ergun_inertial(0.4):.0f} 1/м")
print(f"H жидкость-газ: {HeatTransfer.fluid_fluid_tomiyama(0.3,0.7,0.15,0.05,900,5,2000,1000,1e-5,0.1)/1000:.1f} кВт/м³·К")
print("="*50)