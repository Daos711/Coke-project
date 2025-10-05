# -*- coding: utf-8 -*-
"""Визуализация свойств материалов."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from physics.properties import VacuumResidue, Distillables, Coke
from utils.helpers import ensure_directory

ensure_directory('results/properties')

T = np.linspace(300, 800, 100)
vr1 = VacuumResidue(1)
dist = Distillables()
coke = Coke()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

# Плотности
ax1.plot(T-273, vr1.density(T), 'b-', label='VR1', linewidth=2)
ax1.plot(T-273, dist.density(T), 'g-', label='Дистилляты', linewidth=2)
ax1.plot(T-273, [coke.density(300)]*len(T), 'r--', label='Кокс', linewidth=2)
ax1.set_xlabel('T (°C)'); ax1.set_ylabel('ρ (кг/м³)')
ax1.legend(); ax1.grid(True)

# Вязкости
ax2.semilogy(T-273, vr1.viscosity(T), 'b-', label='VR1', linewidth=2)
ax2.semilogy(T-273, dist.viscosity(T), 'g-', label='Дистилляты', linewidth=2)
ax2.set_xlabel('T (°C)'); ax2.set_ylabel('μ (Па·с)')
ax2.legend(); ax2.grid(True)

# Теплоёмкости
ax3.plot(T-273, vr1.heat_capacity(T), 'b-', label='VR1', linewidth=2)
ax3.plot(T-273, dist.heat_capacity(T), 'g-', label='Дистилляты', linewidth=2)
ax3.plot(T-273, coke.heat_capacity(T), 'r--', label='Кокс', linewidth=2)
ax3.set_xlabel('T (°C)'); ax3.set_ylabel('Cp (Дж/кг·К)')
ax3.legend(); ax3.grid(True)

# Теплопроводности
ax4.plot(T-273, vr1.thermal_conductivity(T), 'b-', label='VR1', linewidth=2)
ax4.plot(T-273, dist.thermal_conductivity(T), 'g-', label='Дистилляты', linewidth=2)
ax4.plot(T-273, [coke.thermal_conductivity(300)]*len(T), 'r--', label='Кокс', linewidth=2)
ax4.set_xlabel('T (°C)'); ax4.set_ylabel('k (Вт/м·К)')
ax4.legend(); ax4.grid(True)

plt.tight_layout()
plt.savefig('results/properties/material_properties_updated.png', dpi=150)
plt.show()

# ---------- Консольная сводка (ASCII в единицах, чтобы не падала кодировка) ----------
T_ref = 370.0 + 273.15  # 370°C
rho_vr1, mu_vr1 = float(vr1.density(T_ref)), float(vr1.viscosity(T_ref))
rho_dist, mu_dist = float(dist.density(T_ref)), float(dist.viscosity(T_ref))

print("\n" + "="*50)
print("Исправленные свойства при 370°C (643.15 K):")
print("="*50)
print(f"VR1 : rho={rho_vr1:.1f} kg/m^3, mu={mu_vr1:.6e} Pa·s")
print(f"Dist: rho={rho_dist:.3e} kg/m^3, mu={mu_dist:.3e} Pa·s")
print(f"     k={float(dist.thermal_conductivity(T_ref)):.3f} W/(m·K)")
print(f"Coke: rho={float(coke.density(T_ref)):.0f} kg/m^3, "
      f"k={float(coke.thermal_conductivity(T_ref)):.2f} W/(m·K)")
print("="*50)
