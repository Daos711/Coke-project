# -*- coding: utf-8 -*-
"""
Шаг 08: Кинетика пористости / образования кокса.

Модель:
  d(α_C ρ_C)/dt = γ ρ_R Γ(T, α_C) (из непрерывности кокса)
  γ = 1 - α_C
где Γ — удельная скорость образования (1/с).
Здесь берём степенную форму с аррениусовским k(T):
  Γ = k0 * exp(-E/(R*T)) * (1 - α_C)^n.

Итоговое обновление (явный шаг):
  α_C^{new} = clip( α_C + dt * γ * ρ_R/ρ_C * Γ , 0, α_C,max )
  γ^{new} = max(γ_min, 1 - α_C^{new})

Примечание:
- ρ_R берём локально по T(r,z).
- Соляночная модель — достаточно для демонстрации сопряжения u↔T↔γ.
- Все величины массивы формы (NR, NZ).
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from physics.properties import VacuumResidue

R_GAS = 8.314462618  # Дж/(моль·К)

@dataclass
class PorositySettings:
    rho_coke: float = 1400.0
    k0: float = 1.0           # было 1e-3
    E: float = 60_000.0
    order: float = 1.0
    gamma_min: float = 1e-3
    alpha_max: float = 0.999
    T_crit: float = 710.0
    dalpha_cap_per_min: float | None = None  # None => без лимитера


class PorosityKinetics:
    """Обновляет поля α_C и γ по температуре T (из EnergySolver)."""
    def __init__(self, fluid: VacuumResidue, settings: PorositySettings):
        self.fluid = fluid
        self.set = settings

    def reaction_rate(self, T: np.ndarray, alpha_c: np.ndarray) -> np.ndarray:
        """Γ(T, α) [1/с] = k0 * exp(-E/(R*T)) * (1 - α)^n, защита от деления."""
        T_clip = np.clip(T, 250.0, 3000.0)
        theta = np.exp(-self.set.E / (R_GAS * T_clip))
        base = self.set.k0 * theta
        factor = np.power(np.clip(1.0 - alpha_c, 0.0, 1.0), self.set.order)
        return base * factor

    def advance(self, T: np.ndarray, alpha_c: np.ndarray, dt: float
                ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        rho_vr = self.fluid.density(T)
        gamma = np.clip(1.0 - alpha_c, self.set.gamma_min, 1.0)

        hot = T >= float(self.set.T_crit)
        Gamma = np.zeros_like(T, dtype=float)
        Gamma[hot] = self.reaction_rate(T[hot], alpha_c[hot])

        dalpha_dt = gamma * rho_vr * Gamma / max(self.set.rho_coke, 1e-30)

        cap = self.set.dalpha_cap_per_min
        if cap is not None and cap > 0.0:
            dalpha_dt = np.minimum(dalpha_dt, float(cap) / 60.0)

        alpha_new = np.clip(alpha_c + dt * dalpha_dt, 0.0,
                            min(self.set.alpha_max, 1.0 - self.set.gamma_min))
        gamma_new = np.clip(1.0 - alpha_new, self.set.gamma_min, 1.0)

        info = dict(dt=dt, dalpha_max=float((dt * dalpha_dt).max()),
                    alpha_mean=float(alpha_new.mean()), gamma_min=float(gamma_new.min()))
        return alpha_new, gamma_new, info
