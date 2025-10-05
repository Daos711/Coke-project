# src/physics/correlations.py
# -*- coding: utf-8 -*-
"""Корреляции для межфазного взаимодействия и пористой среды."""

from __future__ import annotations

import numpy as np
from typing import Union
import sys
from pathlib import Path

# чтобы можно было запускать модули из examples/*
sys.path.insert(0, str(Path(__file__).parent.parent))

NumberOrArray = Union[float, np.ndarray]


class DragCoefficients:
    """Коэффициенты межфазного сопротивления (liquid–gas)."""

    @staticmethod
    def symmetric_model(alpha_1: NumberOrArray,
                        alpha_2: NumberOrArray,
                        rho_1:  NumberOrArray,
                        rho_2:  NumberOrArray,
                        mu_1:   NumberOrArray,
                        mu_2:   NumberOrArray,
                        v_rel:  NumberOrArray) -> NumberOrArray:
        """
        Symmetric drag model (ANSYS, 2014).
        Возвращает K (кг/м³·с). Поддерживает скаляры и массивы (broadcasting).
        """
        eps = 1e-10
        alpha_1 = np.maximum(alpha_1, eps)
        alpha_2 = np.maximum(alpha_2, eps)

        d_bubble = 0.005  # м

        Re = np.maximum(rho_2 * np.abs(v_rel) * d_bubble / np.maximum(mu_2, eps), eps)

        # Cd по стандартной аппроксимации
        Cd = np.where(Re < 1000.0,
                      24.0 / Re * (1.0 + 0.15 * np.power(Re, 0.687)),
                      0.44)

        K = 0.75 * Cd * rho_2 * np.abs(v_rel) / d_bubble * alpha_1 * alpha_2
        return K


class PorousDrag:
    """
    Параметры Эргуна/Форхгеймера для пористых членов.
    Все методы векторизованы (принимают float или np.ndarray).
    Формулы:
      K  = eps^3 * d_p^2 / (180 * (1 - eps)^2)
      C2 = 1.75 * (1 - eps) / (eps^3 * d_p)
    """

    @staticmethod
    def ergun_permeability(eps: NumberOrArray, dp_particle: float = 1e-3) -> NumberOrArray:
        """Возвращает K (м²). dp_particle по умолчанию = 1e-3 (мм-класс)."""
        eps_arr = np.asarray(eps, dtype=float)
        dp = float(dp_particle)
        eps_arr = np.clip(eps_arr, 1e-6, 1.0 - 1e-6)
        K = (eps_arr**3 * dp**2) / (180.0 * (1.0 - eps_arr)**2 + 1e-30)
        return K  # та же форма, что у eps

    @staticmethod
    def ergun_inertial_c2(eps: NumberOrArray, dp_particle: float = 1e-3) -> NumberOrArray:
        """Возвращает C2 (1/м) для инерционного слагаемого Эргуна."""
        eps_arr = np.asarray(eps, dtype=float)
        dp = float(dp_particle)
        eps_arr = np.clip(eps_arr, 1e-6, 1.0 - 1e-6)
        C2 = 1.75 * (1.0 - eps_arr) / (eps_arr**3 * dp + 1e-30)
        return C2

    # Обратная совместимость: в старом коде вызывали ergun_inertial(eps[, dp])
    @staticmethod
    def ergun_inertial(eps: NumberOrArray, dp_particle: float = 1e-3) -> NumberOrArray:
        return PorousDrag.ergun_inertial_c2(eps, dp_particle)


class HeatTransfer:
    """Коэффициенты теплообмена."""

    @staticmethod
    def fluid_fluid_tomiyama(alpha_1: NumberOrArray,
                             alpha_2: NumberOrArray,
                             k_1: NumberOrArray,
                             k_2: NumberOrArray,
                             rho_1: NumberOrArray,
                             rho_2: NumberOrArray,
                             cp_1: NumberOrArray,
                             cp_2: NumberOrArray,
                             mu_2: NumberOrArray,
                             v_rel: NumberOrArray) -> NumberOrArray:
        """
        Теплообмен жидкость–газ по Томияме, возвращает H (Вт/м³·К).
        Векторизовано.
        """
        eps = 1e-10
        alpha_1 = np.maximum(alpha_1, eps)
        alpha_2 = np.maximum(alpha_2, eps)

        d_bubble = 0.005  # м
        Pr = np.maximum(cp_2 * mu_2 / np.maximum(k_2, eps), eps)
        Re = np.maximum(rho_2 * np.abs(v_rel) * d_bubble / np.maximum(mu_2, eps), eps)

        Nu = 2.0 + 0.6 * np.power(Re, 0.5) * np.power(Pr, 1.0/3.0)
        h  = Nu * k_2 / d_bubble

        a_interface = 6.0 * alpha_1 * alpha_2 / d_bubble
        H = h * a_interface
        return H

    @staticmethod
    def fluid_solid_wakao(alpha_f: NumberOrArray,
                          k_f: NumberOrArray,
                          rho_f: NumberOrArray,
                          cp_f: NumberOrArray,
                          mu_f: NumberOrArray,
                          v_f: NumberOrArray,
                          porosity: NumberOrArray,
                          d_particle: float = 0.001) -> NumberOrArray:
        """
        Теплообмен флюид–твёрдое по Wakao–Kaguei, возвращает H (Вт/м³·К).
        Векторизовано.
        """
        eps = 1e-10
        e = np.clip(np.asarray(porosity, dtype=float), 1e-6, 1.0 - 1e-6)

        Pr  = np.maximum(cp_f * mu_f / np.maximum(k_f, eps), eps)
        Rep = np.maximum(rho_f * np.abs(v_f) * d_particle / np.maximum(mu_f, eps), eps)

        Nu = 2.0 + 1.1 * np.power(Rep, 0.6) * np.power(Pr, 1.0/3.0)
        h  = Nu * k_f / d_particle

        a_solid = 6.0 * (1.0 - e) / d_particle
        H = h * a_solid * alpha_f
        return H
