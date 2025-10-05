"""Численные схемы."""

import numpy as np


class NumericalSchemes:
    """Вспомогательные численные схемы."""

    @staticmethod
    def peclet_number(rho: float, v: float, L: float, gamma: float) -> float:
        """Число Пекле."""
        return rho * v * L / gamma

    @staticmethod
    def courant_number(v: float, dt: float, dx: float) -> float:
        """Число Куранта."""
        return v * dt / dx

    @staticmethod
    def blend_factor(peclet: float) -> float:
        """Фактор смешивания для гибридной схемы."""
        if abs(peclet) < 2:
            return 1.0  # Центральные разности
        else:
            return 0.0  # Upwind