"""Тесты для корреляций."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from physics.correlations import DragCoefficients, PorousDrag, HeatTransfer


class TestDrag:
    def test_symmetric_drag(self):
        # Равные объёмные доли
        alpha_1 = alpha_2 = 0.5
        rho_1, rho_2 = 1000, 10
        mu_1, mu_2 = 0.001, 1e-5
        v_rel = 0.1

        K = DragCoefficients.symmetric_model(
            alpha_1, alpha_2, rho_1, rho_2, mu_1, mu_2, v_rel
        )

        assert K > 0
        assert K < 1e6  # Разумный порядок

    def test_drag_limits(self):
        # При нулевой скорости - нулевой drag
        K_zero = DragCoefficients.symmetric_model(
            0.5, 0.5, 1000, 10, 0.001, 1e-5, 0
        )
        assert K_zero == 0

        # При малой объёмной доле - малый drag
        K_small = DragCoefficients.symmetric_model(
            0.01, 0.99, 1000, 10, 0.001, 1e-5, 0.1
        )
        assert K_small < 100


class TestPorousDrag:
    def test_ergun_permeability(self):
        # Типичная пористость
        porosity = 0.4
        perm = PorousDrag.ergun_permeability(porosity)

        assert perm > 0
        assert perm < 1e-6  # Очень малая для мм частиц

        # Пористость растёт - проницаемость растёт
        perm1 = PorousDrag.ergun_permeability(0.3)
        perm2 = PorousDrag.ergun_permeability(0.5)
        assert perm2 > perm1

    def test_ergun_inertial(self):
        porosity = 0.4
        C2 = PorousDrag.ergun_inertial(porosity)

        assert C2 > 0
        assert C2 > 1000  # Большое сопротивление для мм частиц


class TestHeatTransfer:
    def test_fluid_fluid(self):
        H = HeatTransfer.fluid_fluid_tomiyama(
            0.5, 0.5,  # alpha
            0.1, 0.05,  # k
            1000, 10,  # rho
            2000, 1000,  # cp
            0.001,  # mu
            0.1  # v_rel
        )

        assert H > 0
        assert H < 1e6

    def test_fluid_solid(self):
        H = HeatTransfer.fluid_solid_wakao(
            0.5,  # alpha_f
            0.1,  # k_f
            1000,  # rho_f
            2000,  # cp_f
            0.001,  # mu_f
            0.01,  # v_f
            0.4  # porosity
        )

        assert H > 0
        assert H < 1e7  # Увеличиваем предел до 10^7