"""Тест для массового расхода на входе."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from boundary_conditions.bc_handler import BoundaryConditionHandler


class TestMassFlowBC:
    """Тесты для граничного условия по массовому расходу."""

    @pytest.fixture
    def bc_handler(self):
        return BoundaryConditionHandler(nr=7, nz=126)

    def test_mass_flow_to_velocity(self, bc_handler):
        """Тест преобразования массового расхода в скорость."""
        # Параметры
        m_dot = 8.333e-5  # кг/с (5 г/мин)
        rho_in = 900.0    # кг/м³
        radius = 0.0301   # м
        inlet_area = np.pi * radius**2

        # Задаём массовый расход
        v_in = bc_handler.set_mass_flow_inlet(m_dot, rho_in, inlet_area)

        # Проверяем скорость
        expected_v = m_dot / (rho_in * inlet_area)
        assert np.isclose(v_in, expected_v)
        assert np.isclose(bc_handler.velocity_z_bc['inlet'].value, expected_v)

    def test_mass_flux_consistency(self, bc_handler):
        """Тест согласованности массового расхода."""
        # Параметры
        m_dot = 8.333e-5  # кг/с
        rho_in = 900.0
        radius = 0.0301
        inlet_area = np.pi * radius**2

        # Задаём через массовый расход
        bc_handler.set_mass_flow_inlet(m_dot, rho_in, inlet_area)

        # Получаем обратно массовый расход
        calculated_m_dot = bc_handler.get_inlet_mass_flux(rho_in, inlet_area)

        # Должны совпадать
        assert np.isclose(calculated_m_dot, m_dot, rtol=1e-10)

    def test_zero_protection(self, bc_handler):
        """Тест защиты от деления на ноль."""
        m_dot = 8.333e-5

        # Нулевая плотность
        v_in = bc_handler.set_mass_flow_inlet(m_dot, 0.0, 1.0)
        assert v_in > 1e6  # Очень большая скорость

        # Нулевая площадь
        v_in = bc_handler.set_mass_flow_inlet(m_dot, 900.0, 0.0)
        assert v_in > 1e6  # Очень большая скорость

    def test_realistic_values(self, bc_handler):
        """Тест с реалистичными значениями из статьи."""
        # Параметры из статьи
        m_dot = 5.0 / 60 / 1000  # 5 г/мин → кг/с
        rho_vr = 900.0  # плотность VR при 370°C
        diameter = 0.0602  # м (согласованный)
        inlet_area = np.pi * (diameter/2)**2

        v_in = bc_handler.set_mass_flow_inlet(m_dot, rho_vr, inlet_area)

        # Проверяем разумность скорости
        # При малом расходе (5 г/мин) скорость очень мала
        assert 1e-5 < v_in < 0.1  # Скорость в физически разумных пределах

        # Проверяем точное значение
        expected_v = m_dot / (rho_vr * inlet_area)
        assert np.isclose(v_in, expected_v)
        print(f"\nСкорость на входе: {v_in:.2e} м/с")

        # Проверяем число Рейнольдса
        mu_vr = 0.1  # Па·с (примерная вязкость при 370°C)
        Re = rho_vr * v_in * diameter / mu_vr
        assert Re < 100  # Ламинарный режим
        print(f"Число Рейнольдса: {Re:.2f}")