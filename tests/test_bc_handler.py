"""Тесты для граничных условий."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from boundary_conditions.bc_handler import BoundaryConditionHandler, BCType


class TestBoundaryConditions:

    @pytest.fixture
    def bc_handler(self):
        return BoundaryConditionHandler(nr=5, nz=10)

    def test_initialization(self, bc_handler):
        # Проверка инициализации
        assert bc_handler.nr == 5
        assert bc_handler.nz == 10

        # Проверка стандартных ГУ
        assert bc_handler.velocity_r_bc['axis'].bc_type == BCType.SYMMETRY
        assert bc_handler.velocity_r_bc['wall'].value == 0.0

    def test_coking_mode(self, bc_handler):
        # Настройка режима коксования
        bc_handler.set_coking_mode(feed_velocity=0.01, wall_temperature=800)

        assert bc_handler.velocity_z_bc['inlet'].value == 0.01
        assert bc_handler.temperature_bc['wall'].value == 800

    def test_cooling_mode(self, bc_handler):
        # Настройка режима охлаждения
        bc_handler.set_cooling_mode(nitrogen_velocity=0.1, nitrogen_temperature=300)

        assert bc_handler.velocity_z_bc['inlet'].value == 0.1
        assert bc_handler.temperature_bc['inlet'].value == 300
        assert bc_handler.temperature_bc['wall'].bc_type == BCType.NEUMANN

    def test_velocity_bc_axis(self, bc_handler):
        # Тест ГУ на оси симметрии
        vr = np.ones((6, 10))  # nr+1 для vr
        vz = np.ones((5, 11))  # nz+1 для vz

        bc_handler.apply_velocity_bc(vr, vz, 'axis')

        assert np.all(vr[0, :] == 0)  # vr = 0 на оси
        assert np.all(vz[0, :] == vz[1, :])  # Симметрия для vz

    def test_velocity_bc_wall(self, bc_handler):
        # Тест no-slip на стенке
        vr = np.ones((6, 10))
        vz = np.ones((5, 11))

        bc_handler.apply_velocity_bc(vr, vz, 'wall')

        assert np.all(vr[-1, :] == 0)
        assert np.all(vz[-1, :] == 0)

    def test_temperature_bc(self, bc_handler):
        # Тест температурных ГУ
        T = np.ones((5, 10)) * 400

        bc_handler.apply_temperature_bc(T, 'wall')
        assert np.all(T[-1, :] == 783.15)

        bc_handler.apply_temperature_bc(T, 'inlet')
        assert np.all(T[:, 0] == 643.15)

    def test_mass_flux(self, bc_handler):
        # Тест расчёта массового расхода
        bc_handler.set_coking_mode(feed_velocity=0.01, wall_temperature=800)

        mass_flux = bc_handler.get_inlet_mass_flux(density=900, area=0.001)
        assert np.isclose(mass_flux, 0.009)  # 900 * 0.01 * 0.001