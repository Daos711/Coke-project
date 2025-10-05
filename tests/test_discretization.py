"""Тесты для дискретизации."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid
from discretization.finite_volume import FiniteVolumeDiscretization


class TestDiscretization:

    @pytest.fixture
    def grid(self):
        geom = GridGeometry(nr=5, nz=10, radius=0.1, height=0.2)
        return AxiSymmetricGrid(geom)

    @pytest.fixture
    def fvm(self, grid):
        return FiniteVolumeDiscretization(grid)

    def test_initialization(self, fvm):
        assert fvm.nr == 5
        assert fvm.nz == 10
        assert fvm.volumes.shape == (5, 10)

    def test_diffusion_constant(self, fvm):
        # Постоянное поле - нулевая диффузия
        phi = np.ones((5, 10))
        gamma = np.ones((5, 10))

        bc_type = {'axis': 'neumann', 'wall': 'neumann',
                   'inlet': 'neumann', 'outlet': 'neumann'}
        bc_value = {'axis': 0, 'wall': 0, 'inlet': 0, 'outlet': 0}

        flux = fvm.diffusion_term(phi, gamma, bc_type, bc_value)
        assert np.allclose(flux, 0, atol=1e-10)

    def test_diffusion_linear(self, fvm):
        # Линейное поле по z
        phi = np.zeros((5, 10))
        for j in range(10):
            phi[:, j] = j

        gamma = np.ones((5, 10))

        bc_type = {'axis': 'neumann', 'wall': 'neumann',
                   'inlet': 'dirichlet', 'outlet': 'neumann'}
        bc_value = {'axis': 0, 'wall': 0, 'inlet': 0, 'outlet': 0}

        flux = fvm.diffusion_term(phi, gamma, bc_type, bc_value)

        # Внутри должен быть почти нулевой поток
        assert np.abs(flux[2, 5]) < 0.1

    def test_convection_upwind(self, fvm):
        # Проверка переноса вещества
        phi = np.zeros((5, 10))
        phi[:, 0] = 1.0  # Начальное условие: phi=1 в первом слое

        vr = np.zeros((6, 10))
        vz = np.ones((5, 11)) * 0.1  # Поток вверх

        flux = fvm.convection_term(phi, vr, vz, 'upwind')

        # В первой ячейке: уходит вверх больше, чем приходит снизу
        # flux = (flux_n - flux_s)/V < 0 (убыль массы)
        assert flux[2, 0] < 0

        # Во второй ячейке: приходит снизу, ничего не уходит вверх
        # flux > 0 (прибыль массы)
        assert flux[2, 1] > 0

        # В средних ячейках с phi=0 поток нулевой
        assert np.isclose(flux[2, 5], 0)

    def test_convection_transport(self, fvm, grid):
        # Транспорт скалярной величины
        phi = np.zeros((5, 10))
        # Задаём начальное распределение
        phi[:, 0] = 1.0  # Первый слой = 1
        phi[:, 1] = 0.8  # Второй слой = 0.8
        phi[:, 2] = 0.6  # Третий слой = 0.6

        vr = np.zeros((6, 10))
        vz = np.ones((5, 11)) * 0.1

        # Один шаг по времени
        flux = fvm.convection_term(phi, vr, vz, 'upwind')

        # Проверка физики для ячейки [2,0]:
        # Снизу приходит: 0 (граница)
        # Вверх уходит: vz * phi = 0.1 * 1.0
        # flux отрицательный (отток)
        assert flux[2, 0] < 0

        # Для ячейки [2,1]:
        # Снизу приходит: vz * phi[0] = 0.1 * 1.0
        # Вверх уходит: vz * phi[1] = 0.1 * 0.8
        # Приток больше оттока => flux > 0
        assert flux[2, 1] > 0

        # Для ячейки [2,2]:
        # Снизу: 0.1 * 0.8, вверх: 0.1 * 0.6
        # Тоже приток
        assert flux[2, 2] > 0

        # Последняя ячейка с phi=0
        assert np.isclose(flux[2, -1], 0)

    def test_mass_conservation(self, fvm, grid):
        # Проверка консервативности
        phi = np.random.rand(5, 10)
        vr = np.random.rand(6, 10) * 0.01
        vz = np.random.rand(5, 11) * 0.01

        # Нулевые ГУ
        vr[0, :] = 0  # Ось
        vr[-1, :] = 0  # Стенка
        vz[:, 0] = 0  # Вход
        vz[:, -1] = 0  # Выход

        flux = fvm.convection_term(phi, vr, vz)
        volumes = grid.get_cell_volumes()

        # Интегральный баланс
        total = np.sum(flux * volumes)
        assert np.abs(total) < 1e-10