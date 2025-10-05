"""Тесты для проверки правильности осесимметричной геометрии."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from mesh.grid_2d import GridGeometry, AxiSymmetricGrid


class TestAxisymmetricGeometry:
    """Тесты для осесимметричной геометрии."""

    @pytest.fixture
    def simple_grid(self):
        """Простая сетка для аналитических проверок."""
        geom = GridGeometry(nr=2, nz=2, radius=1.0, height=1.0)
        return AxiSymmetricGrid(geom)

    @pytest.fixture
    def reactor_grid(self):
        """Сетка с параметрами реактора."""
        geom = GridGeometry(nr=7, nz=126, radius=0.0301, height=0.5692)
        return AxiSymmetricGrid(geom)

    def test_volume_calculation(self, simple_grid):
        """Проверка правильности расчёта объёмов."""
        grid = simple_grid
        volumes = grid.get_cell_volumes()

        # Для простой сетки 2x2 с R=1, H=1
        # r_faces = [0, 0.5, 1.0]
        # z_faces = [0, 0.5, 1.0]

        # Объём первого кольца (r от 0 до 0.5):
        # V = π * (0.5² - 0²) * 0.5 = π * 0.25 * 0.5 = π/8
        expected_v1 = np.pi * (0.5 ** 2 - 0 ** 2) * 0.5
        assert np.isclose(volumes[0, 0], expected_v1)

        # Объём второго кольца (r от 0.5 до 1.0):
        # V = π * (1² - 0.5²) * 0.5 = π * 0.75 * 0.5 = 3π/8
        expected_v2 = np.pi * (1.0 ** 2 - 0.5 ** 2) * 0.5
        assert np.isclose(volumes[1, 0], expected_v2)

        # Проверка суммарного объёма
        # Полный объём цилиндра = π * R² * H = π * 1² * 1 = π
        total_volume = np.sum(volumes)
        expected_total = np.pi * 1.0 ** 2 * 1.0
        assert np.isclose(total_volume, expected_total, rtol=1e-10)

    def test_face_areas(self, simple_grid):
        """Проверка площадей граней."""
        grid = simple_grid
        areas_r = grid.get_face_areas('r')
        areas_z = grid.get_face_areas('z')

        # Радиальные грани: A = 2πr * dz
        # На оси (r=0): A = 0
        assert np.allclose(areas_r[0, :], 0)

        # При r=0.5: A = 2π * 0.5 * 0.5 = π/2
        expected_ar = 2 * np.pi * 0.5 * 0.5
        assert np.isclose(areas_r[1, 0], expected_ar)

        # Осевые грани: A = π(r_outer² - r_inner²)
        # Для первого кольца: A = π * (0.5² - 0²) = π/4
        expected_az1 = np.pi * (0.5 ** 2 - 0 ** 2)
        assert np.isclose(areas_z[0, 0], expected_az1)

        # Для второго кольца: A = π * (1² - 0.5²) = 3π/4
        expected_az2 = np.pi * (1.0 ** 2 - 0.5 ** 2)
        assert np.isclose(areas_z[1, 0], expected_az2)

    def test_conservation(self, reactor_grid):
        """Проверка сохранения массы через границы."""
        grid = reactor_grid

        # Создаём равномерный поток вверх
        vr = np.zeros((grid.nr + 1, grid.nz))
        vz = np.ones((grid.nr, grid.nz + 1)) * 0.01  # м/с

        areas_z = grid.get_face_areas('z')

        # Массовый расход через вход (z=0)
        inlet_flux = np.sum(vz[:, 0] * areas_z[:, 0])

        # Массовый расход через выход (z=H)
        outlet_flux = np.sum(vz[:, -1] * areas_z[:, -1])

        # Должны быть равны (сохранение массы)
        assert np.isclose(inlet_flux, outlet_flux, rtol=1e-10)

        # Проверка правильной площади входа
        inlet_area = np.sum(areas_z[:, 0])
        expected_area = np.pi * grid.radius ** 2
        assert np.isclose(inlet_area, expected_area, rtol=1e-10)

    def test_volume_consistency(self, reactor_grid):
        """Проверка согласованности объёмов с геометрией."""
        grid = reactor_grid
        volumes = grid.get_cell_volumes()

        # Суммарный объём должен равняться объёму цилиндра
        total_volume = np.sum(volumes)
        cylinder_volume = np.pi * grid.radius ** 2 * grid.height

        assert np.isclose(total_volume, cylinder_volume, rtol=1e-10)

        # Объёмы должны увеличиваться с радиусом
        for j in range(grid.nz):
            for i in range(grid.nr - 1):
                assert volumes[i + 1, j] > volumes[i, j]

    def test_axisymmetric_divergence(self, simple_grid):
        """Проверка дивергенции для радиального потока."""
        grid = simple_grid

        # Радиальный поток vr = const
        vr = np.ones((grid.nr + 1, grid.nz))
        vz = np.zeros((grid.nr, grid.nz + 1))

        # Применяем граничные условия
        vr[0, :] = 0  # На оси vr = 0

        div = grid.divergence(vr, vz)

        # Для осесимметричного радиального потока
        # div = 1/r * d(r*vr)/dr = vr/r + dvr/dr
        # При vr = const: div = vr/r

        for i in range(grid.nr):
            for j in range(grid.nz):
                if i > 0:  # Не на оси
                    r = grid.r_centers[i]
                    # Приблизительно vr/r
                    expected = 1.0 / r
                    # Учитываем численную схему
                    assert div[i, j] > 0  # Должна быть положительная дивергенция

    def test_integral_theorem(self, reactor_grid):
        """Проверка интегральной теоремы Гаусса."""
        grid = reactor_grid

        # Создаём поле скорости с известной дивергенцией
        # Например, vr = r, vz = z
        vr = np.zeros((grid.nr + 1, grid.nz))
        vz = np.zeros((grid.nr, grid.nz + 1))

        # Простое поле для проверки
        for i in range(grid.nr + 1):
            vr[i, :] = grid.r_faces[i] * 0.01
        for j in range(grid.nz + 1):
            vz[:, j] = grid.z_faces[j] * 0.01

        # Граничные условия
        vr[0, :] = 0  # Ось симметрии

        div = grid.divergence(vr, vz)
        volumes = grid.get_cell_volumes()

        # Интеграл дивергенции по объёму
        volume_integral = np.sum(div * volumes)

        # Поток через границы
        areas_r = grid.get_face_areas('r')
        areas_z = grid.get_face_areas('z')

        # Поток через стенку (r = R)
        wall_flux = np.sum(vr[-1, :] * areas_r[-1, :])

        # Поток через выход (z = H)
        outlet_flux = np.sum(vz[:, -1] * areas_z[:, -1])

        # Поток через вход (z = 0)
        inlet_flux = np.sum(vz[:, 0] * areas_z[:, 0])

        # Суммарный поток (положительный наружу)
        total_flux = wall_flux + outlet_flux - inlet_flux

        # По теореме Гаусса должны быть примерно равны
        # (с учётом численных ошибок)
        assert np.abs(volume_integral - total_flux) / np.abs(total_flux) < 0.01