"""Тесты для модуля генерации сетки."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mesh.grid_2d import GridGeometry, AxiSymmetricGrid


class TestGridGeometry:
    """Тесты для класса GridGeometry."""

    def test_valid_geometry(self):
        """Тест создания корректной геометрии."""
        geom = GridGeometry(nr=10, nz=20, radius=0.5, height=1.0)
        assert geom.nr == 10
        assert geom.nz == 20
        assert geom.radius == 0.5
        assert geom.height == 1.0

    def test_invalid_cells(self):
        """Тест валидации количества ячеек."""
        with pytest.raises(ValueError):
            GridGeometry(nr=0, nz=10, radius=0.5, height=1.0)

        with pytest.raises(ValueError):
            GridGeometry(nr=10, nz=-5, radius=0.5, height=1.0)

    def test_invalid_dimensions(self):
        """Тест валидации размеров."""
        with pytest.raises(ValueError):
            GridGeometry(nr=10, nz=20, radius=-0.5, height=1.0)

        with pytest.raises(ValueError):
            GridGeometry(nr=10, nz=20, radius=0.5, height=0)


class TestAxiSymmetricGrid:
    """Тесты для класса AxiSymmetricGrid."""

    @pytest.fixture
    def grid(self):
        """Создание тестовой сетки."""
        geom = GridGeometry(nr=7, nz=126, radius=0.0301, height=0.5692)
        return AxiSymmetricGrid(geom)

    @pytest.fixture
    def small_grid(self):
        """Создание маленькой тестовой сетки для детальных проверок."""
        geom = GridGeometry(nr=3, nz=4, radius=0.3, height=0.4)
        return AxiSymmetricGrid(geom)

    def test_grid_creation(self, grid):
        """Тест создания сетки."""
        assert grid.nr == 7
        assert grid.nz == 126
        assert grid.radius == 0.0301
        assert grid.height == 0.5692

        # Проверка шагов сетки
        assert np.isclose(grid.dr, 0.0301 / 7)
        assert np.isclose(grid.dz, 0.5692 / 126)

    def test_coordinates(self, small_grid):
        """Тест создания координатных массивов."""
        # Проверка размеров
        assert small_grid.r_centers.shape == (3,)
        assert small_grid.z_centers.shape == (4,)
        assert small_grid.r_faces.shape == (4,)
        assert small_grid.z_faces.shape == (5,)

        # Проверка значений
        assert np.isclose(small_grid.r_centers[0], 0.05)  # dr/2
        assert np.isclose(small_grid.r_centers[-1], 0.25)  # radius - dr/2
        assert np.isclose(small_grid.z_centers[0], 0.05)  # dz/2
        assert np.isclose(small_grid.z_centers[-1], 0.35)  # height - dz/2

        # Проверка граней
        assert small_grid.r_faces[0] == 0
        assert small_grid.r_faces[-1] == 0.3
        assert small_grid.z_faces[0] == 0
        assert small_grid.z_faces[-1] == 0.4

    def test_volumes(self, small_grid):
        """Тест расчёта объёмов контрольных объёмов."""
        volumes = small_grid.get_cell_volumes()
        assert volumes.shape == (3, 4)

        # Объёмы должны увеличиваться с радиусом
        assert np.all(volumes[1, :] > volumes[0, :])
        assert np.all(volumes[2, :] > volumes[1, :])

        # Проверка формулы V = 2*pi*r*dr*dz
        r0 = small_grid.r_centers[0]
        expected_vol = 2 * np.pi * r0 * small_grid.dr * small_grid.dz
        assert np.isclose(volumes[0, 0], expected_vol)

    def test_face_areas(self, small_grid):
        """Тест расчёта площадей граней."""
        # Радиальные грани
        areas_r = small_grid.get_face_areas('r')
        assert areas_r.shape == (4, 4)

        # Площадь на оси должна быть нулевой
        assert np.allclose(areas_r[0, :], 0)

        # Осевые грани
        areas_z = small_grid.get_face_areas('z')
        assert areas_z.shape == (3, 5)

        # Площади должны увеличиваться с радиусом
        assert np.all(areas_z[1, :] > areas_z[0, :])
        assert np.all(areas_z[2, :] > areas_z[1, :])

    def test_staggered_indices(self, grid):
        """Тест получения размеров для staggered переменных."""
        indices = grid.get_staggered_indices()

        assert indices['cell'] == (7, 126)
        assert indices['face_r'] == (8, 126)
        assert indices['face_z'] == (7, 127)

    def test_boundary_indices(self, small_grid):
        """Тест получения индексов границ."""
        boundaries = small_grid.get_boundary_indices()

        # Проверка оси симметрии
        axis_idx = boundaries['axis']
        test_field = np.zeros((3, 4))
        test_field[axis_idx] = 1
        assert np.all(test_field[0, :] == 1)
        assert np.all(test_field[1:, :] == 0)

        # Проверка стенки
        wall_idx = boundaries['wall']
        test_field = np.zeros((3, 4))
        test_field[wall_idx] = 1
        assert np.all(test_field[-1, :] == 1)
        assert np.all(test_field[:-1, :] == 0)

    def test_interpolation(self, small_grid):
        """Тест интерполяции на грани."""
        # Создаём тестовое поле
        field = np.ones((3, 4))
        field[1, :] = 2
        field[2, :] = 3

        # Интерполяция на радиальные грани
        face_r = small_grid.interpolate_to_faces(field, 'r')
        assert face_r.shape == (4, 4)
        assert np.allclose(face_r[0, :], 1)  # На оси
        assert np.allclose(face_r[1, :], 1.5)  # Между 1 и 2
        assert np.allclose(face_r[2, :], 2.5)  # Между 2 и 3
        assert np.allclose(face_r[3, :], 3)  # На стенке

        # Интерполяция на осевые грани
        field = np.ones((3, 4))
        field[:, 2] = 2
        face_z = small_grid.interpolate_to_faces(field, 'z')
        assert face_z.shape == (3, 5)
        assert np.allclose(face_z[:, 1], 1)
        assert np.allclose(face_z[:, 2], 1.5)  # Между 1 и 2

    def test_gradient(self, small_grid):
        """Тест расчёта градиента."""
        # Линейное поле в радиальном направлении
        field = np.zeros((3, 4))
        for i in range(3):
            field[i, :] = small_grid.r_centers[i]

        grad_r, grad_z = small_grid.gradient(field)

        # Градиент в r должен быть примерно 1
        assert np.allclose(grad_r[1:-1, :], 1, rtol=0.1)

        # Градиент в z должен быть 0
        assert np.allclose(grad_z, 0)

    def test_divergence(self, small_grid):
        """Тест расчёта дивергенции."""
        # Равномерное радиальное поле
        vr = np.ones((4, 4))
        vz = np.zeros((3, 5))

        div = small_grid.divergence(vr, vz)
        assert div.shape == (3, 4)

        # Для радиального потока div = 1/r * d(r*vr)/dr = vr/r
        for i in range(3):
            expected = 1 / small_grid.r_centers[i]
            assert np.allclose(div[i, :], expected, rtol=0.1)

    def test_conservation(self, small_grid):
        """Тест консервативности (сумма потоков через грани = 0)."""
        # Создаём дивергентное поле
        vr = np.random.rand(4, 4)
        vz = np.random.rand(3, 5)

        # Применяем нулевые граничные условия
        vr[0, :] = 0  # Ось симметрии
        vr[-1, :] = 0  # Стенка
        vz[:, 0] = 0  # Вход
        vz[:, -1] = 0  # Выход

        div = small_grid.divergence(vr, vz)

        # Интеграл дивергенции по объёму должен быть близок к нулю
        volumes = small_grid.get_cell_volumes()
        integral = np.sum(div * volumes)

        assert np.abs(integral) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])