"""
Модуль для создания 2D осесимметричной сетки для реактора замедленного коксования.

Реализует staggered grid (смещённую сетку), где:
- Давление и скалярные величины хранятся в центрах ячеек
- Радиальная скорость (vr) хранится на радиальных гранях
- Осевая скорость (vz) хранится на осевых гранях
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridGeometry:
    """Класс для хранения геометрических параметров сетки."""

    nr: int          # Количество ячеек в радиальном направлении
    nz: int          # Количество ячеек в осевом направлении
    radius: float    # Радиус реактора (м)
    height: float    # Высота реактора (м)

    def __post_init__(self):
        """Валидация параметров после инициализации."""
        if self.nr <= 0 or self.nz <= 0:
            raise ValueError("Количество ячеек должно быть положительным")
        if self.radius <= 0 or self.height <= 0:
            raise ValueError("Размеры реактора должны быть положительными")


class AxiSymmetricGrid:
    """
    Класс для 2D осесимметричной сетки с staggered расположением переменных.

    Система координат:
    - r: радиальное направление (0 - ось симметрии, radius - стенка)
    - z: осевое направление (0 - низ, height - верх)
    """

    def __init__(self, geometry: GridGeometry):
        """
        Инициализация сетки.

        Параметры
        ---------
        geometry : GridGeometry
            Геометрические параметры сетки
        """
        self.geometry = geometry
        self.nr = geometry.nr
        self.nz = geometry.nz
        self.radius = geometry.radius
        self.height = geometry.height

        # Шаги сетки
        self.dr = self.radius / self.nr
        self.dz = self.height / self.nz

        # Создание координатных массивов
        self._create_coordinates()

        # Расчёт геометрических факторов
        self._calculate_geometry_factors()

        logger.info(f"Создана сетка {self.nr}x{self.nz} для реактора "
                   f"R={self.radius:.3f}м, H={self.height:.3f}м")

    def _create_coordinates(self):
        """Создание массивов координат для центров ячеек и граней."""

        # Координаты центров ячеек
        self.r_centers = np.linspace(self.dr/2, self.radius - self.dr/2, self.nr)
        self.z_centers = np.linspace(self.dz/2, self.height - self.dz/2, self.nz)

        # Координаты граней
        self.r_faces = np.linspace(0, self.radius, self.nr + 1)
        self.z_faces = np.linspace(0, self.height, self.nz + 1)

        # 2D массивы координат центров (meshgrid)
        self.R, self.Z = np.meshgrid(self.r_centers, self.z_centers, indexing='ij')

        # 2D массивы для граней
        self.R_r_faces, self.Z_r_faces = np.meshgrid(
            self.r_faces, self.z_centers, indexing='ij'
        )
        self.R_z_faces, self.Z_z_faces = np.meshgrid(
            self.r_centers, self.z_faces, indexing='ij'
        )

    def _calculate_geometry_factors(self):
        """Расчёт геометрических факторов для осесимметричной геометрии."""

        # Объёмы контрольных объёмов
        # Для осесимметрии: V = π * (r_{i+1}^2 - r_i^2) * dz
        self.volumes = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_inner = self.r_faces[i]
            r_outer = self.r_faces[i + 1]
            # Объём кольца = π * (r_outer^2 - r_inner^2) * высота
            ring_area = np.pi * (r_outer ** 2 - r_inner ** 2)
            self.volumes[i, :] = ring_area * self.dz

        # Площади граней
        # Радиальные грани: A = 2*pi*r*dz
        self.areas_r = 2 * np.pi * self.R_r_faces * self.dz

        # Осевые грани: A = π * (r_outer^2 - r_inner^2) для каждого кольца
        self.areas_z = np.zeros((self.nr, self.nz + 1))
        for i in range(self.nr):
            r_inner = self.r_faces[i]
            r_outer = self.r_faces[i + 1]
            # Площадь кольца
            self.areas_z[i, :] = np.pi * (r_outer ** 2 - r_inner ** 2)

        # Расстояния между центрами ячеек
        self.dr_array = np.full((self.nr + 1, self.nz), self.dr)
        self.dz_array = np.full((self.nr, self.nz + 1), self.dz)

        # Специальная обработка для оси симметрии
        self.dr_array[0, :] = self.dr / 2  # Половинное расстояние на оси

    def get_cell_volumes(self) -> np.ndarray:
        """
        Получить объёмы контрольных объёмов.

        Возвращает
        ----------
        np.ndarray
            Массив объёмов размером (nr, nz)
        """
        return self.volumes.copy()

    def get_face_areas(self, direction: str) -> np.ndarray:
        """
        Получить площади граней.

        Параметры
        ---------
        direction : str
            'r' для радиальных граней, 'z' для осевых граней

        Возвращает
        ----------
        np.ndarray
            Массив площадей граней
        """
        if direction == 'r':
            return self.areas_r.copy()
        elif direction == 'z':
            return self.areas_z.copy()
        else:
            raise ValueError(f"Неверное направление: {direction}")

    def get_staggered_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Получить размеры массивов для staggered переменных.

        Возвращает
        ----------
        dict
            Словарь с размерами для каждого типа переменных:
            - 'cell': (nr, nz) - для скалярных величин в центрах
            - 'face_r': (nr+1, nz) - для vr на радиальных гранях
            - 'face_z': (nr, nz+1) - для vz на осевых гранях
        """
        return {
            'cell': (self.nr, self.nz),
            'face_r': (self.nr + 1, self.nz),
            'face_z': (self.nr, self.nz + 1)
        }

    def get_boundary_indices(self) -> Dict[str, Tuple[slice, slice]]:
        """
        Получить индексы для граничных условий.

        Возвращает
        ----------
        dict
            Словарь с индексами для каждой границы:
            - 'axis': ось симметрии (r=0)
            - 'wall': стенка (r=radius)
            - 'inlet': вход (z=0)
            - 'outlet': выход (z=height)
        """
        return {
            'axis': (0, slice(None)),           # r=0, все z
            'wall': (-1, slice(None)),          # r=radius, все z
            'inlet': (slice(None), 0),          # все r, z=0
            'outlet': (slice(None), -1)         # все r, z=height
        }

    def interpolate_to_faces(self, field: np.ndarray, direction: str) -> np.ndarray:
        """
        Интерполяция значений из центров ячеек на грани.

        Параметры
        ---------
        field : np.ndarray
            Поле в центрах ячеек размером (nr, nz)
        direction : str
            'r' для интерполяции на радиальные грани,
            'z' для интерполяции на осевые грани

        Возвращает
        ----------
        np.ndarray
            Интерполированные значения на гранях
        """
        if field.shape != (self.nr, self.nz):
            raise ValueError(f"Неверный размер поля: {field.shape}")

        if direction == 'r':
            # Интерполяция на радиальные грани
            face_values = np.zeros((self.nr + 1, self.nz))

            # Внутренние грани - среднее арифметическое
            face_values[1:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])

            # Граница на оси симметрии - экстраполяция
            face_values[0, :] = field[0, :]

            # Граница на стенке - экстраполяция
            face_values[-1, :] = field[-1, :]

            return face_values

        elif direction == 'z':
            # Интерполяция на осевые грани
            face_values = np.zeros((self.nr, self.nz + 1))

            # Внутренние грани - среднее арифметическое
            face_values[:, 1:-1] = 0.5 * (field[:, :-1] + field[:, 1:])

            # Нижняя граница
            face_values[:, 0] = field[:, 0]

            # Верхняя граница
            face_values[:, -1] = field[:, -1]

            return face_values

        else:
            raise ValueError(f"Неверное направление: {direction}")

    def gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчёт градиента скалярного поля.

        Параметры
        ---------
        field : np.ndarray
            Скалярное поле в центрах ячеек

        Возвращает
        ----------
        tuple
            (grad_r, grad_z) - компоненты градиента в центрах ячеек
        """
        if field.shape != (self.nr, self.nz):
            raise ValueError(f"Неверный размер поля: {field.shape}")

        grad_r = np.zeros_like(field)
        grad_z = np.zeros_like(field)

        # Градиент в радиальном направлении
        # Внутренние точки - центральные разности
        grad_r[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * self.dr)

        # Граница на оси - односторонняя разность
        grad_r[0, :] = (field[1, :] - field[0, :]) / self.dr

        # Граница на стенке - односторонняя разность
        grad_r[-1, :] = (field[-1, :] - field[-2, :]) / self.dr

        # Градиент в осевом направлении
        # Внутренние точки
        grad_z[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * self.dz)

        # Нижняя граница
        grad_z[:, 0] = (field[:, 1] - field[:, 0]) / self.dz

        # Верхняя граница
        grad_z[:, -1] = (field[:, -1] - field[:, -2]) / self.dz

        return grad_r, grad_z

    def divergence(self, vr: np.ndarray, vz: np.ndarray) -> np.ndarray:
        """
        Расчёт дивергенции векторного поля для осесимметричного случая.
        div(v) = 1/r * d(r*vr)/dr + dvz/dz

        Параметры
        ---------
        vr : np.ndarray
            Радиальная компонента скорости на радиальных гранях
        vz : np.ndarray
            Осевая компонента скорости на осевых гранях

        Возвращает
        ----------
        np.ndarray
            Дивергенция в центрах ячеек
        """
        if vr.shape != (self.nr + 1, self.nz):
            raise ValueError(f"Неверный размер vr: {vr.shape}")
        if vz.shape != (self.nr, self.nz + 1):
            raise ValueError(f"Неверный размер vz: {vz.shape}")

        div = np.zeros((self.nr, self.nz))

        # Радиальная часть: 1/r * d(r*vr)/dr
        for i in range(self.nr):
            for j in range(self.nz):
                r_center = self.r_centers[i]
                r_left = self.r_faces[i]
                r_right = self.r_faces[i + 1]

                flux_left = r_left * vr[i, j]
                flux_right = r_right * vr[i + 1, j]

                div[i, j] = (flux_right - flux_left) / (r_center * self.dr)

        # Осевая часть: dvz/dz
        div += (vz[:, 1:] - vz[:, :-1]) / self.dz

        return div

    def info(self) -> str:
        """Получить информацию о сетке."""
        info = [
            f"Осесимметричная сетка {self.nr}x{self.nz}",
            f"Радиус: {self.radius:.3f} м",
            f"Высота: {self.height:.3f} м",
            f"Шаг по r: {self.dr:.4f} м",
            f"Шаг по z: {self.dz:.4f} м",
            f"Всего ячеек: {self.nr * self.nz}",
            f"Минимальный объём: {self.volumes.min():.6e} м³",
            f"Максимальный объём: {self.volumes.max():.6e} м³"
        ]
        return "\n".join(info)

    def get_r_safe(self, min_r: float = 1e-10) -> np.ndarray:
        """
        Получить радиальные координаты с защитой от деления на ноль.

        Параметры
        ---------
        min_r : float
            Минимальное значение радиуса

        Возвращает
        ----------
        np.ndarray
            Безопасные радиальные координаты
        """
        return np.maximum(self.r_centers, min_r)