"""Конечно-объёмная дискретизация для реактора."""

import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mesh.grid_2d import AxiSymmetricGrid


class FiniteVolumeDiscretization:
    """Дискретизация уравнений методом конечных объёмов."""

    def __init__(self, grid: AxiSymmetricGrid):
        """
        Параметры
        ---------
        grid : AxiSymmetricGrid
            Сетка
        """
        self.grid = grid
        self.nr = grid.nr
        self.nz = grid.nz

        # Предвычисленные коэффициенты
        self._setup_coefficients()

    def _setup_coefficients(self):
        """Предвычисление геометрических коэффициентов."""
        # Безопасные радиусы
        self.r_safe = self.grid.get_r_safe()

        # Площади граней
        self.area_r = self.grid.get_face_areas('r')
        self.area_z = self.grid.get_face_areas('z')

        # Объёмы ячеек
        self.volumes = self.grid.get_cell_volumes()

        # Расстояния между центрами
        self.dr_c = np.zeros(self.nr + 1)  # Расстояния по r
        self.dr_c[1:-1] = self.grid.dr
        self.dr_c[0] = self.grid.dr / 2
        self.dr_c[-1] = self.grid.dr / 2

        self.dz_c = self.grid.dz * np.ones(self.nz + 1)

    def diffusion_term(self, phi: np.ndarray, gamma: np.ndarray,
                       bc_type: Dict[str, str], bc_value: Dict[str, float]) -> np.ndarray:
        """
        Дискретизация диффузионного члена: ∇·(Γ∇φ) для осесимметрии.
        Возвращает ∇·(Γ∇φ).
        Договорённость: для Neumann bc_value[...] — это декартовый градиент (∂φ/∂r, ∂φ/∂z).
        """
        flux = np.zeros_like(phi)

        # Γ на гранях
        gamma_r = self.grid.interpolate_to_faces(gamma, 'r')  # shape (nr+1, nz)
        gamma_z = self.grid.interpolate_to_faces(gamma, 'z')  # shape (nr, nz+1)

        # Фантомные ячейки
        phi_ext = np.zeros((self.nr + 2, self.nz + 2))
        phi_ext[1:-1, 1:-1] = phi

        # Ось r=0 (нулевой градиент, зеркало)
        if self.nr > 1:
            phi_ext[0, 1:-1] = phi[1, :]
        else:
            phi_ext[0, 1:-1] = phi[0, :]

        # Стенка r=R
        if bc_type['wall'] == 'dirichlet':
            phi_ext[-1, 1:-1] = 2.0 * bc_value['wall'] - phi[-1, :]
        else:  # neumann -> фантом не используем для потока, берём текущее
            phi_ext[-1, 1:-1] = phi[-1, :]

        # Вход z=0
        if bc_type['inlet'] == 'dirichlet':
            phi_ext[1:-1, 0] = 2.0 * bc_value['inlet'] - phi[:, 0]
        else:  # neumann
            phi_ext[1:-1, 0] = phi[:, 0]

        # Выход z=H
        if bc_type['outlet'] == 'dirichlet':
            phi_ext[1:-1, -1] = 2.0 * bc_value['outlet'] - phi[:, -1]
        else:  # neumann
            phi_ext[1:-1, -1] = phi[:, -1]

        # Углы
        phi_ext[0, 0] = phi_ext[1, 0]
        phi_ext[0, -1] = phi_ext[1, -1]
        phi_ext[-1, 0] = phi_ext[-2, 0]
        phi_ext[-1, -1] = phi_ext[-2, -1]

        # Потоки через грани (используем предвычисленные площади)
        for i in range(self.nr):
            for j in range(self.nz):
                ie = i + 1
                je = j + 1

                # --- Радиальные ---
                # Восточная грань (к стенке)
                area_e = self.area_r[i + 1, j]
                if i < self.nr - 1 or bc_type['wall'] == 'dirichlet':
                    dphi_dr_e = (phi_ext[ie + 1, je] - phi_ext[ie, je]) / self.grid.dr
                    flux_e = gamma_r[i + 1, j] * area_e * dphi_dr_e
                else:
                    # Neumann на стенке: поток = Γ * A * (∂φ/∂r)
                    flux_e = gamma_r[i + 1, j] * area_e * bc_value['wall']

                # Западная грань (к оси)
                if i > 0:
                    area_w = self.area_r[i, j]
                    dphi_dr_w = (phi_ext[ie, je] - phi_ext[ie - 1, je]) / self.grid.dr
                    flux_w = gamma_r[i, j] * area_w * dphi_dr_w
                else:
                    # Ось: площадь = 0 ⇒ поток 0
                    flux_w = 0.0

                # --- Осевые ---
                # Северная (вверх, к выходу)
                area_n = self.area_z[i, j + 1]
                if j < self.nz - 1 or bc_type['outlet'] == 'dirichlet':
                    dphi_dz_n = (phi_ext[ie, je + 1] - phi_ext[ie, je]) / self.grid.dz
                    flux_n = gamma_z[i, j + 1] * area_n * dphi_dz_n
                else:
                    # Neumann на выходе
                    flux_n = gamma_z[i, j + 1] * area_n * bc_value['outlet']

                # Южная (вниз, к входу)
                area_s = self.area_z[i, j]
                if j > 0 or bc_type['inlet'] == 'dirichlet':
                    dphi_dz_s = (phi_ext[ie, je] - phi_ext[ie, je - 1]) / self.grid.dz
                    flux_s = gamma_z[i, j] * area_s * dphi_dz_s
                else:
                    # Neumann на входе
                    flux_s = gamma_z[i, j] * area_s * bc_value['inlet']

                # Суммарно / объём: ∇·(Γ∇φ)
                flux[i, j] = (flux_e - flux_w + flux_n - flux_s) / self.volumes[i, j]

        return flux

    def convection_term(self, phi: np.ndarray, vr: np.ndarray, vz: np.ndarray,
                        scheme: str = 'upwind', phi_bc: Dict[str, float] = None) -> np.ndarray:
        """
        Дискретизация конвективного члена.
        Возвращает значение с минусом:  -∇·(v φ)

        Параметры
        ---------
        phi : np.ndarray
            Переносимая величина (nr, nz)
        vr, vz : np.ndarray
            Скорости на гранях
        scheme : str
            Схема ('upwind', 'central')
        phi_bc : dict
            Граничные значения phi (опционально)

        Возвращает
        ----------
        np.ndarray
            Конвективный поток (nr, nz)
        """

        flux = np.zeros_like(phi)

        # Граничные значения по умолчанию
        if phi_bc is None:
            phi_bc = {'inlet': 0, 'outlet': 0, 'wall': 0, 'axis': 0}

        for i in range(self.nr):
            for j in range(self.nz):
                # Восточная грань
                if i < self.nr - 1:
                    if scheme == 'upwind':
                        phi_e = phi[i, j] if vr[i + 1, j] >= 0 else phi[i + 1, j]
                    else:
                        phi_e = 0.5 * (phi[i, j] + phi[i + 1, j])
                    flux_e = vr[i + 1, j] * self.area_r[i + 1, j] * phi_e
                else:
                    flux_e = 0  # На стенке vr = 0

                # Западная грань
                if i > 0:
                    if scheme == 'upwind':
                        phi_w = phi[i - 1, j] if vr[i, j] >= 0 else phi[i, j]
                    else:
                        phi_w = 0.5 * (phi[i - 1, j] + phi[i, j])
                    flux_w = vr[i, j] * self.area_r[i, j] * phi_w
                else:
                    flux_w = 0  # На оси vr = 0

                # Северная грань
                if j < self.nz - 1:
                    if scheme == 'upwind':
                        phi_n = phi[i, j] if vz[i, j + 1] >= 0 else phi[i, j + 1]
                    else:
                        phi_n = 0.5 * (phi[i, j] + phi[i, j + 1])
                    flux_n = vz[i, j + 1] * self.area_z[i, j + 1] * phi_n
                else:
                    # Выход - экстраполяция
                    phi_n = phi[i, j]
                    flux_n = vz[i, j + 1] * self.area_z[i, j + 1] * phi_n

                # Южная грань
                if j > 0:
                    if scheme == 'upwind':
                        phi_s = phi[i, j - 1] if vz[i, j] >= 0 else phi[i, j]
                    else:
                        phi_s = 0.5 * (phi[i, j - 1] + phi[i, j])
                    flux_s = vz[i, j] * self.area_z[i, j] * phi_s
                else:
                    # Вход - используем граничное значение
                    if vz[i, j] >= 0:
                        phi_s = phi_bc.get('inlet', phi[i, j])
                    else:
                        phi_s = phi[i, j]
                    flux_s = vz[i, j] * self.area_z[i, j] * phi_s

                flux[i, j] = -(flux_e - flux_w + flux_n - flux_s) / self.volumes[i, j]

        return flux

    def source_term(self, source: np.ndarray) -> np.ndarray:
        """
        Дискретизация источникового члена.

        Параметры
        ---------
        source : np.ndarray
            Объёмный источник (nr, nz)

        Возвращает
        ----------
        np.ndarray
            Источник (без изменений для FVM)
        """
        return source

    def time_derivative(self, phi_new: np.ndarray, phi_old: np.ndarray,
                        dt: float) -> np.ndarray:
        """
        Дискретизация производной по времени.

        Параметры
        ---------
        phi_new, phi_old : np.ndarray
            Новое и старое значения
        dt : float
            Шаг по времени

        Возвращает
        ----------
        np.ndarray
            ∂φ/∂t
        """
        return (phi_new - phi_old) / dt