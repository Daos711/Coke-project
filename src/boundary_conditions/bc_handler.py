"""Обработчик граничных условий для реактора замедленного коксования."""

import numpy as np
from enum import Enum
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass


class BCType(Enum):
    """Типы граничных условий."""
    DIRICHLET = "dirichlet"  # Фиксированное значение
    NEUMANN = "neumann"  # Фиксированный градиент
    ROBIN = "robin"  # Смешанное условие
    SYMMETRY = "symmetry"  # Ось симметрии
    INLET = "inlet"  # Вход
    OUTLET = "outlet"  # Выход
    WALL = "wall"  # Стенка


@dataclass
class BoundaryCondition:
    """Граничное условие."""
    bc_type: BCType
    value: Union[float, np.ndarray] = 0.0
    heat_transfer_coeff: Optional[float] = None  # Для Robin BC
    reference_value: Optional[float] = None  # Для Robin BC


class BoundaryConditionHandler:
    """Обработчик граничных условий для реактора."""

    def __init__(self, nr: int, nz: int):
        """
        Параметры
        ---------
        nr : int
            Количество ячеек по радиусу
        nz : int
            Количество ячеек по высоте
        """
        self.nr = nr
        self.nz = nz

        # Словари для хранения ГУ
        self.velocity_r_bc: Dict[str, BoundaryCondition] = {}
        self.velocity_z_bc: Dict[str, BoundaryCondition] = {}
        self.pressure_bc: Dict[str, BoundaryCondition] = {}
        self.temperature_bc: Dict[str, BoundaryCondition] = {}
        self.volume_fraction_bc: Dict[str, BoundaryCondition] = {}

        # Инициализация стандартных ГУ
        self._init_default_bc()

    def _init_default_bc(self):
        """Инициализация стандартных граничных условий."""

        # Скорость радиальная
        self.velocity_r_bc = {
            'axis': BoundaryCondition(BCType.SYMMETRY, 0.0),
            'wall': BoundaryCondition(BCType.DIRICHLET, 0.0),  # no-slip
            'inlet': BoundaryCondition(BCType.DIRICHLET, 0.0),
            'outlet': BoundaryCondition(BCType.NEUMANN, 0.0)
        }

        # Скорость осевая
        self.velocity_z_bc = {
            'axis': BoundaryCondition(BCType.NEUMANN, 0.0),
            'wall': BoundaryCondition(BCType.DIRICHLET, 0.0),  # no-slip
            'inlet': BoundaryCondition(BCType.INLET, 0.0),  # Будет задано
            'outlet': BoundaryCondition(BCType.NEUMANN, 0.0)
        }

        # Давление
        self.pressure_bc = {
            'axis': BoundaryCondition(BCType.NEUMANN, 0.0),
            'wall': BoundaryCondition(BCType.NEUMANN, 0.0),
            'inlet': BoundaryCondition(BCType.DIRICHLET, 101325),  # 1 атм
            'outlet': BoundaryCondition(BCType.NEUMANN, 0.0)
        }

        # Температура
        self.temperature_bc = {
            'axis': BoundaryCondition(BCType.NEUMANN, 0.0),
            'wall': BoundaryCondition(BCType.DIRICHLET, 783.15),  # 510°C
            'inlet': BoundaryCondition(BCType.DIRICHLET, 643.15),  # 370°C
            'outlet': BoundaryCondition(BCType.NEUMANN, 0.0)
        }

        # Объёмные доли
        self.volume_fraction_bc = {
            'axis': BoundaryCondition(BCType.NEUMANN, 0.0),
            'wall': BoundaryCondition(BCType.NEUMANN, 0.0),
            'inlet': BoundaryCondition(BCType.DIRICHLET, 1.0),  # 100% VR
            'outlet': BoundaryCondition(BCType.NEUMANN, 0.0)
        }

    def set_coking_mode(self, feed_velocity: float, wall_temperature: float):
        """
        Настройка режима коксования.

        Параметры
        ---------
        feed_velocity : float
            Скорость подачи VR (м/с)
        wall_temperature : float
            Температура стенки (К)
        """
        # Скорость на входе
        self.velocity_z_bc['inlet'].value = feed_velocity

        # Температура стенки
        self.temperature_bc['wall'].value = wall_temperature

    def set_cooling_mode(self, nitrogen_velocity: float, nitrogen_temperature: float):
        """
        Настройка режима охлаждения.

        Параметры
        ---------
        nitrogen_velocity : float
            Скорость азота (м/с)
        nitrogen_temperature : float
            Температура азота (К)
        """
        # Скорость азота
        self.velocity_z_bc['inlet'].value = nitrogen_velocity

        # Температура азота
        self.temperature_bc['inlet'].value = nitrogen_temperature

        # Стенка становится адиабатической
        self.temperature_bc['wall'] = BoundaryCondition(BCType.NEUMANN, 0.0)

        # На входе теперь азот (alpha_gas = 1)
        self.volume_fraction_bc['inlet'].value = 0.0  # 0% VR

    def apply_velocity_bc(self, vr: np.ndarray, vz: np.ndarray,
                          location: str, phase: str = 'liquid'):
        """
        Применение ГУ для скорости.

        Параметры
        ---------
        vr, vz : np.ndarray
            Поля скоростей (изменяются на месте)
        location : str
            Расположение границы ('axis', 'wall', 'inlet', 'outlet')
        phase : str
            Фаза ('liquid' или 'gas')
        """
        bc_r = self.velocity_r_bc[location]
        bc_z = self.velocity_z_bc[location]

        if location == 'axis':
            vr[0, :] = 0.0  # Всегда 0 на оси симметрии
            # vz - экстраполяция
            vz[0, :] = vz[1, :]

        elif location == 'wall':
            vr[-1, :] = bc_r.value
            vz[-1, :] = bc_z.value

        elif location == 'inlet':
            # Только для жидкой фазы на входе
            if phase == 'liquid':
                vz[:, 0] = bc_z.value
            else:
                vz[:, 0] = 0.0
            # vr - экстраполяция
            vr[:, 0] = vr[:, 1]

        elif location == 'outlet':
            # Нулевой градиент
            vr[:, -1] = vr[:, -2]
            vz[:, -1] = vz[:, -2]

    def apply_pressure_bc(self, p: np.ndarray, location: str):
        """Применение ГУ для давления."""
        bc = self.pressure_bc[location]

        if location == 'inlet' and bc.bc_type == BCType.DIRICHLET:
            p[:, 0] = bc.value
        elif location in ['axis', 'wall', 'outlet']:
            # Нулевой градиент
            if location == 'axis':
                p[0, :] = p[1, :]
            elif location == 'wall':
                p[-1, :] = p[-2, :]
            elif location == 'outlet':
                p[:, -1] = p[:, -2]

    def apply_temperature_bc(self, T: np.ndarray, location: str):
        """Применение ГУ для температуры."""
        bc = self.temperature_bc[location]

        if bc.bc_type == BCType.DIRICHLET:
            if location == 'wall':
                T[-1, :] = bc.value
            elif location == 'inlet':
                T[:, 0] = bc.value

        elif bc.bc_type == BCType.NEUMANN:
            if location == 'axis':
                T[0, :] = T[1, :]  # Симметрия
            elif location == 'wall':
                T[-1, :] = T[-2, :]  # Адиабата
            elif location == 'outlet':
                T[:, -1] = T[:, -2]

    def apply_volume_fraction_bc(self, alpha: np.ndarray, location: str, phase: str):
        """
        Применение ГУ для объёмной доли.

        Параметры
        ---------
        alpha : np.ndarray
            Объёмная доля (изменяется на месте)
        location : str
            Расположение границы
        phase : str
            Фаза ('liquid', 'gas', 'solid')
        """
        bc = self.volume_fraction_bc[location]

        if location == 'inlet' and phase == 'liquid':
            alpha[:, 0] = bc.value
        elif location == 'inlet' and phase == 'gas':
            alpha[:, 0] = 1.0 - bc.value
        elif location == 'inlet' and phase == 'solid':
            alpha[:, 0] = 0.0  # Кокс не подаётся
        else:
            # Нулевой градиент для остальных
            if location == 'axis':
                alpha[0, :] = alpha[1, :]
            elif location == 'wall':
                alpha[-1, :] = alpha[-2, :]
            elif location == 'outlet':
                alpha[:, -1] = alpha[:, -2]

    def get_inlet_mass_flux(self, density: float, area: float) -> float:
        """
        Расчёт массового расхода на входе.

        Параметры
        ---------
        density : float
            Плотность на входе (кг/м³)
        area : float
            Площадь входа (м²)

        Возвращает
        ----------
        float
            Массовый расход (кг/с)
        """
        vz_inlet = self.velocity_z_bc['inlet'].value
        return density * vz_inlet * area

    def set_mass_flow_inlet(self, m_dot: float, rho_in: float, inlet_area: float) -> float:
        """
        Задать вход по массовому расходу: v_in = m_dot / (rho_in * A_in).

        Параметры
        ---------
        m_dot : float
            Массовый расход (кг/с)
        rho_in : float
            Плотность на входе (кг/м³)
        inlet_area : float
            Площадь входа (м²)

        Возвращает
        ----------
        float
            Проставленная скорость (м/с)
        """
        eps = 1e-12
        denom = max(rho_in * inlet_area, eps)
        v_in = m_dot / denom
        self.velocity_z_bc['inlet'] = BoundaryCondition(BCType.INLET, v_in)
        return v_in