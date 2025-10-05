"""Физические свойства материалов для симуляции замедленного коксования."""

import numpy as np
from dataclasses import dataclass
from typing import Union
from pathlib import Path
import sys

# Добавляем путь к корню проекта для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config


def _to_array_and_flag(x):
    """Возвращает (np.ndarray, is_scalar)."""
    arr = np.asarray(x, dtype=float)
    return arr, (arr.ndim == 0)


def _ret(x, is_scalar: bool):
    """Возвращает скаляр, если вход был скаляром, иначе массив."""
    return float(np.asarray(x)) if is_scalar else np.asarray(x)


@dataclass
class MaterialProperties:
    """Базовый класс для свойств материалов."""

    density_ref: float      # Опорная плотность (кг/м³)
    cp_ref: float           # Опорная теплоёмкость (Дж/кг·К)
    viscosity_ref: float    # Опорная вязкость (Па·с)
    k_thermal_ref: float    # Опорная теплопроводность (Вт/м·К)
    temp_ref: float = 288.15  # Опорная температура (К)


class VacuumResidue:
    """Свойства вакуумного остатка (жидкая фаза)."""

    def __init__(self, vr_type: int, config_path: Path = None):
        if vr_type not in [1, 2, 3]:
            raise ValueError(f"Неверный тип VR: {vr_type}")

        self.vr_type = vr_type

        # Загрузка свойств из конфигурации
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config'

        props = load_config(config_path / 'physical_properties.yaml')
        vr_data = props[f'vacuum_residue_{vr_type}']

        # Основные свойства при 15°C
        self.density_15C = vr_data['density_15C']
        self.API = vr_data['API']
        self.CCR = vr_data['CCR']

        # SARA фракции
        self.saturates = vr_data['saturates']
        self.aromatics = vr_data['aromatics']
        self.resins = vr_data['resins']
        self.asphaltenes = vr_data['asphaltenes']

        # Опорные значения (примерные)
        self.props = MaterialProperties(
            density_ref=self.density_15C,
            cp_ref=2000.0,     # Дж/кг·К
            viscosity_ref=0.1, # Па·с при 370°C (референс)
            k_thermal_ref=0.15 # Вт/м·К
        )

    def density(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Плотность как функция температуры (кг/м³)."""
        alpha = 0.0007  # 1/К
        Tarr, was_scalar = _to_array_and_flag(T)
        rho = self.props.density_ref / (1.0 + alpha * (Tarr - 288.15))
        return _ret(rho, was_scalar)

    def viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Вязкость как функция температуры (Па·с), безопасная форма Andrade: ln μ = A + B/T."""
        A, B = -15.0, 5000.0  # подгоночные параметры
        Tarr, was_scalar = _to_array_and_flag(T)
        Tclip = np.maximum(Tarr, 200.0)  # защита от T→0
        mu = np.exp(A + B / Tclip)
        # ограничим разумно и устраним не-числа без in-place
        mu = np.clip(mu, 1e-5, 10.0)
        mu = np.where(np.isfinite(mu), mu, self.props.viscosity_ref)
        return _ret(mu, was_scalar)

    def heat_capacity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Теплоёмкость (Дж/кг·К)."""
        Tarr, was_scalar = _to_array_and_flag(T)
        cp = self.props.cp_ref + 2.5 * (Tarr - self.props.temp_ref)
        return _ret(cp, was_scalar)

    def thermal_conductivity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Теплопроводность (Вт/м·К) с защитой от отрицательных значений."""
        Tarr, was_scalar = _to_array_and_flag(T)
        Tclip = np.maximum(Tarr, 200.0)
        k = self.props.k_thermal_ref * (1.0 - 0.0002 * (Tclip - self.props.temp_ref))
        k = np.maximum(k, 0.02)
        k = np.where(np.isfinite(k), k, self.props.k_thermal_ref)
        return _ret(k, was_scalar)


class Distillables:
    """Свойства дистиллятов (газовая фаза)."""

    def __init__(self):
        # Молярная масса типичных углеводородных паров
        self.molecular_weight = 0.035  # кг/моль (35 г/моль)
        self.R_universal = 8.314462618  # Дж/(моль·К) - универсальная газовая постоянная

        # Базовые свойства
        self.cp_base = 2000.0  # Дж/(кг·К) при 300 K
        self.cp_coeff = 2.5    # Температурный коэффициент

        # Параметры для μ(T) и k(T)
        self.T0_mu = 273.15
        self.mu0   = 8e-6      # Па·с при T0
        self.S_mu  = 200.0     # K
        self.k0    = 0.03      # Вт/м·К при 300 K
        self.k_exp = 0.8
        self.T0_k  = 300.0

    def density(self, T: Union[float, np.ndarray], P: float = 101325) -> Union[float, np.ndarray]:
        """Плотность по уравнению идеального газа (кг/м³).
        ρ = P·M/(R·T) где P в Па, M в кг/моль, R=8.314 Дж/(моль·К), T в К

        Проверка: При T=643K, P=101325 Па, M=0.035 кг/моль:
        ρ = (101325 * 0.035) / (8.314 * 643) = 0.663 кг/м³
        """
        Tarr, was_scalar = _to_array_and_flag(T)
        Tclip = np.maximum(Tarr, 1.0)  # защита от деления на 0

        # Правильная формула идеального газа
        rho = (P * self.molecular_weight) / (self.R_universal * Tclip)

        return _ret(rho, was_scalar)

    def viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Вязкость по Сазерленду (Па·с) с защитой от T→0."""
        Tarr, was_scalar = _to_array_and_flag(T)
        Tclip = np.maximum(Tarr, 50.0)
        mu = self.mu0 * (Tclip / self.T0_mu) ** 1.5 * (self.T0_mu + self.S_mu) / (Tclip + self.S_mu)
        mu = np.maximum(mu, 1e-6)
        mu = np.where(np.isfinite(mu), mu, self.mu0)
        return _ret(mu, was_scalar)

    def heat_capacity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Теплоёмкость (Дж/кг·К)."""
        Tarr, was_scalar = _to_array_and_flag(T)
        cp = self.cp_base + self.cp_coeff * (Tarr - 300.0)
        return _ret(cp, was_scalar)

    def thermal_conductivity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Теплопроводность (Вт/м·К) с защитой от T→0 и нулей."""
        Tarr, was_scalar = _to_array_and_flag(T)
        Tclip = np.maximum(Tarr, 50.0)
        k = self.k0 * (Tclip / self.T0_k) ** self.k_exp
        k = np.maximum(k, 1e-4)
        k = np.where(np.isfinite(k), k, self.k0)
        return _ret(k, was_scalar)


class Coke:
    """Свойства кокса (твёрдая фаза)."""

    def __init__(self):
        self.props = MaterialProperties(
            density_ref=1500.0,   # кг/м³ (истинная плотность)
            cp_ref=1000.0,        # Дж/кг·К
            viscosity_ref=np.inf, # Твёрдое тело
            k_thermal_ref=0.5     # Вт/м·К
        )
        self.particle_diameter = 0.001  # м (1 мм)
        self.bulk_density = 800.0       # кг/м³ (насыпная плотность)

    def density(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        Tarr, was_scalar = _to_array_and_flag(T)
        rho = np.full_like(Tarr, self.props.density_ref, dtype=float)
        return _ret(rho, was_scalar)

    def heat_capacity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        Tarr, was_scalar = _to_array_and_flag(T)
        cp = self.props.cp_ref + 0.5 * (Tarr - self.props.temp_ref)
        return _ret(cp, was_scalar)

    def thermal_conductivity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        Tarr, was_scalar = _to_array_and_flag(T)
        k = np.full_like(Tarr, self.props.k_thermal_ref, dtype=float)
        return _ret(k, was_scalar)