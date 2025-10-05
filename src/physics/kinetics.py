"""Кинетика реакций замедленного коксования."""

import numpy as np
from typing import Union, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import load_config


class ReactionKinetics:
    """Кинетика реакции VR → Coke + Distillables."""

    def __init__(self, vr_type: int, config_path: Path = None):
        """
        Параметры
        ---------
        vr_type : int
            Тип вакуумного остатка (1, 2 или 3)
        """
        if vr_type not in [1, 2, 3]:
            raise ValueError(f"Неверный тип VR: {vr_type}")

        self.vr_type = vr_type

        # Загрузка параметров
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config'

        kinetics = load_config(config_path / 'kinetic_params.yaml')
        self.params = kinetics[f'vacuum_residue_{vr_type}']

        # Извлекаем параметры для каждого порядка (преобразуем в float)
        self.k0_1 = float(self.params['first_order']['k0'])
        self.Ea_1 = float(self.params['first_order']['Ea'])
        self.T_trans_1 = float(self.params['first_order']['transition_T'])

        self.k0_15 = float(self.params['order_1_5']['k0'])
        self.Ea_15 = float(self.params['order_1_5']['Ea'])
        self.T_trans_15 = float(self.params['order_1_5']['transition_T'])

        self.k0_2 = float(self.params['second_order']['k0'])
        self.Ea_2 = float(self.params['second_order']['Ea'])

        self.R_gas = 8.314  # Дж/(моль·К)

    def rate_constant(self, T: Union[float, np.ndarray], order: float) -> Union[float, np.ndarray]:
        """Константа скорости по Аррениусу."""
        if order == 1.0:
            return self.k0_1 * np.exp(-self.Ea_1 / (self.R_gas * T))
        elif order == 1.5:
            return self.k0_15 * np.exp(-self.Ea_15 / (self.R_gas * T))
        elif order == 2.0:
            return self.k0_2 * np.exp(-self.Ea_2 / (self.R_gas * T))

    def reaction_order(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Определение порядка реакции по температуре."""
        if np.isscalar(T):
            if T < self.T_trans_1:
                return 1.0
            elif T < self.T_trans_15:
                return 1.5
            else:
                return 2.0
        else:
            order = np.ones_like(T)
            order[T >= self.T_trans_1] = 1.5
            order[T >= self.T_trans_15] = 2.0
            return order

    def reaction_rate(self, T: Union[float, np.ndarray], C_vr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Скорость реакции.

        Параметры
        ---------
        T : float или array
            Температура (К)
        C_vr : float или array
            Концентрация VR (кг/м³)

        Возвращает
        ----------
        rate : float или array
            Скорость реакции (кг/(м³·с))
        """
        order = self.reaction_order(T)

        if np.isscalar(T):
            k = self.rate_constant(T, order)
            return k * C_vr ** order
        else:
            rate = np.zeros_like(T, dtype=float)

            # Первый порядок
            mask1 = order == 1.0
            if np.any(mask1):
                T_mask = T[mask1] if hasattr(T, '__getitem__') else T
                k1 = self.rate_constant(T_mask, 1.0)
                if np.isscalar(C_vr):
                    rate[mask1] = k1 * C_vr ** 1.0
                else:
                    rate[mask1] = k1 * C_vr[mask1] ** 1.0

            # Порядок 1.5
            mask15 = order == 1.5
            if np.any(mask15):
                T_mask = T[mask15] if hasattr(T, '__getitem__') else T
                k15 = self.rate_constant(T_mask, 1.5)
                if np.isscalar(C_vr):
                    rate[mask15] = k15 * C_vr ** 1.5
                else:
                    rate[mask15] = k15 * C_vr[mask15] ** 1.5

            # Второй порядок
            mask2 = order == 2.0
            if np.any(mask2):
                T_mask = T[mask2] if hasattr(T, '__getitem__') else T
                k2 = self.rate_constant(T_mask, 2.0)
                if np.isscalar(C_vr):
                    rate[mask2] = k2 * C_vr ** 2.0
                else:
                    rate[mask2] = k2 * C_vr[mask2] ** 2.0

            return rate

    def source_terms(self, T: Union[float, np.ndarray], C_vr: Union[float, np.ndarray]) -> Tuple:
        """
        Источниковые члены для всех фаз.

        Возвращает
        ----------
        tuple : (Gamma_R, Gamma_C, Gamma_D)
            Скорости для VR, Coke, Distillables (кг/(м³·с))
        """
        rate = self.reaction_rate(T, C_vr)

        # Стехиометрия: VR → Coke + Distillables
        # Для простоты считаем равные массовые доли
        f_coke = 0.3  # Доля кокса (можно уточнить)
        f_dist = 0.7  # Доля дистиллятов

        Gamma_R = -rate  # VR расходуется
        Gamma_C = f_coke * rate  # Кокс образуется
        Gamma_D = f_dist * rate  # Дистилляты образуются

        return Gamma_R, Gamma_C, Gamma_D