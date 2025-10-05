"""Вспомогательные функции для CFD-симуляции замедленного коксования."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import yaml


# ------------------------- Работа с конфигами/файлами ------------------------- #
def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Загрузить YAML-конфиг.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Глубокое объединение нескольких словарей-конфигов.
    Поздние значения перекрывают ранние, вложенные dict'ы мёржатся.
    """
    def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_update(out[k], v)
            else:
                out[k] = v
        return out

    merged: Dict[str, Any] = {}
    for cfg in configs:
        merged = _deep_update(merged, cfg)
    return merged


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Убедиться, что директория существует; создать при необходимости.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------- Объёмные доли и клиппинг ------------------------ #
def clip01(a: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Обрезка значения/массива в диапазон [eps, 1-eps].
    """
    if np.isscalar(a):
        return min(1.0 - eps, max(eps, float(a)))
    return np.minimum(1.0 - eps, np.maximum(eps, a))


def enforce_volume_fractions(
    alpha_R: Union[float, np.ndarray],
    alpha_D: Union[float, np.ndarray],
    alpha_C: Union[float, np.ndarray],
    eps: float = 1e-12,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Нормировка объёмных долей для трёхфазной системы.

    Гарантирует:
      - gamma = 1 - alpha_C (пористость),
      - alpha_R + alpha_D = gamma (флюидные фазы заполняют поры),
      - все доли в [eps, 1-eps].

    Возвращает (alpha_R, alpha_D, alpha_C, gamma).
    """
    # сначала ограничим кокс
    alpha_C = clip01(alpha_C, eps)
    gamma = 1.0 - alpha_C

    # нормируем флюидные доли на гамму
    s = np.maximum(eps, alpha_R + alpha_D)
    alpha_R = gamma * alpha_R / s
    alpha_D = gamma * alpha_D / s

    alpha_R = clip01(alpha_R, eps)
    alpha_D = clip01(alpha_D, eps)
    return alpha_R, alpha_D, alpha_C, gamma


# ----------------------------- Число Куранта --------------------------------- #
def calculate_courant_number(
    velocity: np.ndarray, dt: float, dr: float, dz: float
) -> float:
    """
    Расчёт максимального числа Куранта для поля скоростей.

    velocity: ndarray со срезами [0] = v_r, [1] = v_z
    dt: шаг по времени
    dr, dz: шаги сетки по r и z (не нулевые)
    """
    vr_max = float(np.max(np.abs(velocity[0])))
    vz_max = float(np.max(np.abs(velocity[1])))
    dr = max(dr, 1e-12)
    dz = max(dz, 1e-12)
    return vr_max * dt / dr + vz_max * dt / dz


# ============================================================================
# LEGACY-функции для ДЕКАРТОВОЙ сетки — в осесимметрии не использовать.
# Оставлены для совместимости со старыми черновиками.
# ============================================================================
def interpolate_to_faces(field: np.ndarray, axis: int) -> np.ndarray:
    """
    [LEGACY, только для декартовой сетки]
    Интерполяция значений из центров ячеек на грани.
    Для осесимметрии используйте методы AxiSymmetricGrid.
    """
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    if axis == 1:
        return 0.5 * (field[:, :-1] + field[:, 1:])
    raise ValueError(f"Неверная ось: {axis}")


def apply_boundary_conditions(
    field: np.ndarray, bc_type: str, bc_value: float, location: str
) -> None:
    """
    [LEGACY, только для декартовой сетки]
    Применить ГУ к полю (in-place).
    Для осесимметрии используйте BoundaryConditionHandler.
    """
    if bc_type == "dirichlet":
        if location == "bottom":
            field[:, 0] = bc_value
        elif location == "top":
            field[:, -1] = bc_value
        elif location == "left":
            field[0, :] = bc_value
        elif location == "right":
            field[-1, :] = bc_value
        else:
            raise ValueError(f"Неизвестное расположение: {location}")

    elif bc_type == "neumann":
        if location == "bottom":
            field[:, 0] = field[:, 1] + bc_value
        elif location == "top":
            field[:, -1] = field[:, -2] + bc_value
        elif location == "left":
            field[0, :] = field[1, :] + bc_value
        elif location == "right":
            field[-1, :] = field[-2, :] + bc_value
        else:
            raise ValueError(f"Неизвестное расположение: {location}")

    elif bc_type == "symmetry":
        if location == "left":
            field[0, :] = field[1, :]
        elif location == "right":
            field[-1, :] = field[-2, :]
        elif location == "bottom":
            field[:, 0] = field[:, 1]
        elif location == "top":
            field[:, -1] = field[:, -2]
        else:
            raise ValueError(f"Неизвестное расположение: {location}")

    else:
        raise ValueError(f"Неизвестный тип ГУ: {bc_type}")
