import numpy as np
from typing import Tuple

def enforce_alpha_constraints(alpha_R: np.ndarray,
                              alpha_D: np.ndarray,
                              alpha_C: np.ndarray,
                              eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Приводит объёмные доли к физически допустимым:
    - 0 <= alpha_k <= 1
    - alpha_R + alpha_D + alpha_C <= 1
    - gamma = 1 - alpha_C >= 0
    Если сумма R+D+C > 1, R и D пропорционально уменьшаются (кокс не трогаем).
    Возвращает (alpha_R, alpha_D, alpha_C, gamma).
    """
    aR = np.clip(alpha_R, 0.0, 1.0)
    aD = np.clip(alpha_D, 0.0, 1.0)
    aC = np.clip(alpha_C, 0.0, 1.0)

    aC = np.minimum(aC, 1.0 - eps)

    s = aR + aD + aC
    mask = s > 1.0
    excess = np.maximum(s - 1.0, 0.0)

    denom = np.maximum(aR + aD, eps)
    shareR = np.where(mask, aR / denom, 0.0)
    shareD = np.where(mask, aD / denom, 0.0)

    aR = np.where(mask, np.maximum(aR - shareR * excess, 0.0), aR)
    aD = np.where(mask, np.maximum(aD - shareD * excess, 0.0), aD)

    gamma = 1.0 - aC
    gamma = np.clip(gamma, 0.0, 1.0)
    return aR, aD, aC, gamma
