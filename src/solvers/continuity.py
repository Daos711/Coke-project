# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

__all__ = ["mass_flux_z_faces", "continuity_residual"]

def mass_flux_z_faces(
    rho: np.ndarray,
    gamma: np.ndarray,
    uz_face: np.ndarray,
    area_z: np.ndarray
) -> np.ndarray:
    """
    Массовый поток через z-гранях (на каждом столбце r):
      m_dot[j] = ∑_i ρ γ uz_face[i,j] * A_z[i,j]

    Parameters
    ----------
    rho : (nr, nz) плотность в центрах
    gamma : (nr, nz) жидкая пористость
    uz_face : (nr, nz+1) скорость на z-гранях (интринзик) [м/с]
    area_z : (nr, nz+1) площади z-граней [м²]

    Returns
    -------
    m_face : (nz+1,) массовый поток через каждую горизонтальную грань (сумма по r), кг/с
    """
    nr, nz = rho.shape
    m = np.zeros(nz + 1)
    # интерполяция ρ, γ на грани по z — среднее
    for j in range(nz + 1):
        jL = max(j - 1, 0)
        jR = min(j, nz - 1)
        rho_f = 0.5 * (rho[:, jL] + rho[:, jR])
        gam_f = 0.5 * (gamma[:, jL] + gamma[:, jR])
        m[j] = np.sum(rho_f * gam_f * uz_face[:, j] * area_z[:, j])
    return m


def continuity_residual(target_mdot: float, m_face: np.ndarray) -> Dict[str, float]:
    """Скалярные нормы нарушения непрерывности."""
    # расход должен быть одинаков на всех z-гранях и равен target_mdot
    err_abs = np.max(np.abs(m_face - target_mdot))
    err_rel = err_abs / (abs(target_mdot) + 1e-30)
    return {
        "max_abs": float(err_abs),
        "max_rel": float(err_rel)
    }
