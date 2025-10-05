# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

__all__ = [
    "EnergySettings", "EnergySolver",
    "TwoTempSettings", "EnergyTwoTemperature",
]

# =============================================================================
# 1) ОДНОТЕМПЕРАТУРНЫЙ РЕШАТЕЛЬ (как был, но с опц. источником на объём)
# =============================================================================

@dataclass
class EnergySettings:
    dt: float = 0.05          # было 0.2 — меньше шаг = стабильнее
    max_iters: int = 5000
    min_iters: int = 40
    tol: float = 2e-4
    print_every: int = 100
    clip_min: float = 400.0
    clip_max: float = 2000.0


class EnergySolver:
    """
    Псевдонеявная итерация к стационару для энергии на r–z сетке.
    Невязка: L∞ от изменения поля между итерациями, отдельно для всего поля и
    для внутренней области (без Dirichlet-границ), чтобы не ловить «ложный ноль».

    Требуется объект fvm с методами:
      - diffusion_term(T, k, bc_type, bc_value) -> (nr,nz) [Вт/м^3]
      - convection_term(T, ur_face, uz_face) -> (nr,nz) [Вт/м^3]
    """
    def __init__(self, grid, fvm, fluid, settings: EnergySettings):
        self.grid = grid
        self.fvm = fvm
        self.fluid = fluid
        self.set = settings

    def step(
        self,
        T: np.ndarray,
        ur_face: np.ndarray,
        uz_face: np.ndarray,
        bc_type: Dict[str, str],
        bc_value: Dict[str, float],
        extra_source: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        # свойства
        rho = np.nan_to_num(self.fluid.density(T), nan=800.0, posinf=1e3, neginf=1.0)
        cp  = np.nan_to_num(self.fluid.heat_capacity(T), nan=2000.0, posinf=5e3, neginf=500.0)
        k   = np.nan_to_num(self.fluid.thermal_conductivity(T), nan=0.1, posinf=2.0, neginf=0.05)
        # члены уравнения
        diff = np.nan_to_num(self.fvm.diffusion_term(T, k, bc_type, bc_value), nan=0.0)
        conv = np.nan_to_num(self.fvm.convection_term(T, ur_face, uz_face), nan=0.0)
        rhs  = diff - conv
        if extra_source is not None:
            rhs = rhs + np.nan_to_num(extra_source, nan=0.0, posinf=0.0, neginf=0.0)

        den = rho * cp + 1e-30
        dT  = self.set.dt * rhs / den
        dT  = np.clip(dT, -50.0, 50.0)  # страховка от «скачков»
        T_new = T + dT

        # жёстко применяем ГУ
        if bc_type.get('inlet','') == 'dirichlet':
            T_new[:, 0] = bc_value.get('inlet', T_new[:, 0])
        if bc_type.get('outlet','') == 'neumann':
            T_new[:, -1] = T_new[:, -2]
        if bc_type.get('axis','') == 'neumann':
            T_new[0, :] = T_new[1, :]
        if bc_type.get('wall','') == 'dirichlet':
            T_new[-1, :] = bc_value.get('wall', T_new[-1, :])

        # физ. ограничения + NaN->числа
        T_new = np.nan_to_num(T_new, nan=self.set.clip_min)
        T_new = np.clip(T_new, self.set.clip_min, self.set.clip_max)

        dT_iter = np.abs(T_new - T)
        res_all = float(np.max(dT_iter))
        mask = np.ones_like(T, dtype=bool)
        if bc_type.get('inlet','') == 'dirichlet':
            mask[:, 0] = False
        if bc_type.get('wall','') == 'dirichlet':
            mask[-1, :] = False
        res_int = float(np.max(dT_iter[mask])) if np.any(mask) else res_all
        return T_new, res_all, res_int

    def solve_pseudo_transient(
        self,
        T0: np.ndarray,
        ur_face: np.ndarray,
        uz_face: np.ndarray,
        bc_type: Dict[str, str],
        bc_value: Dict[str, float],
        extra_source: Optional[np.ndarray] = None,  # опциональный источник на объём
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Гоняет step() до сходимости по внутренней невязке."""
        T = T0.copy()
        res_all_hist, res_int_hist = [], []
        for it in range(1, self.set.max_iters + 1):
            T_new, res_all, res_int = self.step(
                T, ur_face, uz_face, bc_type, bc_value, extra_source=extra_source
            )
            res_all_hist.append(res_all)
            res_int_hist.append(res_int)
            T = T_new
            if (it % self.set.print_every) == 0:
                print(f"[Energy] iter {it:5d} | max|ΔT|_all={res_all:.3e} | max|ΔT|_int={res_int:.3e}")
            if it >= self.set.min_iters and (res_int < self.set.tol):
                print(f"[Energy] CONVERGED (internal) at iter {it} with max|ΔT|_int={res_int:.3e}")
                break
        return T, np.asarray(res_all_hist, float), np.asarray(res_int_hist, float)

# =============================================================================
# 2) ДВУХТЕМПЕРАТУРНАЯ ОБЁРТКА (fluid ↔ coke через объёмный теплообмен)
# =============================================================================

@dataclass
class TwoTempSettings(EnergySettings):
    # параметры межфазного теплообмена и кокса
    dp_particle: float = 1.0e-3   # м, «частица кокса» для корреляций
    gamma_min: float = 1e-3
    k_solid: float = 1.5          # Вт/(м·К), теплопроводность кокса
    rho_solid: float = 1400.0     # кг/м^3
    cp_solid: float = 1000.0      # Дж/(кг·К)
    h_mult: float = 1.0           # множитель на U_v (тонкая подстройка)
    max_coupling_iters: int = 400
    coupling_tol: float = 1e-4    # критерий по max|ΔT| двух фаз


class EnergyTwoTemperature:
    """
    Связанные уравнения энергии для жидкости и кокса:
      fluid: ρf cpf ∂T_f/∂t = ∇·(k_f ∇T_f) - ∇·(ρf cpf v T_f) + U_v (T_c - T_f)
      coke : ρc cpc ∂T_c/∂t = ∇·(k_c ∇T_c) + U_v (T_f - T_c)
    где U_v — объёмный коэф. теплообмена (модель Wakao–Kaguei + уд.поверхность).
    """
    def __init__(self, grid, fvm, fluid, settings: TwoTempSettings):
        self.grid = grid
        self.fvm = fvm
        self.fluid = fluid
        self.set = settings

        # Переиспользуем базовый решатель для каждой «фазы»
        self.solv_f = EnergySolver(grid, fvm, fluid, settings)

        # «Солид» с константными свойствами
        class _Solid:
            def density(self, T): return np.full_like(T, settings.rho_solid, dtype=float)
            def heat_capacity(self, T): return np.full_like(T, settings.cp_solid, dtype=float)
            def thermal_conductivity(self, T): return np.full_like(T, settings.k_solid, dtype=float)
        self.solid = _Solid()
        self.solv_c = EnergySolver(grid, fvm, self.solid, settings)

    # ------------- служебные штуки -------------
    @staticmethod
    def _cell_speed_from_faces(uz_face: np.ndarray) -> np.ndarray:
        """uz_face: (nr, nz+1) → |u| в ячейках (nr, nz)."""
        return 0.5 * (uz_face[:, 1:] + uz_face[:, :-1])

    def _volumetric_htc(self, T_f, gamma, uz_face) -> np.ndarray:
        """Uv = a_s * h; защищённые Re/Pr и клипы, чтобы не «взрывалось»."""
        eps = np.clip(gamma, self.set.gamma_min, 0.999999)
        u = np.abs(self._cell_speed_from_faces(uz_face)) + 1e-30
        rho = np.nan_to_num(self.fluid.density(T_f), nan=800.0)
        mu  = np.nan_to_num(self.fluid.viscosity(T_f), nan=1e-3)
        cp  = np.nan_to_num(self.fluid.heat_capacity(T_f), nan=2000.0)
        kf  = np.nan_to_num(self.fluid.thermal_conductivity(T_f), nan=0.1)

        Re = np.clip(rho * u * self.set.dp_particle / np.maximum(mu, 1e-30), 0.0, 1e6)
        Pr = np.clip(mu * cp / np.maximum(kf, 1e-30), 1e-3, 1e3)
        Nu = 2.0 + 1.1 * np.power(Re, 0.6) * np.power(Pr, 1.0/3.0)
        Nu = np.clip(Nu, 2.0, 5e2)

        h  = Nu * kf / (self.set.dp_particle + 1e-30)           # Вт/(м²·К)
        a_s = 6.0 * (1.0 - eps) / (self.set.dp_particle + 1e-30)  # м²/м³
        Uv = self.set.h_mult * a_s * h                           # Вт/(м³·К)
        return np.clip(Uv, 0.0, 5e5)  # верхняя отсечка

    # ------------- основной связанный цикл -------------
    def solve_coupled(
        self,
        T_f0: np.ndarray, T_c0: np.ndarray,
        ur_face: np.ndarray, uz_face: np.ndarray,
        gamma: np.ndarray,
        bc_f_type: Dict[str, str], bc_f_value: Dict[str, float],
        bc_c_type: Dict[str, str], bc_c_value: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает T_f, T_c, history(max|ΔT|) за итерации связи."""
        T_f = T_f0.copy();
        T_c = T_c0.copy()
        hist = []
        zeros_r = np.zeros_like(ur_face);
        zeros_z = np.zeros_like(uz_face)

        dt_local = float(self.set.dt)
        for it in range(1, self.set.max_coupling_iters + 1):
            Uv = self._volumetric_htc(T_f, gamma, uz_face)
            S_f = Uv * (T_c - T_f)
            S_c = Uv * (T_f - T_c)

            # подстраиваем шаг если вдруг «поплыли» температуры
            self.solv_f.set.dt = dt_local
            self.solv_c.set.dt = dt_local

            T_f_new, _, _ = self.solv_f.step(T_f, ur_face, uz_face, bc_f_type, bc_f_value, extra_source=S_f)
            T_c_new, _, _ = self.solv_c.step(T_c, zeros_r, zeros_z, bc_c_type, bc_c_value, extra_source=S_c)

            if not (np.all(np.isfinite(T_f_new)) and np.all(np.isfinite(T_c_new))):
                # уменьшаем шаг и повторяем итерацию
                dt_local *= 0.5
                if dt_local < 1e-4:
                    raise FloatingPointError("Energy 2T diverged: dt too small after halving.")
                continue

            dTf = float(np.max(np.abs(T_f_new - T_f)))
            dTc = float(np.max(np.abs(T_c_new - T_c)))
            res = max(dTf, dTc);
            hist.append(res)

            T_f, T_c = T_f_new, T_c_new

            if (it % self.set.print_every) == 0:
                print(f"[2T] iter {it:4d} | max|ΔT_f|={dTf:.3e} | max|ΔT_c|={dTc:.3e} | dt={dt_local:.3e}")
            if it >= self.set.min_iters and (res < self.set.coupling_tol):
                print(f"[2T] CONVERGED at iter {it} with max|ΔT|={res:.3e}")
                break

        return T_f, T_c, np.asarray(hist, float)
