# -*- coding: utf-8 -*-
"""
SIMPLEC в пористой среде (осесимметрия r–z, Дарси–Форшхаймер)
----------------------------------------------------------------
Drop-in модуль для проекта delayed_coking_cfd.

Основная идея (Патанкар–Сполдинг + SIMPLEC):
  S_P,f = (γ_f^2 μ_f / K_f) + 0.5 γ_f^3 C2_f ρ_f |u_old,f|
  u*_f  = - (1/S_P,f) * ∂p/∂n                      — прогноз на гранях
  d_f   =  A_f / S_P,f                             — коэффициенты коррекции
  div[ (ρ γ)_f d_f grad(p') ] = m_res / ΔV         — уравнение p′
  u_f   = u*_f - (1/S_P,f) * ∂p'/∂n                — коррекция скоростей
  p     = p + α_p p'                               — коррекция давления

Границы:
  - Вход (нижняя осевая грань): фиксированный массовый расход →
    в уравнении p′ «замораживаем» западную грань (нет коррекции),
    после коррекции скоростей жёстко перенормируем ṁ.
  - Выход (верх): p′=0 (референс по давлению на выходе).
  - Ось и стенка: Neumann для p′ (нулевой поток коррекции).

Сетка — «staggered»: давление в центрах; vr на r-гранях (nr+1,nz),
 vz на z-гранях (nr,nz+1). Площади граней и объёмы берутся из сетки.

Совместим с:
  - mesh.grid_2d.AxiSymmetricGrid
  - physics.properties.VacuumResidue (density(T), viscosity(T))
  - physics.correlations.PorousDrag (ergun_permeability, ergun_inertial)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from mesh.grid_2d import AxiSymmetricGrid
from physics.properties import VacuumResidue
from physics.correlations import PorousDrag


# =========================
# Настройки SIMPLEC
# =========================
@dataclass
class SimpleCPorousSettings:
    dp_particle: float = 1.0e-3      # экв. диаметр «частиц кокса» (Эргун)
    gamma_min: float  = 1.0e-3       # минимум пористости на численной стороне
    max_outer: int    = 200          # максимум внешних итераций SIMPLEC
    tol_m: float      = 1e-8         # относительный критерий по массе
    relax_p: float    = 0.7          # релаксация давления
    relax_u: float    = 1.0          # релаксация скоростей (в Дарси-пределе можно 1.0)
    sor_w: float      = 1.93         # фактор SOR для уравнения p′
    sor_max: int      = 2000         # максимум итераций SOR
    sor_tol: float    = 1e-10        # критерий SOR по max|Δp'|


class SimpleCPorous2D:
    """
    SIMPLEC для осесимметрии в пористой среде (Дарси–Форшхаймер).
    Моментум линеаризован на гранях:
        S_P,f = (γ_f^2 μ_f/K_f) + 0.5 γ_f^3 C2_f ρ_f |u_old,f|.
    Уравнение p′ собирается по Патанкару–Сполдингу из непрерывности.
    """

    # ----------------------------- init -----------------------------
    def __init__(self, grid: AxiSymmetricGrid, fluid: VacuumResidue, settings: SimpleCPorousSettings):
        self.g   = grid
        self.f   = fluid
        self.set = settings

        self.nr = int(grid.nr)
        self.nz = int(grid.nz)
        self.dr = float(grid.dr)
        self.dz = float(grid.dz)

        # Площади граней (осесимметрия) и объёмы ячеек
        self.A_r = grid.get_face_areas('r')  # (nr+1, nz)
        self.A_z = grid.get_face_areas('z')  # (nr,   nz+1)
        self.Vol = grid.get_cell_volumes()   # (nr,   nz)

    # ----------------------------- utils ----------------------------
    @staticmethod
    def _avg2(aL: np.ndarray, aR: np.ndarray) -> np.ndarray:
        return 0.5 * (aL + aR)

    def _face_avg_col(self, col: np.ndarray) -> np.ndarray:
        """Из столбца центров (nz,) получить значения на осевых гранях (nz+1,)."""
        out = np.empty(self.nz + 1, float)
        out[0] = col[0]; out[-1] = col[-1]
        if self.nz > 1:
            out[1:-1] = 0.5 * (col[:-1] + col[1:])
        return out

    def _face_avg_row(self, row: np.ndarray) -> np.ndarray:
        """Из строки центров (nr,) получить значения на радиальных гранях (nr+1,)."""
        out = np.empty(self.nr + 1, float)
        out[0] = row[0]; out[-1] = row[-1]
        if self.nr > 1:
            out[1:-1] = 0.5 * (row[:-1] + row[1:])
        return out

    # ---------------- коэффициенты Эргуна на ГРАНЯХ ----------------
    def _ergun_on_faces(
        self,
        gamma: np.ndarray,
        mu_cell: np.ndarray,
        rho_cell: np.ndarray,
        u_r: np.ndarray,
        u_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает:
          SPr : (nr+1,nz) — линейный коэф. на r-гранях
          SPz : (nr,nz+1) — линейный коэф. на z-гранях
          rhog_r : (nr+1,nz) — (ρ γ) на r-гранях
          rhog_z : (nr,nz+1) — (ρ γ) на z-гранях
        """
        # средние γ, μ, ρ на гранях
        gam_r = np.zeros((self.nr + 1, self.nz))
        gam_z = np.zeros((self.nr, self.nz + 1))
        mu_r  = np.zeros_like(gam_r); mu_z  = np.zeros_like(gam_z)
        rho_r = np.zeros_like(gam_r); rho_z = np.zeros_like(gam_z)

        for j in range(self.nz):
            col = gamma[:, j]
            mu_col  = mu_cell[:, j]
            rho_col = rho_cell[:, j]
            gam_r[:, j] = self._face_avg_row(col)
            mu_r[:, j]  = self._face_avg_row(mu_col)
            rho_r[:, j] = self._face_avg_row(rho_col)

        for i in range(self.nr):
            row = gamma[i, :]
            mu_row  = mu_cell[i, :]
            rho_row = rho_cell[i, :]
            gam_z[i, :] = self._face_avg_col(row)
            mu_z[i, :]  = self._face_avg_col(mu_row)
            rho_z[i, :] = self._face_avg_col(rho_row)

        gam_r = np.clip(gam_r, self.set.gamma_min, 0.999999)
        gam_z = np.clip(gam_z, self.set.gamma_min, 0.999999)

        # Эргун (K и C2) из пористости на гранях
        K_r  = PorousDrag.ergun_permeability(gam_r, self.set.dp_particle)
        C2_r = PorousDrag.ergun_inertial   (gam_r, self.set.dp_particle)
        K_z  = PorousDrag.ergun_permeability(gam_z, self.set.dp_particle)
        C2_z = PorousDrag.ergun_inertial   (gam_z, self.set.dp_particle)

        # линейный коэффициент S_P,f
        SPr = (gam_r**2) * mu_r / np.maximum(K_r, 1e-30) \
              + 0.5 * (gam_r**3) * C2_r * rho_r * np.abs(u_r)
        SPz = (gam_z**2) * mu_z / np.maximum(K_z, 1e-30) \
              + 0.5 * (gam_z**3) * C2_z * rho_z * np.abs(u_z)

        return SPr, SPz, rho_r * gam_r, rho_z * gam_z

    # -------------------- уравнение p'-коррекции --------------------
    def _solve_pressure_correction(
        self,
        m_res: np.ndarray,
        d_r: np.ndarray,
        d_z: np.ndarray,
        rho_g_r: np.ndarray,
        rho_g_z: np.ndarray,
        inlet_is_fixed_flux: bool = True,
    ) -> np.ndarray:
        """
        Решаем: div[ (ργ)_f d_f grad(p') ] = m_res / ΔV
        где d_f = A_f / S_P,f. Схема — SOR.
        """
        nr, nz = self.nr, self.nz
        dr, dz = self.dr, self.dz

        # Коэффициенты на ячейку
        aE = np.zeros((nr, nz))  # восток  (j+1/2)
        aW = np.zeros((nr, nz))  # запад   (j-1/2)
        aN = np.zeros((nr, nz))  # север   (i+1/2)
        aS = np.zeros((nr, nz))  # юг      (i-1/2)

        # осевые лица → aE/aW
        aW[:, 1:] = rho_g_z[:, 1:nz] * d_z[:, 1:nz] / dz   # для j>=1
        aE[:, :-1] = rho_g_z[:, 1:nz] * d_z[:, 1:nz] / dz  # для j<=nz-2 (тот же набор лиц)
        # радиальные лица → aN/aS
        aS[1:, :] = rho_g_r[1:nr, :] * d_r[1:nr, :] / dr   # для i>=1
        aN[:-1, :] = rho_g_r[1:nr, :] * d_r[1:nr, :] / dr  # для i<=nr-2

        # ГУ: ось и стенка — Neumann (нет коррекции через r-грань)
        aS[0, :]  = 0.0     # ось r=0
        aN[-1, :] = 0.0     # стенка r=R

        # Правая часть (массовая невязка на объём)
        b = m_res / np.maximum(self.Vol, 1e-30)

        # Главный коэффициент
        aP = aE + aW + aN + aS

        # Вход — фиксированный расход → западная грань «заморожена»
        if inlet_is_fixed_flux:
            aP[:, 0] -= aW[:, 0]
            aW[:, 0] = 0.0

        # SOR для p′
        pcor = np.zeros((nr, nz))
        omg = float(self.set.sor_w)
        for _ in range(int(self.set.sor_max)):
            max_corr = 0.0
            for i in range(nr):
                for j in range(nz):
                    ap = aP[i, j] + 1e-30
                    rhs = b[i, j]
                    if j > 0:       rhs += aW[i, j] * pcor[i, j - 1]
                    if j < nz - 1:  rhs += aE[i, j] * pcor[i, j + 1]  # p′=0 на выходе учитывается отсутствием члена
                    if i > 0:       rhs += aS[i, j] * pcor[i - 1, j]
                    if i < nr - 1:  rhs += aN[i, j] * pcor[i + 1, j]
                    p_new = (1.0 - omg) * pcor[i, j] + omg * (rhs / ap)
                    max_corr = max(max_corr, abs(p_new - pcor[i, j]))
                    pcor[i, j] = p_new
            if max_corr < self.set.sor_tol:
                break
        return pcor

    # ----------------------------- основной шаг -----------------------------
    def solve(
        self,
        T: np.ndarray,
        gamma: np.ndarray,
        m_dot_in: float,
        p_init: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Параметры
        ---------
        T        : (nr,nz) — температура (для ρ(T), μ(T))
        gamma    : (nr,nz) — жидкая пористость (ε≈γ)
        m_dot_in : float   — массовый расход через вход (нижние z-граня)
        p_init   : (nr,nz) — начальное поле давления (опц.)

        Возвращает
        ---------
        vr : (nr+1,nz) — радиальная скорость на r-гранях
        vz : (nr,nz+1) — осевая скорость на z-гранях
        p  : (nr,nz)   — давление (p(H)=0 как референс)
        info : dict     — сводные метрики (сходимость по массе и пр.)
        """
        nr, nz = self.nr, self.nz
        dr, dz = self.dr, self.dz

        # Свойства в ячейках
        mu  = self.f.viscosity(T)
        rho = self.f.density(T)

        # Стартовые поля
        p  = np.zeros((nr, nz), float) if p_init is None else p_init.copy()
        vr = np.zeros((nr + 1, nz), float)
        vz = np.zeros((nr, nz + 1), float)

        # Начальное распределение входной скорости (plug по γ)
        rho_in_col = rho[:, 0]
        gam_in_col = np.clip(gamma[:, 0], self.set.gamma_min, 1.0)
        w_in = rho_in_col * gam_in_col * self.A_z[:, 0]  # веса на нижних z-гранях
        vz[:, 0] = m_dot_in / (np.sum(w_in) + 1e-30)

        # Внешние итерации SIMPLEC
        mass_hist: list[float] = []
        for _ in range(int(self.set.max_outer)):
            # 1) S_P на гранях и (ργ)_f
            SPr, SPz, rhog_r, rhog_z = self._ergun_on_faces(gamma, mu, rho, vr, vz)

            # 2) Прогноз скоростей u* из текущего p
            #   радиальные грани (внутренние)
            for i in range(1, nr):
                dpdr = (p[i, :] - p[i - 1, :]) / dr
                vr[i, :] = (1.0 - self.set.relax_u) * vr[i, :] - self.set.relax_u * dpdr / (SPr[i, :] + 1e-30)
            vr[0, :]  = 0.0           # ось
            vr[-1, :] = 0.0           # стенка (no-slip)
            #   осевые грани (внутренние)
            for j in range(1, nz):
                dpdz = (p[:, j] - p[:, j - 1]) / dz
                vz[:, j] = (1.0 - self.set.relax_u) * vz[:, j] - self.set.relax_u * dpdz / (SPz[:, j] + 1e-30)
            #   вход: перенастроить так, чтобы Σ ρ γ vz A = ṁ_in
            mass_face_in = float(np.sum(rhog_z[:, 0] * vz[:, 0] * self.A_z[:, 0]))
            if abs(mass_face_in - m_dot_in) / max(abs(m_dot_in), 1e-30) > 1e-14:
                vz[:, 0] *= m_dot_in / (mass_face_in + 1e-30)

            # 3) Массовые потоки m* и невязка в ячейках
            m_r = rhog_r * vr * self.A_r  # (nr+1,nz)
            m_z = rhog_z * vz * self.A_z  # (nr,  nz+1)
            m_res = (
                (m_r[:-1, :] - m_r[1:, :]) +  # приток–отток по r
                (m_z[:, :-1] - m_z[:, 1:])    # приток–отток по z
            )

            # 4) Уравнение p′-коррекции и SOR
            d_r = self.A_r / (SPr + 1e-30)
            d_z = self.A_z / (SPz + 1e-30)
            pcor = self._solve_pressure_correction(m_res, d_r, d_z, rhog_r, rhog_z, inlet_is_fixed_flux=True)

            # 5) Коррекция p и u
            p += self.set.relax_p * pcor
            #   радиальные грани
            for i in range(1, nr):
                dpdr_cor = (pcor[i, :] - pcor[i - 1, :]) / dr
                vr[i, :] -= dpdr_cor / (SPr[i, :] + 1e-30)
            vr[0, :]  = 0.0
            vr[-1, :] = 0.0
            #   осевые грани
            for j in range(1, nz):
                dpdz_cor = (pcor[:, j] - pcor[:, j - 1]) / dz
                vz[:, j] -= dpdz_cor / (SPz[:, j] + 1e-30)
            #   вход: жёсткая нормировка ṁ
            mass_face_in = float(np.sum(rhog_z[:, 0] * vz[:, 0] * self.A_z[:, 0]))
            vz[:, 0] *= m_dot_in / (mass_face_in + 1e-30)

            # 6) Контроль сходимости по массе (max_rel)
            mr_in  = m_r[:-1, :]
            mr_out = m_r[1:, :]
            mz_in  = m_z[:, :-1]
            mz_out = m_z[:, 1:]
            cell_res = np.abs((mr_in - mr_out) + (mz_in - mz_out))
            max_abs = float(cell_res.max())
            max_rel = max_abs / (abs(m_dot_in) + 1e-30)
            mass_hist.append(max_rel)
            if max_rel < self.set.tol_m:
                break

        info = dict(
            outer_iters=len(mass_hist),
            mass_res_hist=np.array(mass_hist, float),
            max_rel_mass_res=mass_hist[-1] if mass_hist else np.nan,
        )
        return vr, vz, p, info
