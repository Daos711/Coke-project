# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from mesh.grid_2d import AxiSymmetricGrid
from physics.properties import VacuumResidue
from physics.correlations import PorousDrag

# =========================
# НАСТРОЙКИ МОМЕНТУМА
# =========================

@dataclass
class MomentumSettings:
    p_outlet: float = 0.0         # давление на выходе [Па]
    diameter: float = 0.0602      # диаметр реактора [м]
    dp_particle: float = 1.0e-3   # экв. диаметр частиц (Эргун) [м]
    gamma_min: float = 1.0e-3     # нижняя планка пористости
    rho_ref_mode: str = "inlet"   # 'inlet' или 'mean' — для plug-скорости


class MomentumBrinkman:
    """
    Осевая (compute) и радиальная (compute_radial_analytic) модели импульса
    на основе Бринкмана + Эргуна с авто-переключением в режим трубы (Пуазёйль)
    при высокой пористости (gamma ~ 1).
    """
    def __init__(self, grid, fluid, settings):
        # --- сетка/геометрия ---
        self.g = grid
        self.grid = grid
        self.nr = grid.nr
        self.nz = grid.nz
        self.R = float(grid.radius)         # радиус
        self.D = float(settings.diameter)   # диаметр для формулы Пуазёйля

        # dz может быть float либо массив (nz,)
        dz_raw = np.asarray(getattr(grid, "dz"))
        self.dz = np.full(self.nz, float(dz_raw)) if dz_raw.ndim == 0 else dz_raw.astype(float).copy()

        self.zc = np.asarray(grid.z_centers, dtype=float)
        self.rc = np.asarray(grid.r_centers, dtype=float)

        # --- физика ---
        self.fluid = fluid
        self.set = settings

        # предвычисления
        self.area_full = np.pi * self.R**2

    # ----------------- служебные формулы -----------------
    @staticmethod
    def _ergun_K_C2(eps: float, dp: float) -> Tuple[float, float]:
        """(K, C2) из классической формы Эргуна для интринзик-скорости."""
        eps = float(np.clip(eps, 1e-9, 0.99))
        K  = (eps**3 * dp**2) / (150.0 * (1.0 - eps)**2 + 1e-30)        # м^2
        C2 = 1.75 * (1.0 - eps) / (eps**3 * dp + 1e-30)                 # 1/м
        return K, C2

    @staticmethod
    def _dpdz_poiseuille(mu: float, Umean: float, D: float) -> float:
        """Ламинарная труба: dp/dz = 32 μ Ū / D²."""
        return 32.0 * mu * Umean / (D**2 + 1e-30)

    def _free_area(self, gamma_col: np.ndarray) -> Tuple[float, float]:
        eps = float(np.clip(np.mean(gamma_col), self.set.gamma_min, 1.0))
        return eps * self.area_full, eps

    # ----------------- ОСЕВАЯ МОДЕЛЬ -----------------
    def compute(
        self, T: np.ndarray, gamma: np.ndarray, m_dot: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Осевая модель dp/dz. При eps>=0.97 используется Пуазёйль,
        иначе — Эргун (вязк.+инерц.).
        Возвращает:
          p_z: (nz,) давление вдоль оси
          v_rz: (nr,nz) поле скорости 'plug' (по радиусу одинаково)
          dpdz: (nz,) профиль градиента давления
          info: словарь
        """
        Tm  = T.mean(axis=0)                  # (nz,)
        mu_z  = self.fluid.viscosity(Tm)      # (nz,)
        rho_z = self.fluid.density(Tm)        # (nz,)

        # плотность для вычисления plug-скорости
        if self.set.rho_ref_mode == "mean":
            rho_ref = float(np.mean(rho_z))
        else:
            rho_ref = float(self.fluid.density(float(T[:, 0].mean())))

        v_z  = np.zeros(self.nz)
        dpdz = np.zeros(self.nz)
        eps_z = np.zeros(self.nz)
        K_z  = np.zeros(self.nz)
        C2_z = np.zeros(self.nz)
        mode = np.empty(self.nz, dtype=object)

        for j in range(self.nz):
            A_free, eps = self._free_area(gamma[:, j])
            eps_z[j] = eps
            v_z[j] = m_dot / (rho_ref * A_free + 1e-30)  # интринзик Ū(z)

            if eps >= 0.97:
                # режим пустой трубы
                dpdz[j] = self._dpdz_poiseuille(mu_z[j], v_z[j], self.D)
                K_z[j], C2_z[j] = np.inf, 0.0
                mode[j] = "pipe"
            else:
                # пористый слой (Эргун)
                K, C2 = self._ergun_K_C2(eps, self.set.dp_particle)
                K_z[j], C2_z[j] = K, C2
                dpdz_vis = mu_z[j] * v_z[j] / (K + 1e-30)
                dpdz_in  = 0.5 * C2 * rho_z[j] * abs(v_z[j]) * v_z[j]
                dpdz[j]  = dpdz_vis + dpdz_in
                mode[j] = "porous"

        # давление интегрируем от выхода вверх
        p_z = np.zeros(self.nz)
        p_z[-1] = self.set.p_outlet
        for j in range(self.nz - 2, -1, -1):
            p_z[j] = p_z[j + 1] + dpdz[j] * self.dz[j]

        v_rz = np.tile(v_z, (self.nr, 1))
        info = dict(
            rho_ref=rho_ref, mu_z=mu_z, rho_z=rho_z, eps_z=eps_z,
            K_z=K_z, C2_z=C2_z, mode=mode,
            v_min=float(v_z.min()), v_max=float(v_z.max()),
            dpdz_vis_mean=float(np.mean(np.where(np.isfinite(K_z), mu_z * v_z / (K_z + 1e-30), 0.0))),
            dpdz_in_mean =float(np.mean(0.5 * C2_z * rho_z * np.abs(v_z) * v_z)),
            delta_p=float(np.trapz(dpdz, self.zc)),
        )
        return p_z, v_rz, dpdz, info

    # --------------- РАДИАЛЬНАЯ АНАЛИТИЧЕСКАЯ МОДЕЛЬ ---------------
    def compute_radial_analytic(
        self, T: np.ndarray, gamma: np.ndarray, m_dot: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Радиальный профиль u(r,z):
          • при eps>=0.97 — ламинарная труба (Пуазёйль): u(r)=2Ū(1-(r/R)^2)
          • при eps<0.97 — Бринкман + Эргун, форма через модифиц. Бессель I0,
            нормировка по Ū(ṁ, ρ(T), eps)
        """
        try:
            from numpy import i0 as I0
        except Exception:
            from scipy.special import i0 as I0  # type: ignore

        r, z, R = self.rc, self.zc, self.R
        Tm = T.mean(axis=0)
        mu_z  = self.fluid.viscosity(Tm)
        rho_z = self.fluid.density(Tm)

        # средняя скорость из точного m_dot на колонке j
        eps_z = np.clip(gamma.mean(axis=0), self.set.gamma_min, 1.0)  # (nz,)
        A_free = eps_z * self.area_full
        vbar_z = m_dot / (rho_z * A_free + 1e-30)  # (nz,)

        u_rz = np.zeros((self.nr, self.nz))
        dpdz_z = np.zeros(self.nz)
        K_z = np.zeros(self.nz)
        C2_z = np.zeros(self.nz)
        mode = np.empty(self.nz, dtype=object)

        for j in range(self.nz):
            mu, rho, U = float(mu_z[j]), float(rho_z[j]), float(vbar_z[j])
            eps = float(np.clip(eps_z[j], 1e-9, 1.0))
            if eps >= 0.97:
                u_rz[:, j] = 2.0 * U * (1.0 - (r / R)**2)
                dpdz_z[j]  = self._dpdz_poiseuille(mu, U, self.D)
                K_z[j], C2_z[j] = np.inf, 0.0
                mode[j] = "pipe"
            else:
                K, C2 = self._ergun_K_C2(eps, self.set.dp_particle)
                K_z[j], C2_z[j] = K, C2
                dpdz_vis = mu * U / (K + 1e-30)
                dpdz_in  = 0.5 * C2 * rho * abs(U) * U
                dpdz = dpdz_vis + dpdz_in
                dpdz_z[j] = dpdz
                rootK = np.sqrt(K)
                S = 1.0 - I0(r / (rootK + 1e-30)) / (I0(R / (rootK + 1e-30)) + 1e-300)
                # нормировка, чтобы ⟨u⟩ = U
                Sbar = (2.0 / (R**2 + 1e-30)) * np.trapz(r * S, r)
                u_rz[:, j] = (U / (Sbar + 1e-30)) * S
                mode[j] = "porous"

        # давление вдоль оси
        p_z = np.zeros(self.nz)
        p_z[-1] = self.set.p_outlet
        for j in range(self.nz - 2, -1, -1):
            p_z[j] = p_z[j + 1] + dpdz_z[j] * self.dz[j]

        info = dict(
            mu_z=mu_z, rho_z=rho_z, eps_z=eps_z, vbar_z=vbar_z,
            K_z=K_z, C2_z=C2_z, mode=mode,
            u_min=float(u_rz.min()), u_max=float(u_rz.max()),
            delta_p=float(np.trapz(dpdz_z, z)),
        )
        return p_z, u_rz, dpdz_z, info


# =========================
# НАСТРОЙКИ РАДИАЛЬНОГО
# =========================

@dataclass
class RadialSettings:
    dp_particle: float = 1.0e-3
    gamma_min: float = 1e-3
    max_outer: int = 80
    max_inner: int = 80
    tol_u: float = 1e-10
    tol_mdot: float = 1e-6
    wall_bc: float = 0.0  # no-slip


class BrinkmanRadialSolver:
    """
    Радиальный профиль u(r,z) из уравнения Бринкмана
      μ γ * (1/r d/dr (r du/dr)) - [ γ^2 μ/K + 0.5 γ^3 C2 ρ |u| ] * u = γ * (dp/dz)
    Для каждого z подбираем dp/dz так, чтобы ∫ ρ γ u dA = m_dot (строго).
    """
    def __init__(self, grid: AxiSymmetricGrid, fluid: VacuumResidue, settings: RadialSettings):
        self.grid = grid
        self.fluid = fluid
        self.set = settings
        self.R  = float(self.grid.radius)
        self.dr = float(self.grid.dr)
        self.r  = self.grid.r_centers.astype(float)  # площадь кольцевой ячейки: dA = 2π r dr

    @staticmethod
    def _ring_area(r: np.ndarray, dr: float) -> np.ndarray:
        return 2.0 * np.pi * r * dr

    def _g_bar(self, gamma_col: np.ndarray) -> float:
        return float(np.clip(np.mean(gamma_col), self.set.gamma_min, 0.999999))

    # один линейный шаг Пикара при фиксированном dp/dz
    def _solve_radial_linear(self, mu: float, rho: float, g_bar: float, dpdz: float, u_old: np.ndarray) -> np.ndarray:
        nr, dr, r = self.r.size, self.dr, self.r
        K  = float(PorousDrag.ergun_permeability(g_bar, self.set.dp_particle))
        C2 = float(PorousDrag.ergun_inertial   (g_bar, self.set.dp_particle))
        A = np.zeros((nr, nr), dtype=float)
        b = np.zeros(nr, dtype=float)
        for i in range(nr):
            r_i  = r[i]
            r_ip = r_i + 0.5 * dr
            r_im = max(r_i - 0.5 * dr, 0.0)
            # диффузионные коэффициенты (радиальная FV-аппроксимация)
            aE = mu * g_bar * (r_ip / (max(r_i, 1e-16) * dr * dr))
            aW = mu * g_bar * (r_im / (max(r_i, 1e-16) * dr * dr))
            # источник (Дарси + линеаризованный Форшхаймер)
            S_P = g_bar * g_bar * mu / max(K, 1e-30) + 0.5 * (g_bar ** 3) * C2 * rho * abs(u_old[i])
            aP = aW + aE + S_P
            A[i, i] = aP
            # осевая симметрия в центре: du/dr|_{r=0}=0
            if i == 0:
                A[i, i] += aW
            else:
                A[i, i - 1] = -aW
            # no-slip на стенке: u(R)=0 (Дирихле)
            if i == nr - 1:
                A[i, i] += aE
                b[i] += aE * self.set.wall_bc
            else:
                A[i, i + 1] = -aE
            b[i] += g_bar * dpdz
        return np.linalg.solve(A, b)

    # решаем уравнение для заданного dp/dz → (u, m_calc)
    def _solve_for_dpdz(self, dpdz: float, mu: float, rho: float, g_bar: float, u_guess: np.ndarray, area_r: np.ndarray) -> Tuple[np.ndarray, float]:
        u_old = u_guess.copy()
        for _ in range(self.set.max_inner):
            u_new = self._solve_radial_linear(mu, rho, g_bar, dpdz, u_old)
            if np.max(np.abs(u_new - u_old)) < self.set.tol_u:
                break
            u_old = 0.5 * u_old + 0.5 * u_new
        u = u_new
        m_calc = rho * float(np.sum(g_bar * u * area_r))
        return u, m_calc

    # главный метод: подбор dp/dz (бракетинг + секущая с Illinois)
    def compute(self, T: np.ndarray, gamma: np.ndarray, m_dot: float, dpdz_init: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        NR, NZ = T.shape
        u = np.zeros_like(T)
        dpdz = np.asarray(dpdz_init, dtype=float).copy()
        area_r = self._ring_area(self.r, self.dr)
        mu_z  = self.fluid.viscosity(np.mean(T, axis=0))
        rho_z = self.fluid.density (np.mean(T, axis=0))
        mdot_err_max = 0.0
        outer_iters_max = 0

        for j in range(NZ):
            g_bar = self._g_bar(gamma[:, j])
            mu = float(mu_z[j]); rho = float(rho_z[j])

            # стартовый plug-профиль
            v_guess = m_dot / (rho * g_bar * (np.pi * self.R ** 2) + 1e-30)
            u_guess = np.full(NR, v_guess, dtype=float)

            # первичная оценка dp/dz (Дарси + Эргун)
            K  = float(PorousDrag.ergun_permeability(g_bar, self.set.dp_particle))
            C2 = float(PorousDrag.ergun_inertial   (g_bar, self.set.dp_particle))
            dp_est = abs(mu * v_guess / max(K, 1e-30) + 0.5 * C2 * rho * abs(v_guess) * v_guess)

            # берём максимум из переданного и оценки
            dp0 = max(1e-12, float(dpdz[j]), dp_est)
            u0, m0 = self._solve_for_dpdz(dp0, mu, rho, g_bar, u_guess, area_r)
            if abs(m0 - m_dot) / max(abs(m_dot), 1e-30) < self.set.tol_mdot:
                u[:, j] = u0; dpdz[j] = dp0; continue

            # вторая точка — масштабирование по расходу
            dp1 = dp0 * (m_dot / max(m0, 1e-30))
            u1, m1 = self._solve_for_dpdz(dp1, mu, rho, g_bar, u0, area_r)

            f0, f1 = (m0 - m_dot), (m1 - m_dot)
            dpL, fL, uL = (dp0, f0, u0)
            dpR, fR, uR = (dp1, f1, u1)

            # расширяем интервал до смены знака
            expand = 0
            while fL * fR > 0.0 and expand < 20:
                if abs(fL) < abs(fR):
                    dpR *= 2.0
                    uR, mR = self._solve_for_dpdz(dpR, mu, rho, g_bar, uR, area_r)
                    fR = mR - m_dot
                else:
                    dpL *= 0.5
                    uL, mL = self._solve_for_dpdz(dpL, mu, rho, g_bar, uL, area_r)
                    fL = mL - m_dot
                expand += 1

            iters = 0
            while iters < self.set.max_outer:
                iters += 1
                # regula falsi (Illinois)
                dp = dpR - fR * (dpR - dpL) / max((fR - fL), 1e-30)
                u_try, m_try = self._solve_for_dpdz(dp, mu, rho, g_bar, 0.5 * (uL + uR), area_r)
                f = m_try - m_dot
                if abs(f) / max(abs(m_dot), 1e-30) < self.set.tol_mdot:
                    u[:, j] = u_try; dpdz[j] = dp
                    mdot_err_max = max(mdot_err_max, abs(f) / max(abs(m_dot), 1e-30))
                    break
                if f * fL < 0.0:
                    dpR, fR, uR = dp, f, u_try
                    fL *= 0.5
                else:
                    dpL, fL, uL = dp, f, u_try
                    fR *= 0.5
            outer_iters_max = max(outer_iters_max, iters)

            if iters >= self.set.max_outer:
                dp_mid = 0.5 * (dpL + dpR)
                u_mid, m_mid = self._solve_for_dpdz(dp_mid, mu, rho, g_bar, 0.5 * (uL + uR), area_r)
                u[:, j] = u_mid; dpdz[j] = dp_mid
                mdot_err_max = max(mdot_err_max, abs(m_mid - m_dot) / max(abs(m_dot), 1e-30))

        # давление от выхода вверх
        pz = np.zeros(NZ)
        for j in range(NZ - 1, 0, -1):
            pz[j - 1] = pz[j] + dpdz[j] * float(self.grid.dz)
        p = np.repeat(pz[np.newaxis, :], NR, axis=0)

        info = {
            "mdot_rel_error_max": float(mdot_err_max * 100.0),  # %
            "outer_iters_max": int(outer_iters_max),
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "rho_z": rho_z,
            "mu_z": mu_z
        }
        return u, p, dpdz, info


# =========================
# ИНТЕРФЕЙС ДЛЯ ЭНЕРГИИ
# =========================

def u_to_face_velocities(
    u_c: np.ndarray,
    grid: AxiSymmetricGrid,
    gamma: np.ndarray,
    rho_ref: float,
    m_dot: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переносим u(r,z) в центрах -> на грани по z.
    Массовый расход через каждую грань нормируем к m_dot (по rho_ref),
    но только если расхождение заметно.
    """
    NR, NZ = u_c.shape
    r  = grid.r_centers.astype(float)
    dr = float(grid.dr)
    ring_area = 2.0 * np.pi * r * dr

    ur_face = np.zeros((NR + 1, NZ), dtype=float)
    uz_face = np.zeros((NR, NZ + 1), dtype=float)

    Q_target = m_dot / max(rho_ref, 1e-30)
    for jf in range(NZ + 1):
        jL, jR = max(jf - 1, 0), min(jf, NZ - 1)
        u_face = 0.5 * (u_c[:, jL] + u_c[:, jR])
        g_face = 0.5 * (gamma[:, jL] + gamma[:, jR])
        Q_raw  = float(np.sum(g_face * u_face * ring_area))  # м^3/с
        rel = abs(Q_raw - Q_target) / max(abs(Q_target), 1e-30)
        scale = (Q_target / max(Q_raw, 1e-30)) if rel > 1e-6 else 1.0
        uz_face[:, jf] = scale * u_face
    return ur_face, uz_face
