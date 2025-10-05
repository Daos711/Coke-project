# -*- coding: utf-8 -*-
"""
Солвер импульса для двух флюидных фаз в пористой среде.
Реализация уравнений (5) из статьи Díaz et al. с алгоритмом SIMPLEC.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Импортируем существующие корреляции
from physics.correlations import PorousDrag, DragCoefficients


@dataclass
class SimpleC2PSettings:
    """Настройки SIMPLEC солвера для двух фаз."""
    dp_particle: float = 1e-3  # Диаметр частиц кокса, м
    gamma_cut: float = 1e-3  # Минимальная пористость
    alpha_cut: float = 1e-6  # Минимальная объёмная доля
    max_outer: int = 200  # Макс. внешних итераций SIMPLEC
    max_inner: int = 100  # Макс. внутренних итераций для p'
    tol_m: float = 1e-8  # Критерий по невязке массы
    tol_p: float = 1e-10  # Критерий для решателя давления
    relax_p: float = 0.7  # Релаксация давления
    relax_u: float = 0.8  # Релаксация скорости
    use_pea: bool = True  # Использовать PEA для межфазного обмена


def porous_sink_coeffs(alpha_n: np.ndarray, gamma: np.ndarray,
                       mu_n: float, rho_n: float,
                       vmag_n: np.ndarray, d_p: float) -> np.ndarray:
    """
    Коэффициент сопротивления пористой среды для фазы n.
    Возвращает C [кг/(м³·с)] для члена: S_porous = -C * v_n
    """
    eps = 1e-30

    # Используем существующие функции из PorousDrag
    k = PorousDrag.ergun_permeability(gamma, d_p)

    # Вязкий член (Дарси)
    C_mu = gamma ** 2 * mu_n / (k + eps)

    # Инерционный член (Форхгеймер)
    C2 = PorousDrag.ergun_inertial_c2(gamma, d_p)
    C_in = gamma ** 3 * C2 * rho_n * vmag_n / 2.0

    # Полный коэффициент с учетом объёмной доли
    C = alpha_n * (C_mu + C_in)

    return C


def symmetric_interphase_K(alpha_m: np.ndarray, alpha_n: np.ndarray,
                           rho_m: float, rho_n: float,
                           mu_m: float, mu_n: float,
                           v_rel: np.ndarray, d_eq: float = 1e-3) -> np.ndarray:
    """
    Межфазный коэффициент обмена импульсом.
    Использует DragCoefficients.symmetric_model с параметром d_eq.
    """
    # Напрямую используем улучшенный метод из correlations
    K = DragCoefficients.symmetric_model(
        alpha_m, alpha_n,
        rho_m, rho_n,
        mu_m, mu_n,
        v_rel,
        d_eq=d_eq  # Передаём d_eq для использования улучшенной версии
    )
    return K


def build_momentum_matrix(grid, fields: Dict, props: Dict,
                          phase: str, cfg: SimpleC2PSettings) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Построение матрицы и правой части для уравнения импульса фазы.

    phase: 'R' или 'D'
    Возвращает: (A, b) для системы A*v = b
    """
    NR, NZ = fields['alpha_R'].shape
    N = NR * NZ

    # Получаем поля для текущей фазы
    alpha_n = fields[f'alpha_{phase}']
    v_r_n = fields[f'v_{phase}_r']
    v_z_n = fields[f'v_{phase}_z']
    rho_n = props[f'rho_{phase}']
    mu_n = props[f'mu_{phase}']
    gamma = fields['gamma']

    # Другая фаза
    other_phase = 'D' if phase == 'R' else 'R'
    alpha_m = fields[f'alpha_{other_phase}']
    v_r_m = fields[f'v_{other_phase}_r']
    v_z_m = fields[f'v_{other_phase}_z']
    rho_m = props[f'rho_{other_phase}']
    mu_m = props[f'mu_{other_phase}']

    # Размеры ячеек
    dr = grid.dr if hasattr(grid, 'dr') else 0.0301 / NR
    dz = grid.dz if hasattr(grid, 'dz') else 0.5692 / NZ

    # Матрица в формате COO для удобства заполнения
    row_indices = []
    col_indices = []
    data = []
    b = np.zeros(N * 2)  # Для v_r и v_z

    # Заполняем матрицу по ячейкам
    for i in range(NR):
        for j in range(NZ):
            idx = i * NZ + j

            # Проверка на отсутствие фазы
            if alpha_n[i, j] < cfg.alpha_cut or gamma[i, j] < cfg.gamma_cut:
                # Фаза отсутствует - зануляем уравнение
                # v_r = 0
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(1.0)
                b[idx] = 0.0

                # v_z = 0
                row_indices.append(N + idx)
                col_indices.append(N + idx)
                data.append(1.0)
                b[N + idx] = 0.0
                continue

            # Скорость фазы
            vmag_n = np.sqrt(v_r_n[i, j] ** 2 + v_z_n[i, j] ** 2)
            vmag_m = np.sqrt(v_r_m[i, j] ** 2 + v_z_m[i, j] ** 2)
            v_rel = np.sqrt((v_r_n[i, j] - v_r_m[i, j]) ** 2 +
                            (v_z_n[i, j] - v_z_m[i, j]) ** 2)

            # Коэффициенты сопротивления
            C_porous = porous_sink_coeffs(alpha_n[i, j], gamma[i, j],
                                          mu_n, rho_n, vmag_n, cfg.dp_particle)

            K_mn = symmetric_interphase_K(alpha_n[i, j], alpha_m[i, j],
                                          rho_n, rho_m, mu_n, mu_m,
                                          v_rel, cfg.dp_particle)

            # Диагональный коэффициент (с PEA)
            if cfg.use_pea:
                a_p = C_porous + gamma[i, j] * K_mn
            else:
                a_p = C_porous

            # === Уравнение для v_r ===
            row = idx

            # Диагональ
            row_indices.append(row)
            col_indices.append(idx)
            data.append(a_p)

            # Источник от межфазного обмена (PEA)
            if cfg.use_pea:
                b[row] += gamma[i, j] * K_mn * v_r_m[i, j]

            # Гравитация (если есть радиальная компонента - обычно нет)
            # b[row] += gamma[i, j] * alpha_n[i, j] * rho_n * g_r

            # === Уравнение для v_z ===
            row = N + idx

            # Диагональ
            row_indices.append(row)
            col_indices.append(N + idx)
            data.append(a_p)

            # Источник от межфазного обмена (PEA)
            if cfg.use_pea:
                b[row] += gamma[i, j] * K_mn * v_z_m[i, j]

            # Гравитация (вертикальная)
            g_z = -9.81  # м/с²
            b[row] += gamma[i, j] * alpha_n[i, j] * rho_n * g_z

    # Создаём разреженную матрицу
    A = sp.coo_matrix((data, (row_indices, col_indices)),
                      shape=(N * 2, N * 2)).tocsr()

    return A, b


def solve_pressure_correction(grid, fields: Dict, props: Dict,
                              v_star: Dict, a_p: Dict,
                              cfg: SimpleC2PSettings) -> np.ndarray:
    """
    Решение уравнения для поправки давления p'.
    Использует уравнение непрерывности для смеси фаз.
    """
    NR, NZ = fields['alpha_R'].shape
    N = NR * NZ

    dr = grid.dr if hasattr(grid, 'dr') else 0.0301 / NR
    dz = grid.dz if hasattr(grid, 'dz') else 0.5692 / NZ

    # Построение матрицы для уравнения Пуассона
    row_indices = []
    col_indices = []
    data = []
    b = np.zeros(N)

    for i in range(NR):
        for j in range(NZ):
            idx = i * NZ + j

            # Коэффициенты d_n = (gamma * alpha_n) / a_p_n
            gamma = fields['gamma'][i, j]

            d_R = gamma * fields['alpha_R'][i, j] / (a_p['R'][i, j] + 1e-30)
            d_D = gamma * fields['alpha_D'][i, j] / (a_p['D'][i, j] + 1e-30)

            # Сумма коэффициентов для смеси
            d_sum = d_R + d_D

            # Центральный коэффициент
            a_P = 0.0

            # Соседи по r
            if i > 0:
                a_W = d_sum / dr ** 2
                a_P += a_W
                row_indices.append(idx)
                col_indices.append((i - 1) * NZ + j)
                data.append(-a_W)

            if i < NR - 1:
                a_E = d_sum / dr ** 2
                a_P += a_E
                row_indices.append(idx)
                col_indices.append((i + 1) * NZ + j)
                data.append(-a_E)

            # Соседи по z
            if j > 0:
                a_S = d_sum / dz ** 2
                a_P += a_S
                row_indices.append(idx)
                col_indices.append(i * NZ + (j - 1))
                data.append(-a_S)

            if j < NZ - 1:
                a_N = d_sum / dz ** 2
                a_P += a_N
                row_indices.append(idx)
                col_indices.append(i * NZ + (j + 1))
                data.append(-a_N)

            # Диагональ
            row_indices.append(idx)
            col_indices.append(idx)
            data.append(a_P if a_P > 0 else 1.0)  # Защита от нулевой диагонали

            # Правая часть - невязка по массе
            # div(rho_R * v_R* + rho_D * v_D*)
            if a_P > 0:
                # Вычисляем дивергенцию скоростей
                div_flux = 0.0

                # Потоки для фазы R
                if i > 0:
                    div_flux += props['rho_R'] * fields['alpha_R'][i, j] * v_star['R_r'][i, j] / dr
                if i < NR - 1:
                    div_flux -= props['rho_R'] * fields['alpha_R'][i, j] * v_star['R_r'][i, j] / dr
                if j > 0:
                    div_flux += props['rho_R'] * fields['alpha_R'][i, j] * v_star['R_z'][i, j] / dz
                if j < NZ - 1:
                    div_flux -= props['rho_R'] * fields['alpha_R'][i, j] * v_star['R_z'][i, j] / dz

                # Потоки для фазы D
                if i > 0:
                    div_flux += props['rho_D'] * fields['alpha_D'][i, j] * v_star['D_r'][i, j] / dr
                if i < NR - 1:
                    div_flux -= props['rho_D'] * fields['alpha_D'][i, j] * v_star['D_r'][i, j] / dr
                if j > 0:
                    div_flux += props['rho_D'] * fields['alpha_D'][i, j] * v_star['D_z'][i, j] / dz
                if j < NZ - 1:
                    div_flux -= props['rho_D'] * fields['alpha_D'][i, j] * v_star['D_z'][i, j] / dz

                b[idx] = div_flux
            else:
                b[idx] = 0.0

    # Фиксируем давление в одной точке (reference)
    ref_idx = (NR - 1) * NZ + (NZ - 1)  # Верхний правый угол
    row_indices.append(ref_idx)
    col_indices.append(ref_idx)
    data.append(1e10)  # Большое число для фиксации
    b[ref_idx] = 0.0

    # Создаём и решаем систему
    A = sp.coo_matrix((data, (row_indices, col_indices)),
                      shape=(N, N)).tocsr()

    p_prime = spsolve(A, b)

    return p_prime.reshape((NR, NZ))


def simplec_two_phase_step(grid, fields: Dict, props: Dict,
                           cfg: SimpleC2PSettings) -> Dict:
    """
    Один шаг SIMPLEC для двух фаз в пористой среде.
    """
    NR, NZ = fields['alpha_R'].shape

    # Инициализация давления если его нет
    if 'p' not in fields:
        fields['p'] = np.zeros((NR, NZ))

    converged = False
    mass_residual = 1.0

    for outer in range(cfg.max_outer):
        # === 1. Предиктор скоростей ===

        # Фаза R
        A_R, b_R = build_momentum_matrix(grid, fields, props, 'R', cfg)

        # Добавляем градиент давления
        for i in range(NR):
            for j in range(NZ):
                idx = i * NZ + j
                gamma = fields['gamma'][i, j]
                alpha_R = fields['alpha_R'][i, j]

                if alpha_R > cfg.alpha_cut and gamma > cfg.gamma_cut:
                    # Градиент давления по r
                    if i > 0 and i < NR - 1:
                        dp_dr = (fields['p'][i + 1, j] - fields['p'][i - 1, j]) / (2 * grid.dr)
                        b_R[idx] -= gamma * alpha_R * dp_dr

                    # Градиент давления по z
                    if j > 0 and j < NZ - 1:
                        dp_dz = (fields['p'][i, j + 1] - fields['p'][i, j - 1]) / (2 * grid.dz)
                        b_R[NR * NZ + idx] -= gamma * alpha_R * dp_dz

        # Решаем для v_R*
        v_R_star_flat = spsolve(A_R, b_R)
        v_R_r_star = v_R_star_flat[:NR * NZ].reshape((NR, NZ))
        v_R_z_star = v_R_star_flat[NR * NZ:].reshape((NR, NZ))

        # Фаза D
        A_D, b_D = build_momentum_matrix(grid, fields, props, 'D', cfg)

        # Добавляем градиент давления
        for i in range(NR):
            for j in range(NZ):
                idx = i * NZ + j
                gamma = fields['gamma'][i, j]
                alpha_D = fields['alpha_D'][i, j]

                if alpha_D > cfg.alpha_cut and gamma > cfg.gamma_cut:
                    # Градиент давления по r
                    if i > 0 and i < NR - 1:
                        dp_dr = (fields['p'][i + 1, j] - fields['p'][i - 1, j]) / (2 * grid.dr)
                        b_D[idx] -= gamma * alpha_D * dp_dr

                    # Градиент давления по z
                    if j > 0 and j < NZ - 1:
                        dp_dz = (fields['p'][i, j + 1] - fields['p'][i, j - 1]) / (2 * grid.dz)
                        b_D[NR * NZ + idx] -= gamma * alpha_D * dp_dz

        # Решаем для v_D*
        v_D_star_flat = spsolve(A_D, b_D)
        v_D_r_star = v_D_star_flat[:NR * NZ].reshape((NR, NZ))
        v_D_z_star = v_D_star_flat[NR * NZ:].reshape((NR, NZ))

        v_star = {
            'R_r': v_R_r_star, 'R_z': v_R_z_star,
            'D_r': v_D_r_star, 'D_z': v_D_z_star
        }

        # Получаем диагональные коэффициенты
        a_p = {
            'R': np.ones((NR, NZ)),  # Упрощенно - нужно извлечь из матрицы
            'D': np.ones((NR, NZ))
        }

        # === 2. Поправка давления ===
        p_prime = solve_pressure_correction(grid, fields, props, v_star, a_p, cfg)

        # === 3. Коррекция полей ===

        # Коррекция давления
        fields['p'] += cfg.relax_p * p_prime

        # Коррекция скоростей
        dr = grid.dr if hasattr(grid, 'dr') else 0.0301 / NR
        dz = grid.dz if hasattr(grid, 'dz') else 0.5692 / NZ

        for i in range(NR):
            for j in range(NZ):
                gamma = fields['gamma'][i, j]

                # Градиенты p'
                dp_dr = 0.0
                dp_dz = 0.0

                if i > 0 and i < NR - 1:
                    dp_dr = (p_prime[i + 1, j] - p_prime[i - 1, j]) / (2 * dr)
                if j > 0 and j < NZ - 1:
                    dp_dz = (p_prime[i, j + 1] - p_prime[i, j - 1]) / (2 * dz)

                # Коррекция фазы R
                if fields['alpha_R'][i, j] > cfg.alpha_cut:
                    d_R = gamma * fields['alpha_R'][i, j] / (a_p['R'][i, j] + 1e-30)
                    fields['v_R_r'][i, j] = v_R_r_star[i, j] - d_R * dp_dr
                    fields['v_R_z'][i, j] = v_R_z_star[i, j] - d_R * dp_dz

                # Коррекция фазы D
                if fields['alpha_D'][i, j] > cfg.alpha_cut:
                    d_D = gamma * fields['alpha_D'][i, j] / (a_p['D'][i, j] + 1e-30)
                    fields['v_D_r'][i, j] = v_D_r_star[i, j] - d_D * dp_dr
                    fields['v_D_z'][i, j] = v_D_z_star[i, j] - d_D * dp_dz

        # === 4. Проверка сходимости ===

        # Вычисляем невязку по массе
        mass_residual = np.max(np.abs(p_prime))

        if mass_residual < cfg.tol_m:
            converged = True
            break

        if outer % 10 == 0:
            print(f"  SIMPLEC итерация {outer}: невязка массы = {mass_residual:.3e}")

    # Вычисляем перепад давления
    dp_total = fields['p'][0, 0] - fields['p'][-1, -1]

    return {
        'converged': converged,
        'iterations': outer + 1,
        'mass_residual': mass_residual,
        'dp_total': dp_total,
        'p': fields['p'],
        'v_R_r': fields['v_R_r'],
        'v_R_z': fields['v_R_z'],
        'v_D_r': fields['v_D_r'],
        'v_D_z': fields['v_D_z']
    }


def apply_momentum_bc(fields: Dict, grid, inlet_vr_z: float = 4e-5):
    """
    Граничные условия для импульса.

    - Вход (низ, j=0): заданная скорость VR
    - Выход (верх, j=NZ-1): нулевой градиент
    - Стенка (i=NR-1): no-slip
    - Ось (i=0): симметрия
    """
    NR, NZ = fields['alpha_R'].shape

    # Вход (j=0)
    fields['v_R_z'][:, 0] = inlet_vr_z  # Скорость VR на входе
    fields['v_D_z'][:, 0] = 0.0  # Нет входа дистиллятов
    fields['v_R_r'][:, 0] = 0.0
    fields['v_D_r'][:, 0] = 0.0

    # Выход (j=NZ-1) - нулевой градиент
    fields['v_R_z'][:, -1] = fields['v_R_z'][:, -2]
    fields['v_D_z'][:, -1] = fields['v_D_z'][:, -2]
    fields['v_R_r'][:, -1] = fields['v_R_r'][:, -2]
    fields['v_D_r'][:, -1] = fields['v_D_r'][:, -2]

    # Стенка (i=NR-1) - no-slip
    fields['v_R_r'][-1, :] = 0.0
    fields['v_R_z'][-1, :] = 0.0
    fields['v_D_r'][-1, :] = 0.0
    fields['v_D_z'][-1, :] = 0.0

    # Ось (i=0) - симметрия
    fields['v_R_r'][0, :] = 0.0  # Нет радиальной скорости на оси
    fields['v_D_r'][0, :] = 0.0

    return fields