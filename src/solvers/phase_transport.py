# -*- coding: utf-8 -*-
"""
Транспорт объёмных долей фаз (массовый баланс).
Реализация уравнений (2)-(4) из статьи Díaz et al.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional


@dataclass
class TransportSettings:
    """Настройки транспортного солвера."""
    dt: float = 0.01  # Шаг по времени, с (как в статье)
    xi_coke: float = 0.28  # Доля кокса из VR (по данным статьи)
    xi_dist: float = 0.72  # Доля дистиллятов = 1 - xi_coke
    upwind_eps: float = 1e-12  # Малое число для upwind
    max_iterations: int = 100  # Макс. итераций для неявной схемы
    tolerance: float = 1e-6  # Критерий сходимости

    def __post_init__(self):
        self.xi_dist = 1.0 - self.xi_coke


def reaction_rate_omega_R(T: np.ndarray, alpha_R: np.ndarray,
                          rho_R: float, gamma: np.ndarray,
                          kinetics) -> np.ndarray:
    """
    Скорость расхода VR из кинетики: ω_R [кг/(м³·с)]

    Концентрация VR: C_R = α_R * rho_R [кг/м³]
    С учетом пористости: эффективная концентрация = γ * α_R * rho_R
    """
    # Защита от деления на ноль
    eps = 1e-14

    # Эффективная концентрация VR в порах
    C_vr = np.maximum(gamma * alpha_R * rho_R, eps)

    # Используем метод reaction_rate из класса ReactionKinetics
    omega_R = kinetics.reaction_rate(T, C_vr)

    # Обнуляем реакцию там, где нет VR
    omega_R = np.where(alpha_R > 1e-6, omega_R, 0.0)

    return omega_R


def split_sources(omega_R: np.ndarray, xi_c: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделение источников реакции между фазами.

    omega_R: скорость расхода VR [кг/(м³·с)]
    xi_c: массовая доля кокса

    Возвращает: (S_R, S_D, S_C) - источники для каждой фазы
    """
    S_R = -omega_R  # VR расходуется
    S_C = xi_c * omega_R  # Кокс образуется
    S_D = (1.0 - xi_c) * omega_R  # Дистилляты образуются

    return S_R, S_D, S_C


def solve_tridiagonal(a, b, c, d):
    """
    Решение трехдиагональной системы методом прогонки.
    a - нижняя диагональ, b - главная диагональ,
    c - верхняя диагональ, d - правая часть
    """
    n = len(d)
    x = np.zeros(n)

    # Прямой ход
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        if i < n - 1:
            c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i - 1])
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / (b[i] - a[i] * c_prime[i - 1] if i > 0 else b[i])

    # Обратный ход
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def implicit_upwind_convection(M_old: np.ndarray,
                               v_r: np.ndarray, v_z: np.ndarray,
                               source: np.ndarray,
                               grid, dt: float,
                               bc_type: str = 'outflow') -> np.ndarray:
    """
    Неявная upwind схема для уравнения переноса:
    dM/dt + div(M*v) = S

    Полунеявная версия для лучшей устойчивости
    """
    NR, NZ = M_old.shape

    # Получаем размеры ячеек безопасно
    if hasattr(grid, 'dr'):
        dr = grid.dr
    elif hasattr(grid, 'geom') and hasattr(grid.geom, 'dr'):
        dr = grid.geom.dr
    else:
        dr = 0.0301 / NR

    if hasattr(grid, 'dz'):
        dz = grid.dz
    elif hasattr(grid, 'geom') and hasattr(grid.geom, 'dz'):
        dz = grid.geom.dz
    else:
        dz = 0.5692 / NZ

    # Сначала добавляем источники
    M_intermediate = M_old + dt * source

    # Полунеявная схема для конвекции
    M_new = M_intermediate.copy()

    # Решаем построчно (по z) - более стабильно
    for i in range(NR):
        # Трехдиагональная система для каждой радиальной линии
        a = np.zeros(NZ)  # нижняя диагональ
        b = np.ones(NZ)  # главная диагональ
        c = np.zeros(NZ)  # верхняя диагональ
        d = M_intermediate[i, :].copy()  # правая часть

        for j in range(NZ):
            # Коэффициент Куранта
            cfl_z = dt * abs(v_z[i, j]) / dz

            if v_z[i, j] > 0 and j > 0:
                # Поток снизу вверх
                a[j] = -cfl_z
                b[j] = 1.0 + cfl_z
            elif v_z[i, j] < 0 and j < NZ - 1:
                # Поток сверху вниз
                b[j] = 1.0 + cfl_z
                c[j] = -cfl_z

        # Граничные условия
        # Вход (j=0): заданное значение из M_intermediate (правильный Dirichlet)
        b[0] = 1.0
        c[0] = 0.0
        d[0] = M_intermediate[i, 0]  # Фиксируем входное значение

        # Выход (j=NZ-1): нулевой градиент
        a[-1] = -1.0
        b[-1] = 1.0
        d[-1] = 0.0  # Относительно предыдущей ячейки

        # Решаем трехдиагональную систему (метод прогонки)
        M_new[i, :] = solve_tridiagonal(a, b, c, d)

    # Неотрицательность
    M_new = np.maximum(M_new, 0.0)

    return M_new


def apply_boundary_conditions(fields: Dict, grid, inlet_mass_rate: float = 5e-3 / 60):
    """
    Граничные условия:
    - Вход (низ): массовый расход VR = 5 г/мин
    - Выход (верх): нулевой градиент
    - Стенки: непроницаемые
    """
    NR, NZ = fields['alpha_R'].shape

    # Получаем радиус из grid
    if hasattr(grid, 'radius'):
        radius = grid.radius
    elif hasattr(grid, 'geom') and hasattr(grid.geom, 'radius'):
        radius = grid.geom.radius
    else:
        radius = 0.0301  # D/2 для D=0.0602

    # Проверяем наличие rho_R в fields
    if 'rho_R' in fields:
        rho_R = fields['rho_R']
    else:
        # Используем типичное значение для VR
        rho_R = 800.0

    # Вход (j=0): заданный массовый расход VR
    # Распределяем равномерно по радиусу
    inlet_area = np.pi * radius ** 2
    inlet_velocity = inlet_mass_rate / (rho_R * inlet_area)

    fields['v_R_z'][:, 0] = inlet_velocity
    fields['v_D_z'][:, 0] = 0.0  # Нет входа дистиллятов

    # Выход (j=NZ-1): нулевой градиент
    fields['alpha_R'][:, -1] = fields['alpha_R'][:, -2]
    fields['alpha_D'][:, -1] = fields['alpha_D'][:, -2]

    # Стенки (i=0, i=NR-1): непроницаемые
    fields['v_R_r'][0, :] = 0.0
    fields['v_R_r'][-1, :] = 0.0
    fields['v_D_r'][0, :] = 0.0
    fields['v_D_r'][-1, :] = 0.0

    return fields


def advance_one_timestep(fields: Dict, props: Dict, kinetics,
                         grid, cfg: TransportSettings) -> Dict:
    """
    Продвижение на один шаг по времени.

    fields: словарь с полями (alpha_R, alpha_D, alpha_C, gamma, T_R, T_D, v_R, v_D)
    props: физические свойства (rho_R, rho_D, rho_C)
    kinetics: объект кинетики
    grid: сетка
    cfg: настройки
    """
    dt = cfg.dt

    # 1. Вычисляем массы фаз M = γ * α * ρ
    M_R_old = fields['gamma'] * fields['alpha_R'] * props['rho_R']
    M_D_old = fields['gamma'] * fields['alpha_D'] * props['rho_D']

    # 2. Вычисляем источники из кинетики
    omega_R = reaction_rate_omega_R(
        fields['T_R'], fields['alpha_R'],
        props['rho_R'], fields['gamma'], kinetics
    )
    S_R, S_D, S_C = split_sources(omega_R, cfg.xi_coke)

    # 3. Решаем уравнения переноса для R и D
    M_R_new = implicit_upwind_convection(
        M_R_old, fields['v_R_r'], fields['v_R_z'],
        S_R, grid, dt
    )

    M_D_new = implicit_upwind_convection(
        M_D_old, fields['v_D_r'], fields['v_D_z'],
        S_D, grid, dt
    )

    # 4. Обновляем кокс (только накопление, без переноса)
    alpha_C_rho_C_old = fields['alpha_C'] * props['rho_C']
    alpha_C_rho_C_new = alpha_C_rho_C_old + dt * S_C
    alpha_C_new = np.clip(alpha_C_rho_C_new / props['rho_C'], 0.0, 1.0)

    # 5. Обновляем пористость
    gamma_new = np.maximum(1.0 - alpha_C_new, 1e-6)

    # 6. Восстанавливаем объёмные доли из масс
    eps = 1e-14
    alpha_R_new = np.where(
        gamma_new > eps,
        M_R_new / (gamma_new * props['rho_R']),
        0.0
    )
    alpha_D_new = np.where(
        gamma_new > eps,
        M_D_new / (gamma_new * props['rho_D']),
        0.0
    )

    # 7. Вычисляем насыщенности и нормализуем
    S_R_new = np.zeros_like(alpha_R_new)
    S_D_new = np.zeros_like(alpha_D_new)

    mask = gamma_new > eps
    S_R_new[mask] = alpha_R_new[mask] / gamma_new[mask]
    S_D_new[mask] = alpha_D_new[mask] / gamma_new[mask]

    # ВСЕГДА нормализуем насыщенности в порах для соблюдения S_R + S_D = 1
    S_sum = S_R_new + S_D_new
    valid = mask & (S_sum > eps)

    # Нормализация при любом отклонении от 1
    S_R_new[valid] /= S_sum[valid]
    S_D_new[valid] /= S_sum[valid]

    # Если поры пустые или S_sum близка к нулю - устанавливаем значения по умолчанию
    fallback = mask & ~valid
    S_R_new[fallback] = 1.0
    S_D_new[fallback] = 0.0

    # Пересчитываем объёмные доли после нормализации
    alpha_R_new = gamma_new * S_R_new
    alpha_D_new = gamma_new * S_D_new

    # 8. Проверка баланса массы
    sum_alpha = alpha_R_new + alpha_D_new + alpha_C_new
    balance_error = np.max(np.abs(sum_alpha - 1.0))

    if balance_error > 1e-6:
        print(f"⚠ Ошибка баланса массы: {balance_error:.3e}")

    # 9. Вычисляем число Куранта для диагностики
    v_max_r = np.max(np.abs(fields['v_R_r']))
    v_max_z = np.max(np.abs(fields['v_R_z']))

    # Получаем размеры ячеек
    if hasattr(grid, 'dr'):
        dr = grid.dr
    else:
        dr = 0.0301 / fields['alpha_R'].shape[0]

    if hasattr(grid, 'dz'):
        dz = grid.dz
    else:
        dz = 0.5692 / fields['alpha_R'].shape[1]

    courant = dt * (v_max_r / dr + v_max_z / dz)

    # Возвращаем обновленные поля
    return {
        'alpha_R': alpha_R_new,
        'alpha_D': alpha_D_new,
        'alpha_C': alpha_C_new,
        'S_R': S_R_new,
        'S_D': S_D_new,
        'gamma': gamma_new,
        'M_R': M_R_new,
        'M_D': M_D_new,
        'courant': courant,
        'balance_error': balance_error,
        'omega_R': omega_R  # Для диагностики
    }


def check_conservation(fields_old: Dict, fields_new: Dict,
                       props: Dict, grid, dt: float) -> Dict:
    """
    Проверка законов сохранения.
    """
    # Объем ячеек для осесимметричной геометрии
    # Используем готовый массив volumes если есть, или вычисляем
    if hasattr(grid, 'volumes'):
        dV = grid.volumes
    else:
        # Если volumes нет, вычисляем сами
        # Используем 2D массив координат grid.R если есть
        if hasattr(grid, 'R'):
            r_centers = grid.R  # Это уже 2D массив (NR, NZ)
            dr = grid.dr if hasattr(grid, 'dr') else grid.geom.dr
            dz = grid.dz if hasattr(grid, 'dz') else grid.geom.dz
            dV = 2 * np.pi * r_centers * dr * dz
        else:
            # Крайний случай - строим 2D массив из 1D
            NR, NZ = fields_old['alpha_R'].shape
            r_1d = grid.r_centers if hasattr(grid, 'r_centers') else np.linspace(0, 0.0301, NR)
            z_1d = grid.z_centers if hasattr(grid, 'z_centers') else np.linspace(0, 0.5692, NZ)
            R, Z = np.meshgrid(r_1d, z_1d, indexing='ij')
            dr = 0.0301 / NR
            dz = 0.5692 / NZ
            dV = 2 * np.pi * R * dr * dz

    # Общая масса до и после
    mass_R_old = np.sum(fields_old['gamma'] * fields_old['alpha_R'] * props['rho_R'] * dV)
    mass_D_old = np.sum(fields_old['gamma'] * fields_old['alpha_D'] * props['rho_D'] * dV)
    mass_C_old = np.sum(fields_old['alpha_C'] * props['rho_C'] * dV)
    total_mass_old = mass_R_old + mass_D_old + mass_C_old

    mass_R_new = np.sum(fields_new['gamma'] * fields_new['alpha_R'] * props['rho_R'] * dV)
    mass_D_new = np.sum(fields_new['gamma'] * fields_new['alpha_D'] * props['rho_D'] * dV)
    mass_C_new = np.sum(fields_new['alpha_C'] * props['rho_C'] * dV)
    total_mass_new = mass_R_new + mass_D_new + mass_C_new

    # Изменение массы
    dm_R = (mass_R_new - mass_R_old) / dt
    dm_D = (mass_D_new - mass_D_old) / dt
    dm_C = (mass_C_new - mass_C_old) / dt
    dm_total = (total_mass_new - total_mass_old) / dt

    return {
        'mass_R': mass_R_new,
        'mass_D': mass_D_new,
        'mass_C': mass_C_new,
        'mass_total': total_mass_new,
        'dm_R': dm_R,
        'dm_D': dm_D,
        'dm_C': dm_C,
        'dm_total': dm_total,
        'conservation_error': abs(dm_total) / total_mass_new if total_mass_new > 0 else 0
    }