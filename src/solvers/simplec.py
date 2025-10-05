# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class SimpleC1DSettings:
    """Мини-настройки 1D Stokes–Brinkman."""
    p_outlet: float = 0.0  # давление на выходе, Па


class SimpleC1DBrinkman:
    """
    1D Stokes–Brinkman по оси (Eq.5):
        dp/dz = μ/K * u + 0.5 * C2 * ρ * |u| * u
    u берётся из ṁ на каждой осевой грани; p(z) интегрируется сверху вниз.

    Shapes:
      T: (nr, nz)
      p_face: (nz+1)
      uz_face: (nr, nz+1)
      info:
        z_faces: (nz+1), z_centers: (nz,), dz:(nz,)
        dpdz_faces: (nz+1), dpdz_centers:(nz,)
        dpdz_vis_faces, dpdz_in_faces: (nz+1)
        mu_face, rho_face, A_face: (nz+1)
        max_rel_mass_err: float
        dpdz: alias of dpdz_faces
    """
    def __init__(self, grid, fluid, K: float, gamma: np.ndarray,
                 C2: Optional[float] = None,
                 settings: Optional[SimpleC1DSettings] = None) -> None:
        self.g = grid
        self.fluid = fluid
        self.K = float(K)
        self.C2 = 0.0 if C2 is None else float(C2)
        self.settings = settings or SimpleC1DSettings()
        self.gamma = np.asarray(gamma, dtype=float)

        self.nr = int(getattr(self.g, "nr", self.gamma.shape[0]))
        self.nz = int(getattr(self.g, "nz", self.gamma.shape[1]))

        geom = getattr(self.g, "geom", self.g)
        H = float(getattr(geom, "height", 1.0))
        R = getattr(geom, "radius", None) or getattr(self.g, "radius", None)
        if R is None:
            raise AttributeError("Grid must provide 'radius' (м).")
        self.R = float(R)

        # ось z
        z_faces = getattr(self.g, "z_faces", None)
        if z_faces is None:
            z_faces = np.linspace(0.0, H, self.nz + 1)
        self.z_faces = np.asarray(z_faces, dtype=float).reshape(-1)
        self.z_centers = getattr(self.g, "z_centers", None)
        if self.z_centers is None:
            self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])
        self.z_centers = np.asarray(self.z_centers, dtype=float).reshape(-1)
        self.dz = self.z_faces[1:] - self.z_faces[:-1]

        # свободная площадь на z-гранях (учёт γ)
        self.A_face = self._compute_A_free_faces(self.gamma, self.R)

    @staticmethod
    def _compute_A_free_faces(gamma: np.ndarray, R: float) -> np.ndarray:
        nr, nz = gamma.shape
        A = np.empty(nz + 1, dtype=float)
        for j in range(nz + 1):
            jj = min(max(j - 1, 0), nz - 1)
            g_col = float(np.clip(np.mean(gamma[:, jj]), 1e-6, 1.0))
            A[j] = g_col * np.pi * R**2
        return A

    @staticmethod
    def _face_average_cell_column(col: np.ndarray) -> np.ndarray:
        nz = col.size
        out = np.empty(nz + 1, dtype=float)
        out[0] = col[0]
        out[1:] = col
        return out

    def solve(self, T: np.ndarray, m_dot: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        T = np.asarray(T, dtype=float)
        assert T.shape == (self.nr, self.nz), f"Ожидается T формы (nr, nz) = ({self.nr},{self.nz}), получено {T.shape}"

        # свойства на гранях
        T_mean = T.mean(axis=0)                       # (nz,)
        T_face = self._face_average_cell_column(T_mean)  # (nz+1,)
        mu_face  = self.fluid.viscosity(T_face)       # (nz+1,)
        rho_face = self.fluid.density(T_face)         # (nz+1,)

        # скорость из расхода на каждой грани
        denom = rho_face * self.A_face
        uz_line = m_dot / (denom + 1e-30)             # (nz+1,)
        uz_face = np.tile(uz_line, (self.nr, 1))      # (nr, nz+1)

        # Brinkman + Ergun
        dpdz_vis_faces = mu_face * uz_line / max(self.K, 1e-30)
        dpdz_in_faces  = 0.5 * self.C2 * rho_face * np.abs(uz_line) * uz_line
        dpdz_faces     = dpdz_vis_faces + dpdz_in_faces             # (nz+1,)
        dpdz_centers   = 0.5 * (dpdz_faces[:-1] + dpdz_faces[1:])   # (nz,)

        # p(z) сверху вниз
        p_face = np.zeros(self.nz + 1, dtype=float)
        p_face[-1] = float(self.settings.p_outlet)
        for j in range(self.nz - 1, -1, -1):
            p_face[j] = p_face[j + 1] + dpdz_faces[j] * self.dz[j]

        # диагностика расхода
        mdot_face = rho_face * uz_line * self.A_face
        max_rel_mass_err = float(np.max(np.abs(mdot_face - m_dot)) / max(abs(m_dot), 1e-30))

        info: Dict[str, Any] = {
            "mu_face": mu_face,
            "rho_face": rho_face,
            "A_face": self.A_face,
            "z_faces": self.z_faces,
            "z_centers": self.z_centers,
            "dz": self.dz,
            "dpdz": dpdz_faces,               # alias
            "dpdz_faces": dpdz_faces,
            "dpdz_centers": dpdz_centers,
            "dpdz_vis_faces": dpdz_vis_faces,
            "dpdz_in_faces": dpdz_in_faces,
            "max_rel_mass_err": max_rel_mass_err,
        }
        return p_face, uz_face, info
