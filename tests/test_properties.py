"""Тесты для физических свойств."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from physics.properties import VacuumResidue, Distillables, Coke


class TestVacuumResidue:
    def test_initialization(self):
        for vr_type in [1, 2, 3]:
            vr = VacuumResidue(vr_type)
            assert vr.vr_type == vr_type
            assert vr.density_15C > 0

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            VacuumResidue(4)

    def test_density_temperature(self):
        vr = VacuumResidue(1)
        assert vr.density(400) < vr.density(300)

    def test_viscosity_temperature(self):
        vr = VacuumResidue(1)
        assert vr.viscosity(500) < vr.viscosity(400)


class TestDistillables:
    def test_ideal_gas(self):
        dist = Distillables()
        T, P = 273.15, 101325
        rho = dist.density(T, P)
        assert 0.001 < rho < 10
        assert np.isclose(P, rho * dist.R_gas * T, rtol=1e-6)

    def test_viscosity(self):
        dist = Distillables()
        T = np.linspace(300, 800, 10)
        assert np.all(np.diff(dist.viscosity(T)) > 0)


class TestCoke:
    def test_constant_density(self):
        coke = Coke()
        assert coke.density(300) == coke.density(700)

    def test_properties(self):
        coke = Coke()
        assert coke.bulk_density < coke.props.density_ref
        assert coke.particle_diameter == 0.001