"""Тесты для кинетики реакций."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from physics.kinetics import ReactionKinetics


class TestKinetics:
    def test_initialization(self):
        for vr_type in [1, 2, 3]:
            kin = ReactionKinetics(vr_type)
            assert kin.vr_type == vr_type

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            ReactionKinetics(4)

    def test_reaction_order(self):
        kin = ReactionKinetics(1)
        assert kin.reaction_order(400) == 1.0
        assert kin.reaction_order(800) == 1.5
        assert kin.reaction_order(900) == 2.0

    def test_rate_constant(self):
        kin = ReactionKinetics(1)
        T = np.array([600, 700, 800])
        k = [kin.rate_constant(t, 1.0) for t in T]
        assert k[0] < k[1] < k[2]

    def test_reaction_rate(self):
        kin = ReactionKinetics(1)
        rate = kin.reaction_rate(700, 800)
        assert rate > 0

        T_array = np.array([600, 700, 800])
        rate_array = kin.reaction_rate(T_array, 800)
        assert rate_array.shape == T_array.shape
        assert np.all(rate_array >= 0)

    def test_mass_conservation(self):
        kin = ReactionKinetics(1)
        G_R, G_C, G_D = kin.source_terms(700, 800)
        assert G_R < 0
        assert np.isclose(G_R + G_C + G_D, 0)