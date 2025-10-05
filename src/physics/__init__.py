# src/physics/__init__.py
"""Модуль физических свойств и кинетики."""

from .properties import VacuumResidue, Distillables, Coke
from .kinetics import ReactionKinetics

__all__ = ["VacuumResidue", "Distillables", "Coke", "ReactionKinetics"]
