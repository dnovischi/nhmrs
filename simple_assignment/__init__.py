"""NHMRS: A PettingZoo environment for non-holonomic mobile robots."""

__version__ = "0.1.0"

from simple_assignment.simple_assignment_v0 import env
from simple_assignment.env.scenario import Scenario

__all__ = ["env", "Scenario"]
