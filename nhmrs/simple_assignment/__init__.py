"""Simple Assignment: Task allocation environment for NHMRS.

A PettingZoo environment where non-holonomic agents must be assigned to
tasks to minimize total cost (e.g., travel distance).
"""

from nhmrs.simple_assignment.simple_assignment import env, raw_env

__all__ = ["env", "raw_env"]
