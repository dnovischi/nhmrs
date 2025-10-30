"""NHMRS: Non-Holonomic Mobile Robot Systems.

A collection of PettingZoo environments and kinematic models for multi-agent
reinforcement learning with non-holonomic mobile robots.

Environments:
    - simple_assignment_v0: Basic task allocation problem

Kinematics Models:
    - UnicycleKinematics: Standard unicycle model
    - DifferentialDriveKinematics: Two-wheel differential drive
    - AckermannKinematics: Car-like steering

Example:
    >>> from nhmrs import simple_assignment_v0
    >>> from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics
    >>> 
    >>> env = simple_assignment_v0.env(render_mode="human")
    >>> obs, info = env.reset()
"""

__version__ = "0.1.0"
