"""Kinematics models for non-holonomic mobile robots.

This subpackage provides various kinematic models that can be used across
different NHMRS environments. All models inherit from the abstract
KinematicsModel base class.

Available models:
    - UnicycleKinematics: Standard unicycle model
    - DifferentialDriveKinematics: Two-wheel differential drive
    - AckermannKinematics: Car-like Ackermann steering
"""

from nhmrs._nhmrs_utils.kinematics.base import KinematicsModel
from nhmrs._nhmrs_utils.kinematics.unicycle import UnicycleKinematics
from nhmrs._nhmrs_utils.kinematics.differential_drive import DifferentialDriveKinematics
from nhmrs._nhmrs_utils.kinematics.ackermann import AckermannKinematics

__all__ = [
    "KinematicsModel",
    "UnicycleKinematics",
    "DifferentialDriveKinematics",
    "AckermannKinematics",
]

from nhmrs._nhmrs_utils.kinematics.base import KinematicsModel
from nhmrs._nhmrs_utils.kinematics.unicycle import UnicycleKinematics
from nhmrs._nhmrs_utils.kinematics.differential_drive import DifferentialDriveKinematics
from nhmrs._nhmrs_utils.kinematics.ackermann import AckermannKinematics

__all__ = [
    "KinematicsModel",
    "UnicycleKinematics",
    "DifferentialDriveKinematics",
    "AckermannKinematics",
]
