"""Differential drive kinematic model.

Differential drive is one of the most common locomotion mechanisms for
mobile robots. It uses two independently driven wheels to achieve motion
and rotation.

References:
    - Siegwart, R., & Nourbakhsh, I. R. (2004). Introduction to autonomous
      mobile robots. MIT press.
"""
import numpy as np
from typing import Tuple

from nhmrs._nhmrs_utils.kinematics.base import KinematicsModel


class DifferentialDriveKinematics(KinematicsModel):
    """Differential drive kinematic model.
    
    Differential drive robots have two wheels that can be controlled independently.
    The robot's motion is determined by the difference in wheel velocities.
    
    State Representation:
        [x, y, θ]
        - x: Position along world x-axis (units)
        - y: Position along world y-axis (units)
        - θ: Orientation angle in radians, range [-π, π]
    
    Control Inputs:
        [v_left, v_right]
        - v_left: Left wheel velocity (units/s)
        - v_right: Right wheel velocity (units/s)
    
    Derived Velocities:
        v = (v_left + v_right) / 2     # Linear velocity
        ω = (v_right - v_left) / L     # Angular velocity
    
    Where L is the wheelbase (distance between wheels).
    
    Continuous-Time Dynamics:
        ẋ = v cos(θ)
        ẏ = v sin(θ)
        θ̇ = ω
    
    Special Cases:
        - v_left = v_right: Straight line motion
        - v_left = -v_right: Rotation in place
        - v_left = 0 or v_right = 0: Pivot turn
    
    Attributes:
        dt (float): Integration time step (seconds)
        wheelbase (float): Distance between left and right wheels (units)
        wheel_vel_max (float): Maximum wheel velocity magnitude (units/s)
    
    Example:
        >>> kinematics = DifferentialDriveKinematics(wheelbase=0.3)
        >>> state = np.array([0.0, 0.0, 0.0])
        >>> action = np.array([1.0, 1.5])  # Right wheel faster -> turn left
        >>> next_state = kinematics.step(state, action)
    """
    
    def __init__(self, dt: float = 0.1, wheelbase: float = 0.3, 
                 wheel_vel_max: float = 2.0):
        """Initialize differential drive kinematics model.
        
        Args:
            dt (float): Time step for integration (default: 0.1 seconds)
            wheelbase (float): Distance between wheels (default: 0.3 units)
            wheel_vel_max (float): Maximum wheel velocity (default: 2.0 units/s)
        """
        super().__init__(dt)
        self.wheelbase = wheelbase
        self.wheel_vel_max = wheel_vel_max
    
    @property
    def state_dim(self) -> int:
        """Return state dimension: 3 for [x, y, θ]."""
        return 3
    
    @property
    def action_dim(self) -> int:
        """Return action dimension: 2 for [v_left, v_right]."""
        return 2
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return action bounds for wheel velocities.
        
        Returns:
            tuple: ([-v_max, -v_max], [v_max, v_max]) for both wheels
        """
        lower = np.array([-self.wheel_vel_max, -self.wheel_vel_max], dtype=np.float32)
        upper = np.array([self.wheel_vel_max, self.wheel_vel_max], dtype=np.float32)
        return lower, upper
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Integrate differential drive dynamics forward by one time step.
        
        Args:
            state (np.ndarray): Current state [x, y, θ]
            action (np.ndarray): Control input [v_left, v_right]
        
        Returns:
            np.ndarray: Next state [x', y', θ'] after dt seconds
        """
        x, y, theta = state
        v_left, v_right = action
        
        # Clip wheel velocities to feasible range
        v_left = np.clip(v_left, -self.wheel_vel_max, self.wheel_vel_max)
        v_right = np.clip(v_right, -self.wheel_vel_max, self.wheel_vel_max)
        
        # Compute linear and angular velocities from wheel velocities
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wheelbase
        
        # Integrate dynamics (equivalent to unicycle model)
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        
        # Normalize angle to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new], dtype=np.float32)
    
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Extract (x, y) position from differential drive state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            np.ndarray: Position [x, y]
        """
        return state[:2]
    
    def get_orientation(self, state: np.ndarray) -> float:
        """Extract orientation angle from differential drive state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            float: Orientation θ in radians
        """
        return state[2]
