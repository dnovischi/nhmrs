"""Unicycle kinematic model.

The unicycle model is the simplest and most commonly used non-holonomic
kinematic model. It represents a robot that can move forward/backward
and rotate, but cannot move sideways (non-holonomic constraint).

References:
    - Siegwart, R., & Nourbakhsh, I. R. (2004). Introduction to autonomous
      mobile robots. MIT press.
"""
import numpy as np
from typing import Tuple

from nhmrs._nhmrs_utils.kinematics.base import KinematicsModel


class UnicycleKinematics(KinematicsModel):
    """Standard unicycle kinematic model.
    
    The unicycle model is widely used as a simple approximation for
    differential drive robots and other non-holonomic mobile platforms.
    
    State Representation:
        [x, y, θ]
        - x: Position along world x-axis (units)
        - y: Position along world y-axis (units)
        - θ: Orientation angle in radians, range [-π, π]
    
    Control Inputs:
        [v, ω]
        - v: Linear velocity (units/s) - forward/backward speed
        - ω: Angular velocity (rad/s) - rotation rate
    
    Continuous-Time Dynamics:
        ẋ = v cos(θ)
        ẏ = v sin(θ)
        θ̇ = ω
    
    Discrete-Time Integration (Euler):
        x_{t+1} = x_t + v cos(θ_t) * dt
        y_{t+1} = y_t + v sin(θ_t) * dt
        θ_{t+1} = θ_t + ω * dt
    
    Non-Holonomic Constraint:
        The robot cannot move sideways: ẋ sin(θ) - ẏ cos(θ) = 0
    
    Attributes:
        dt (float): Integration time step (seconds)
        v_max (float): Maximum linear velocity magnitude (units/s)
        omega_max (float): Maximum angular velocity magnitude (rad/s)
    
    Example:
        >>> kinematics = UnicycleKinematics(dt=0.1, v_max=2.0, omega_max=np.pi)
        >>> state = np.array([0.0, 0.0, 0.0])  # Start at origin facing right
        >>> action = np.array([1.0, 0.5])  # Move forward while turning left
        >>> next_state = kinematics.step(state, action)
        >>> print(next_state)  # [~0.1, ~0.0, ~0.05]
    """
    
    def __init__(self, dt: float = 0.1, v_max: float = 2.0, 
                 omega_max: float = np.pi):
        """Initialize unicycle kinematics model.
        
        Args:
            dt (float): Time step for integration (default: 0.1 seconds)
            v_max (float): Maximum linear velocity magnitude (default: 2.0 units/s)
            omega_max (float): Maximum angular velocity magnitude (default: π rad/s)
        """
        super().__init__(dt)
        self.v_max = v_max
        self.omega_max = omega_max
    
    @property
    def state_dim(self) -> int:
        """Return state dimension: 3 for [x, y, θ]."""
        return 3
    
    @property
    def action_dim(self) -> int:
        """Return action dimension: 2 for [v, ω]."""
        return 2
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return action bounds for unicycle controls.
        
        Returns:
            tuple: ([-v_max, -ω_max], [v_max, ω_max])
        """
        lower = np.array([-self.v_max, -self.omega_max], dtype=np.float32)
        upper = np.array([self.v_max, self.omega_max], dtype=np.float32)
        return lower, upper
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Integrate unicycle dynamics forward by one time step.
        
        Uses first-order Euler integration. For more accurate results with
        larger time steps, consider using higher-order integration methods.
        
        Args:
            state (np.ndarray): Current state [x, y, θ]
            action (np.ndarray): Control input [v, ω]
        
        Returns:
            np.ndarray: Next state [x', y', θ'] after dt seconds
        """
        x, y, theta = state
        v, omega = action
        
        # Clip actions to feasible range
        v = np.clip(v, -self.v_max, self.v_max)
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        # Euler integration of unicycle dynamics
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        
        # Normalize angle to [-π, π] to prevent numerical drift
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new], dtype=np.float32)
    
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Extract (x, y) position from unicycle state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            np.ndarray: Position [x, y]
        """
        return state[:2]
    
    def get_orientation(self, state: np.ndarray) -> float:
        """Extract orientation angle from unicycle state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            float: Orientation θ in radians
        """
        return state[2]
