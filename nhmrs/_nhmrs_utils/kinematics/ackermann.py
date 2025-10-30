"""Ackermann steering kinematic model.

Ackermann steering is the geometry used in most cars and car-like robots.
It provides a realistic model for vehicles with front-wheel steering.

References:
    - Siegwart, R., & Nourbakhsh, I. R. (2004). Introduction to autonomous
      mobile robots. MIT press.
    - Lavalle, S. M. (2006). Planning algorithms. Cambridge university press.
"""
import numpy as np
from typing import Tuple

from nhmrs._nhmrs_utils.kinematics.base import KinematicsModel


class AckermannKinematics(KinematicsModel):
    """Ackermann steering kinematic model (car-like robot).
    
    The Ackermann steering model represents vehicles with front-wheel steering,
    such as cars, trucks, and many mobile robot platforms. The rear wheels are
    fixed and driven, while the front wheels can be steered.
    
    State Representation:
        [x, y, θ]
        - x: Position of rear axle center along world x-axis (units)
        - y: Position of rear axle center along world y-axis (units)
        - θ: Heading angle in radians, range [-π, π]
    
    Control Inputs:
        [v, δ]
        - v: Velocity at rear axle (units/s) - forward/backward speed
        - δ: Steering angle (radians) - angle of front wheels relative to vehicle
    
    Continuous-Time Dynamics:
        ẋ = v cos(θ)
        ẏ = v sin(θ)
        θ̇ = (v / L) tan(δ)
    
    Where L is the wheelbase (distance from rear to front axle).
    
    Turning Radius:
        R = L / tan(δ)
    
    Constraints:
        - The steering angle δ is typically limited to avoid sharp turns
        - At δ = 0, the vehicle moves straight
        - Larger |δ| values create tighter turns (smaller radius)
    
    Attributes:
        dt (float): Integration time step (seconds)
        wheelbase (float): Distance between rear and front axles (units)
        v_max (float): Maximum velocity magnitude (units/s)
        delta_max (float): Maximum steering angle magnitude (radians)
    
    Example:
        >>> kinematics = AckermannKinematics(wheelbase=0.5, delta_max=np.pi/4)
        >>> state = np.array([0.0, 0.0, 0.0])
        >>> action = np.array([1.0, 0.3])  # Forward with steering
        >>> next_state = kinematics.step(state, action)
    """
    
    def __init__(self, dt: float = 0.1, wheelbase: float = 0.5,
                 v_max: float = 2.0, delta_max: float = np.pi/4):
        """Initialize Ackermann steering kinematics model.
        
        Args:
            dt (float): Time step for integration (default: 0.1 seconds)
            wheelbase (float): Distance between axles (default: 0.5 units)
            v_max (float): Maximum velocity (default: 2.0 units/s)
            delta_max (float): Maximum steering angle (default: π/4 rad = 45°)
        """
        super().__init__(dt)
        self.wheelbase = wheelbase
        self.v_max = v_max
        self.delta_max = delta_max
    
    @property
    def state_dim(self) -> int:
        """Return state dimension: 3 for [x, y, θ]."""
        return 3
    
    @property
    def action_dim(self) -> int:
        """Return action dimension: 2 for [v, δ]."""
        return 2
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return action bounds for Ackermann controls.
        
        Returns:
            tuple: ([-v_max, -δ_max], [v_max, δ_max])
        """
        lower = np.array([-self.v_max, -self.delta_max], dtype=np.float32)
        upper = np.array([self.v_max, self.delta_max], dtype=np.float32)
        return lower, upper
    
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Integrate Ackermann steering dynamics forward by one time step.
        
        Args:
            state (np.ndarray): Current state [x, y, θ]
            action (np.ndarray): Control input [v, δ]
        
        Returns:
            np.ndarray: Next state [x', y', θ'] after dt seconds
        """
        x, y, theta = state
        v, delta = action
        
        # Clip actions to feasible range
        v = np.clip(v, -self.v_max, self.v_max)
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        
        # Integrate Ackermann dynamics
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + (v / self.wheelbase) * np.tan(delta) * self.dt
        
        # Normalize angle to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new], dtype=np.float32)
    
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Extract (x, y) position from Ackermann state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            np.ndarray: Position [x, y]
        """
        return state[:2]
    
    def get_orientation(self, state: np.ndarray) -> float:
        """Extract orientation angle from Ackermann state.
        
        Args:
            state (np.ndarray): State [x, y, θ]
        
        Returns:
            float: Orientation θ in radians
        """
        return state[2]
