"""Abstract base class for robot kinematics models.

This module defines the interface that all kinematic models must implement
to be compatible with NHMRS environments.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class KinematicsModel(ABC):
    """Abstract base class for robot kinematics models.
    
    A kinematics model defines:
    1. The state representation (typically [x, y, θ, ...])
    2. The control input space
    3. How controls map to state derivatives
    
    This abstraction allows environments to work with different robot
    types by simply swapping the kinematics model without changing
    the task allocation logic.
    
    Attributes:
        dt (float): Time step for discrete-time integration (seconds)
    
    Properties:
        state_dim (int): Dimension of the state vector
        action_dim (int): Dimension of the action/control vector
    
    Example:
        >>> class MyKinematics(KinematicsModel):
        ...     @property
        ...     def state_dim(self) -> int:
        ...         return 3
        ...     
        ...     def step(self, state, action):
        ...         # Implement dynamics
        ...         return next_state
    """
    
    def __init__(self, dt: float = 0.1):
        """Initialize kinematics model.
        
        Args:
            dt (float): Time step for integration (default: 0.1 seconds)
        """
        self.dt = dt
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the dimension of the state vector.
        
        Returns:
            int: Number of state variables (e.g., 3 for [x, y, θ])
        """
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the dimension of the action vector.
        
        Returns:
            int: Number of control inputs (e.g., 2 for [v, ω])
        """
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper bounds for actions.
        
        These bounds define the feasible control input space and are
        used to construct Gymnasium Box spaces in environments.
        
        Returns:
            tuple: (lower_bounds, upper_bounds) as numpy arrays of shape (action_dim,)
        
        Example:
            >>> lower, upper = kinematics.get_action_bounds()
            >>> print(lower)  # array([-2.0, -3.14])
            >>> print(upper)  # array([2.0, 3.14])
        """
        pass
    
    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute next state given current state and action.
        
        This method integrates the robot's dynamics forward by one time step (dt).
        It should handle action clipping and angle normalization internally.
        
        Args:
            state (np.ndarray): Current state vector of shape (state_dim,)
            action (np.ndarray): Control input vector of shape (action_dim,)
        
        Returns:
            np.ndarray: Next state vector of shape (state_dim,) after applying dynamics
        
        Example:
            >>> state = np.array([0.0, 0.0, 0.0])
            >>> action = np.array([1.0, 0.5])
            >>> next_state = kinematics.step(state, action)
        """
        pass
    
    @abstractmethod
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Extract (x, y) position from state.
        
        This method is used by environments to compute distances, rewards,
        and rendering positions.
        
        Args:
            state (np.ndarray): State vector of shape (state_dim,)
        
        Returns:
            np.ndarray: Position [x, y] of shape (2,)
        
        Example:
            >>> state = np.array([1.5, 2.3, 0.7])
            >>> pos = kinematics.get_position(state)
            >>> print(pos)  # array([1.5, 2.3])
        """
        pass
    
    @abstractmethod
    def get_orientation(self, state: np.ndarray) -> float:
        """Extract orientation angle from state.
        
        This method is used by environments for rendering and computing
        heading-based observations.
        
        Args:
            state (np.ndarray): State vector of shape (state_dim,)
        
        Returns:
            float: Orientation angle in radians, typically in range [-π, π]
        
        Example:
            >>> state = np.array([1.5, 2.3, 0.7])
            >>> theta = kinematics.get_orientation(state)
            >>> print(theta)  # 0.7
        """
        pass
