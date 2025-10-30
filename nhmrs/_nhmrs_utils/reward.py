"""Reward computation utilities for NHMRS environments.

Provides abstract base class for reward functions.
Environment-specific reward implementations should inherit from BaseReward.
"""

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Abstract base class for NHMRS reward functions.
    
    All environment-specific reward classes should inherit from this class
    and implement the compute_reward method.
    """
    
    def reset(self):
        """Reset any state tracking at episode start."""
        pass

    def update_timestep(self, t: int):
        """Update current timestep for time-dependent rewards."""
        pass

    @abstractmethod
    def compute_reward(self, agent, world) -> float:
        """Compute reward for an agent in the world.
        
        Args:
            agent: Agent object to compute reward for
            world: World object containing all agents and landmarks
            
        Returns:
            float: Computed reward value
        """
        pass
