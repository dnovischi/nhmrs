"""Base scenario class for NHMRS environments.

Defines the interface that all scenario implementations must follow.
"""


class BaseScenario:
    """Base class defining scenario interface for NHMRS environments."""
    
    def make_world(self):
        """Create elements of the world.
        
        Returns:
            World: Initialized world with agents and landmarks.
        """
        raise NotImplementedError()
    
    def reset_world(self, world, np_random):
        """Create initial conditions of the world.
        
        Args:
            world: World object to reset.
            np_random: Random number generator for reproducibility.
        """
        raise NotImplementedError()
    
    def reward(self, agent, world):
        """Calculate reward for a specific agent.
        
        Args:
            agent: Agent to calculate reward for.
            world: Current world state.
            
        Returns:
            float: Reward value for the agent.
        """
        raise NotImplementedError()
    
    def observation(self, agent, world):
        """Calculate observation for a specific agent.
        
        Args:
            agent: Agent to calculate observation for.
            world: Current world state.
            
        Returns:
            np.ndarray: Observation vector for the agent.
        """
        raise NotImplementedError()
