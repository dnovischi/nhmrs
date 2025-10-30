"""Versioned entry point for multi-unicycle environment."""
from .env.simple_assignment import CustomEnvironment


def env(**kwargs):
    """
    Create a multi-unicycle environment instance.
    
    Args:
        scenario: Scenario object for initial configuration (optional)
        render_mode: "human" or "rgb_array" (optional)
        max_steps: Maximum number of steps per episode (default: 500)
    
    Returns:
        CustomEnvironment instance
    """
    return CustomEnvironment(**kwargs)
