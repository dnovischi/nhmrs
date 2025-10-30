"""Tests for simple assignment environment."""
import numpy as np
import pytest

from nhmrs import simple_assignment_v0
from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics, DifferentialDriveKinematics


class TestSimpleAssignmentEnvironment:
    """Test suite for simple assignment environment."""

    def test_environment_creation(self):
        """Test that environment can be created."""
        
        env = simple_assignment_v0.env()
        assert env is not None
        env.close()

    def test_environment_reset(self):
        """Test that environment can be reset."""
        
        env = simple_assignment_v0.env()
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert len(env.agents) > 0
        
        env.close()

    def test_observation_space(self):
        """Test observation space consistency."""
        
        env = simple_assignment_v0.env()
        obs, info = env.reset(seed=42)
        
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            agent_obs = obs[agent]
            
            assert agent_obs in obs_space
            assert agent_obs.shape == obs_space.shape
        
        env.close()

    def test_action_space(self):
        """Test action space consistency."""
        
        env = simple_assignment_v0.env()
        obs, info = env.reset(seed=42)
        
        for agent in env.agents:
            action_space = env.action_space(agent)
            
            # Sample should be within bounds
            action = action_space.sample()
            assert action in action_space
            assert action.shape == action_space.shape
        
        env.close()

    def test_step_function(self):
        """Test that environment step works correctly."""
        
        env = simple_assignment_v0.env(max_steps=10)
        obs, info = env.reset(seed=42)
        
        # Take a step
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)
        
        # Check all agents present
        for agent in env.agents:
            assert agent in obs
            assert agent in rewards
            assert agent in terminated
            assert agent in truncated
        
        env.close()

    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        
        env = simple_assignment_v0.env(max_steps=5)
        obs, info = env.reset(seed=42)
        
        # Run until termination
        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            if all(terminated.values()) or all(truncated.values()):
                break
        
        # Should have truncated after max_steps
        assert any(truncated.values())
        
        env.close()

    def test_different_kinematics(self):
        """Test that environment works with different kinematic models."""
        
        
        # Test with unicycle
        unicycle = UnicycleKinematics(dt=0.1)
        env1 = simple_assignment_v0.env(kinematics=unicycle)
        obs1, _ = env1.reset(seed=42)
        assert len(obs1) > 0
        env1.close()
        
        # Test with differential drive
        diff_drive = DifferentialDriveKinematics(dt=0.1)
        env2 = simple_assignment_v0.env(kinematics=diff_drive)
        obs2, _ = env2.reset(seed=42)
        assert len(obs2) > 0
        env2.close()

    def test_reward_function(self):
        """Test that rewards are computed correctly."""
        
        env = simple_assignment_v0.env()
        obs, info = env.reset(seed=42)
        
        # Take a step
        actions = {agent: np.array([0.0, 0.0]) for agent in env.agents}  # No movement
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Rewards should be negative (distance-based)
        for agent in env.agents:
            assert isinstance(rewards[agent], (float, np.floating))
            # Typically negative unless agent is exactly on target
            assert rewards[agent] <= 0.0
        
        env.close()

    def test_parallel_api(self):
        """Test that parallel API works."""
        
        env = simple_assignment_v0.parallel_env()
        
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert len(env.agents) > 0
        
        env.close()


if __name__ == "__main__":
    import os
    import sys
    
    # Unset ROS environment variables to avoid plugin conflicts
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('ROS_DISTRO', None)
    os.environ.pop('AMENT_PREFIX_PATH', None)
    
    # Re-exec with clean environment
    if 'NHMRS_TEST_REEXEC' not in os.environ:
        os.environ['NHMRS_TEST_REEXEC'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # Now run pytest
    pytest.main([__file__, "-v"])
