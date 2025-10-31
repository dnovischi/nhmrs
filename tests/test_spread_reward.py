"""Unit tests for SimpleSpreadReward class.

Tests the MPE2-inspired spread reward mode with occupancy and collision
components.
"""

import unittest
import numpy as np
from nhmrs._nhmrs_utils.core import World, Agent, Landmark
from nhmrs.simple_assignment.simple_assignment_reward import SimpleSpreadReward


class TestSimpleSpreadReward(unittest.TestCase):
    """Test suite for SimpleSpreadReward."""
    
    def setUp(self):
        """Create a simple 2-agent, 2-landmark world."""
        self.world = World()
        self.world.t = 0
        
        # Create 2 agents
        for i in range(2):
            agent = Agent()
            agent.name = f'agent_{i}'
            agent.size = 0.15
            agent.state.p_pos = np.zeros(2, dtype=np.float32)
            self.world.agents.append(agent)
        
        # Create 2 landmarks
        for i in range(2):
            landmark = Landmark()
            landmark.name = f'landmark_{i}'
            landmark.state.p_pos = np.zeros(2, dtype=np.float32)
            self.world.landmarks.append(landmark)
    
    def test_occupancy_reward_close_to_landmark(self):
        """Agent close to landmark should get higher occupancy reward (less negative)."""
        reward_computer = SimpleSpreadReward(collision_weight=1.0, agent_radius=0.15)
        
        # Place agent_0 at origin
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        
        # Place one landmark very close
        self.world.landmarks[0].state.p_pos = np.array([0.1, 0.0])
        # Place other landmark far away
        self.world.landmarks[1].state.p_pos = np.array([10.0, 10.0])
        
        # Place agent_1 far away to avoid collision
        self.world.agents[1].state.p_pos = np.array([20.0, 20.0])
        
        reward = reward_computer.compute_reward(self.world.agents[0], self.world)
        
        # Occupancy reward should be -0.1 (negative distance to closest landmark)
        # No collision penalty since other agent is far
        self.assertAlmostEqual(reward, -0.1, places=5)
    
    def test_occupancy_reward_far_from_landmarks(self):
        """Agent far from all landmarks should get low occupancy reward (very negative)."""
        reward_computer = SimpleSpreadReward(collision_weight=1.0, agent_radius=0.15)
        
        # Place agent_0 far from origin
        self.world.agents[0].state.p_pos = np.array([10.0, 10.0])
        
        # Place landmarks at origin
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[1].state.p_pos = np.array([0.0, 0.0])
        
        # Place agent_1 far away to avoid collision
        self.world.agents[1].state.p_pos = np.array([20.0, 20.0])
        
        reward = reward_computer.compute_reward(self.world.agents[0], self.world)
        
        # Distance to nearest landmark should be ~14.14
        expected_reward = -np.sqrt(200)  # -sqrt(10^2 + 10^2)
        self.assertAlmostEqual(reward, expected_reward, places=5)
    
    def test_collision_penalty_close_agents(self):
        """Agents within 2*radius should receive collision penalty."""
        reward_computer = SimpleSpreadReward(collision_weight=1.0, agent_radius=0.15)
        
        # Place both agents very close to each other
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([0.1, 0.0])  # Distance = 0.1
        
        # Place landmark at origin so occupancy reward is 0
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[1].state.p_pos = np.array([10.0, 10.0])
        
        reward = reward_computer.compute_reward(self.world.agents[0], self.world)
        
        # Occupancy: 0 (agent at landmark)
        # Collision: dist=0.1 < 2*radius=0.3, penalty = 1 - (0.1/0.3) = 0.6667
        expected_reward = 0.0 - 0.6667
        self.assertAlmostEqual(reward, expected_reward, places=3)
    
    def test_no_collision_penalty_far_agents(self):
        """Agents farther than 2*radius should have no collision penalty."""
        reward_computer = SimpleSpreadReward(collision_weight=1.0, agent_radius=0.15)
        
        # Place agents far apart
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([1.0, 0.0])  # Distance = 1.0 > 0.3
        
        # Place landmark at origin
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[1].state.p_pos = np.array([10.0, 10.0])
        
        reward = reward_computer.compute_reward(self.world.agents[0], self.world)
        
        # Only occupancy reward (0), no collision penalty
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    def test_collision_weight_scaling(self):
        """Collision weight should scale the collision penalty."""
        # High collision weight
        reward_computer_high = SimpleSpreadReward(collision_weight=5.0, agent_radius=0.15)
        
        # Low collision weight
        reward_computer_low = SimpleSpreadReward(collision_weight=0.5, agent_radius=0.15)
        
        # Setup collision scenario
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([0.1, 0.0])
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[1].state.p_pos = np.array([10.0, 10.0])
        
        reward_high = reward_computer_high.compute_reward(self.world.agents[0], self.world)
        reward_low = reward_computer_low.compute_reward(self.world.agents[0], self.world)
        
        # Base collision penalty is ~0.6667
        # High should be more negative than low
        self.assertLess(reward_high, reward_low)
        self.assertAlmostEqual(reward_high, 0.0 - 5.0 * 0.6667, places=3)
        self.assertAlmostEqual(reward_low, 0.0 - 0.5 * 0.6667, places=3)
    
    def test_reset_clears_state(self):
        """Reset should clear internal state (though SimpleSpreadReward is stateless)."""
        reward_computer = SimpleSpreadReward()
        
        # Reset should not raise an error
        reward_computer.reset()
        
        # Compute reward should still work
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([10.0, 10.0])
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[1].state.p_pos = np.array([10.0, 10.0])
        
        reward = reward_computer.compute_reward(self.world.agents[0], self.world)
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    def test_environment_integration(self):
        """Test that spread reward mode works in the full environment."""
        from nhmrs.simple_assignment.simple_assignment import Scenario, raw_env
        
        # Create scenario with spread reward mode
        scenario = Scenario(reward_mode='spread')
        
        # Create environment
        env = raw_env(
            scenario=scenario,
            max_steps=10,
            render_mode=None
        )
        
        obs, info = env.reset(seed=42)
        
        # Take a random step
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Should receive rewards for all agents
        self.assertEqual(len(rewards), len(env.agents))
        for agent in env.agents:
            self.assertIn(agent, rewards)
        
        # Rewards should be floats
        for agent, reward in rewards.items():
            self.assertIsInstance(reward, (float, np.floating))
        
        env.close()


if __name__ == '__main__':
    unittest.main()
