"""Demo script for the 'spread' reward mode.

This script demonstrates the MPE2-style SimpleSpreadReward where agents
are encouraged to spread out and occupy landmarks without explicit task
assignment. Agents receive occupancy reward (negative distance to nearest
landmark) and collision penalties.

Usage:
    python demo_spread.py
"""

import numpy as np
from nhmrs.simple_assignment.simple_assignment import Scenario, raw_env
from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics


def circle_policy(agent_id, obs):
    """Simple policy: move forward with constant angular velocity (creates circles).
    
    Args:
        agent_id: Agent index
        obs: Observation vector (not used)
        
    Returns:
        np.ndarray: [v, omega] action
    """
    v = 0.2  # constant forward velocity
    omega = -0.5 - agent_id * 0.2  # different rotation speed per agent
    return np.array([v, omega], dtype=np.float32)


def main():
    """Run the spread reward demo with visualization."""
    print("=" * 70)
    print("NHMRS Simple Assignment - Spread Reward Mode Demo")
    print("=" * 70)
    print("\nThis demo uses the 'spread' reward mode inspired by MPE2's")
    print("simple_spread_v3. Agents receive:")
    print("  - Occupancy reward: -distance_to_nearest_landmark")
    print("  - Collision penalty: When agents get too close")
    print("\nAgents are running a simple circular motion policy.")
    print("Press Ctrl+C to exit.\n")
    
    # Create scenario with spread reward
    scenario = Scenario(reward_mode='spread')
    
    # Create environment with human rendering
    env = raw_env(
        scenario=scenario,
        render_mode='human',
        max_steps=500,
        kinematics=UnicycleKinematics(dt=0.1, v_max=2.0, omega_max=3.14)
    )
    
    # Run multiple episodes
    n_episodes = 3
    
    for episode in range(n_episodes):
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'=' * 70}\n")
        
        obs, info = env.reset(seed=42 + episode)
        
        episode_rewards = {agent: 0.0 for agent in env.agents}
        
        for t in range(env.max_steps):
            # Generate actions using circle policy
            actions = {}
            for i, agent in enumerate(env.agents):
                actions[agent] = circle_policy(i, obs[agent])
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
            
            # Render
            env.render()
            
            # Check if episode is done
            if all(terminated.values()) or all(truncated.values()):
                break
        
        # Print episode statistics
        print(f"\nEpisode {episode + 1} complete after {t + 1} steps")
        print("Total rewards per agent:")
        for agent, total_reward in episode_rewards.items():
            print(f"  {agent}: {total_reward:.2f}")
    
    env.close()
    print("\nDemo complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
