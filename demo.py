"""Demo: 3 robots moving in circles to verify rendering."""
import time
import numpy as np

from nhmrs import simple_assignment_v0


def circle_policy(agent_id, obs, t):
    """Simple policy: move forward with constant angular velocity (creates circles)."""
    v = 0.2  # constant forward velocity
    omega = -0.5 - agent_id * 0.2  # different rotation speed per agent
    return np.array([v, omega], dtype=np.float32)


def main():
    print("=" * 70)
    print("NHMRS Environment Demo")
    print("=" * 70)
    print("\nScenario:")
    print("  - 3 non-holonomic mobile robots (colored arrowheads)")
    print("  - 4 landmarks (red circles) arranged in a square pattern")
    print("  - Robots move in circles with different rotation speeds")
    print("  - Camera auto-zooms to fit all entities (like MPE2)")
    print("\nStarting visualization...")
    print("Close the window or wait 15 seconds to exit.\n")
    
    # Create environment
    environment = simple_assignment_v0.env(render_mode="human", max_steps=500)
    
    obs, info = environment.reset(seed=42)
    
    try:
        # Run for ~15 seconds at 10 Hz
        for step in range(150):
            actions = {}
            for i, agent in enumerate(environment.agents):
                actions[agent] = circle_policy(i, obs[agent], step)
            
            obs, rewards, terminated, truncated, info = environment.step(actions)
            environment.render()
            
            if step % 30 == 0:
                print(f"  Step {step}/150 - Avg reward: {np.mean(list(rewards.values())):.2f}")
            
            time.sleep(0.1)  # 10 Hz
            
            if all(terminated.values()) or all(truncated.values()):
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        environment.close()
        print("\nDemo completed!")
        print("=" * 70)


if __name__ == "__main__":
    main()
