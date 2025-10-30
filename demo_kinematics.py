"""Demo: Compare different kinematic models for NHMRS task allocation."""
import time
import numpy as np

from nhmrs import simple_assignment_v0
from nhmrs._nhmrs_utils.kinematics import (
    UnicycleKinematics,
    DifferentialDriveKinematics,
    AckermannKinematics
)


def circle_policy(agent_id, obs, t, kinematics_type="unicycle"):
    """Policy that creates circular motion for different kinematics."""
    if kinematics_type == "unicycle":
        # [v, ω]
        v = 0.2
        omega = -0.5 - agent_id * 0.2
        return np.array([v, omega], dtype=np.float32)
    
    elif kinematics_type == "differential":
        # [v_left, v_right] - different wheel speeds create rotation
        v_base = 0.2
        diff = 0.05 + agent_id * 0.02
        return np.array([v_base - diff, v_base + diff], dtype=np.float32)
    
    elif kinematics_type == "ackermann":
        # [v, δ] - constant velocity with steering angle
        v = 0.2
        delta = -0.3 - agent_id * 0.1
        return np.array([v, delta], dtype=np.float32)


def run_demo(kinematics_model, kinematics_type, model_name):
    """Run demo with specified kinematics model."""
    print("=" * 70)
    print(f"{model_name} Kinematics Demo")
    print("=" * 70)
    print(f"\nKinematics Model: {model_name}")
    print(f"  - State dimension: {kinematics_model.state_dim}")
    print(f"  - Action dimension: {kinematics_model.action_dim}")
    action_low, action_high = kinematics_model.get_action_bounds()
    print(f"  - Action bounds: [{action_low}, {action_high}]")
    print(f"\nScenario:")
    print(f"  - 3 non-holonomic robots (colored arrowheads)")
    print(f"  - 4 task locations (red circles)")
    print(f"  - Robots move with {model_name.lower()} dynamics")
    print(f"\nStarting visualization...")
    print(f"Close the window or wait 15 seconds to exit.\n")
    
    # Create environment with specified kinematics
    environment = simple_assignment_v0.env(
        render_mode="human",
        max_steps=500,
        kinematics=kinematics_model
    )
    
    obs, info = environment.reset(seed=42)
    
    try:
        # Run for ~15 seconds at 10 Hz
        for step in range(150):
            actions = {}
            for i, agent in enumerate(environment.agents):
                actions[agent] = circle_policy(i, obs[agent], step, kinematics_type)
            
            obs, rewards, terminated, truncated, info = environment.step(actions)
            environment.render()
            
            if step % 30 == 0:
                avg_reward = np.mean(list(rewards.values()))
                print(f"  Step {step}/150 - Avg reward: {avg_reward:.2f}")
            
            time.sleep(0.1)  # 10 Hz
            
            if all(terminated.values()) or all(truncated.values()):
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        environment.close()
        print("\nDemo completed!")
        print("=" * 70)
        print()


def main():
    """Run demos for different kinematics models."""
    print("\n" + "=" * 70)
    print("NHMRS Kinematics Comparison Demo")
    print("=" * 70)
    print("\nThis demo showcases different kinematic models for non-holonomic")
    print("mobile robots in a task allocation scenario.")
    print("\nAvailable models:")
    print("  1. Unicycle - Standard non-holonomic model [v, ω]")
    print("  2. Differential Drive - Two-wheel model [v_left, v_right]")
    print("  3. Ackermann - Car-like steering [v, δ]")
    
    choice = input("\nSelect model (1-3) or 'all' to run sequentially [1]: ").strip() or "1"
    
    models = {
        "1": (UnicycleKinematics(dt=0.1), "unicycle", "Unicycle"),
        "2": (DifferentialDriveKinematics(dt=0.1), "differential", "Differential Drive"),
        "3": (AckermannKinematics(dt=0.1), "ackermann", "Ackermann"),
    }
    
    if choice.lower() == "all":
        for key in ["1", "2", "3"]:
            kinematics, kin_type, name = models[key]
            run_demo(kinematics, kin_type, name)
            if key != "3":
                input("\nPress Enter to continue to next model...")
    elif choice in models:
        kinematics, kin_type, name = models[choice]
        run_demo(kinematics, kin_type, name)
    else:
        print("Invalid choice. Running Unicycle model.")
        kinematics, kin_type, name = models["1"]
        run_demo(kinematics, kin_type, name)
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
