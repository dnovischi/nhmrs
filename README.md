# NHMRS — Non-Holonomic Mobile Robot Systems

Multi-agent reinforcement learning environments for non-holonomic mobile robots, built on the [PettingZoo](https://pettingzoo.farama.org/) API.

## Overview

NHMRS provides PettingZoo-compatible environments for coordinating teams of non-holonomic mobile robots. The current environment, `simple_assignment_v0`, challenges N agents to visit M landmarks while respecting kinematic constraints.

**Key features:**
- **Pluggable kinematics**: Unicycle, differential drive, and Ackermann steering models
- **Modular reward components**: Assignment, coverage, collision avoidance, idleness penalties, and efficiency
- **Visual rendering**: Auto-zoom 2D visualization with Pygame (human window or RGB array modes)
- **Training scaffolds**: Framework-agnostic MADDPG skeletons for training and inference
- **Tests**: Unit tests for kinematics, collision detection, and environment API compliance

## Repository Structure

```text
nhmrs/
├── nhmrs/                           # Core package
│   ├── _nhmrs_utils/                # Shared utilities
│   │   ├── core.py                  # World, Agent, Landmark containers
│   │   ├── rendering.py             # Pygame visualization
│   │   ├── scenario.py              # Scenario base class
│   │   ├── reward.py                # Reward computation helpers
│   │   └── kinematics/              # Kinematic models
│   │       ├── base.py
│   │       ├── unicycle.py
│   │       ├── differential_drive.py
│   │       └── ackermann.py
│   ├── simple_assignment/
│   │   ├── simple_assignment.py     # Environment and scenario
│   │   └── simple_assignment_reward.py  # Modular reward components
│   └── simple_assignment_v0.py      # Versioned entry point
├── example_training_skeleton.py     # MADDPG training scaffold
├── example_inference_skeleton.py    # MADDPG inference scaffold
├── demo.py                          # Basic demonstration
├── demo_kinematics.py               # Kinematics comparison demo
├── tests/                           # Unit and integration tests
│   ├── test_collision.py
│   ├── test_kinematics.py
│   └── test_simple_assignment.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quick Start

```python
from nhmrs import simple_assignment_v0

# Create environment
env = simple_assignment_v0.env(render_mode="human", max_steps=500)

# Run episode
obs, info = env.reset(seed=42)
for _ in range(200):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, info = env.step(actions)
    if all(terminated.values()) or all(truncated.values()):
        break
env.close()
```

## API Reference

### Environment Interface

The environment follows the PettingZoo ParallelEnv API:

- **`reset(seed=None, options=None)`** → `(observations, infos)`
  - Returns dicts keyed by agent name
  - Observations: `np.ndarray` per agent
  - Infos: auxiliary data (currently empty)

- **`step(actions)`** → `(observations, rewards, terminated, truncated, infos)`
  - `actions`: Dict[agent_name, np.ndarray] of continuous control inputs
  - `rewards`: Dict[agent_name, float]
  - `terminated`: Dict[agent_name, bool] (always False; no early termination)
  - `truncated`: Dict[agent_name, bool] (True when reaching max_steps)

### Observation Space

Per-agent observation vector (flat):

- Own state: `[x, y, θ]` (3 values)
- Landmark positions: `[x₁, y₁, ..., x_M, y_M]` (2M values)
- Other agent states: `[x_j, y_j, θ_j]` for each j ≠ i (3(N-1) values)

**Total dimension:** `3 + 2M + 3(N-1)` where N = number of agents, M = number of landmarks.

### Action Space

Per-agent continuous Box determined by the kinematic model:

- **Unicycle:** `[v, ω]` where v ∈ [-v_max, v_max], ω ∈ [-ω_max, ω_max]
- **Differential drive:** `[v_left, v_right]`
- **Ackermann:** `[v, δ]` (velocity, steering angle)

Actions must be clipped to `env.action_space(agent).low/high` bounds.

## Configuration

### Kinematic Models

Switch between non-holonomic models:

```python
from nhmrs import simple_assignment_v0
from nhmrs._nhmrs_utils.kinematics import (
    UnicycleKinematics,
    DifferentialDriveKinematics,
    AckermannKinematics
)

# Unicycle (default)
env = simple_assignment_v0.env(
    kinematics=UnicycleKinematics(dt=0.1, v_max=2.0, omega_max=3.14),
    render_mode="human"
)

# Differential drive
env = simple_assignment_v0.env(
    kinematics=DifferentialDriveKinematics(wheelbase=0.2, dt=0.1),
    render_mode="human"
)

# Ackermann steering
env = simple_assignment_v0.env(
    kinematics=AckermannKinematics(wheelbase=0.3, dt=0.1),
    render_mode="human"
)
```

### Reward Modes

Customize reward components via `Scenario`:

```python
from nhmrs.simple_assignment.simple_assignment import Scenario

# Simple (fast learning)
scenario = Scenario(reward_mode='simple')
env = simple_assignment_v0.env(scenario=scenario)

# Balanced (default, full components)
scenario = Scenario(reward_mode='balanced')
env = simple_assignment_v0.env(scenario=scenario)

# Patrol (for N < M, emphasizes coverage/idleness)
scenario = Scenario(reward_mode='patrol')
env = simple_assignment_v0.env(scenario=scenario)
```

## Examples

### Demonstrations

```bash
# Basic demo with 3 robots running a circle policy
python demo.py

# Compare unicycle, differential drive, and Ackermann kinematics
python demo_kinematics.py
```

### Training Scaffold (MADDPG)

A framework-agnostic training skeleton demonstrating:

- Per-agent actors and centralized critics
- Shared replay buffer for joint transitions
- Target networks and soft updates
- Headless execution with random actions (dependency-free)

```bash
python example_training_skeleton.py
```

**Implementation guide:**

- Define actor/critic/target networks and optimizers in `MADDPGAgent.__init__`
- Implement deterministic action selection with exploration noise
- Build centralized (O, A) inputs for critic updates
- Compute TD targets and actor/critic losses
- Soft-update target networks

### Inference Scaffold (MADDPG)

Load trained actors and run deterministic evaluation:

```bash
python example_inference_skeleton.py
```

**Implementation guide:**

- Replace `MADDPGActor` with your trained model class
- Implement `load_actor(...)` to restore weights from checkpoints
- Return clipped actions via `.act(obs)`
- Set `RENDER_MODE='rgb_array'` for offscreen video capture

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Tests cover:

- Kinematic model updates (unicycle, differential drive, Ackermann)
- Environment API compliance (spaces, reset, step)
- Collision penalty behavior
- Reward computation

## Rendering

The environment provides 2D visualization with Pygame:

- **Window size:** 700×700 pixels
- **Auto-zoom camera:** Automatically fits all agents and landmarks
- **Agents:** Colored arrowheads with gray collision circles
- **Landmarks:** Colored circles
- **Modes:** `'human'` (window) or `'rgb_array'` (numpy array for recording)

## Dependencies

Core requirements:

- pettingzoo ≥ 1.24.0
- gymnasium ≥ 0.29.0
- numpy ≥ 1.23
- scipy ≥ 1.10 (for Hungarian algorithm)
- pygame ≥ 2.0 (for rendering)

Machine learning frameworks (optional, for MADDPG implementation):

- PyTorch, TensorFlow, or JAX

## Citation

If you use NHMRS in your research, please cite:

```bibtex
@software{nhmrs2025,
  title={NHMRS: Non-Holonomic Mobile Robot Systems Environments for Multi-Agent Reinforcement Learning},
  author={Dan Novischi},
  year={2025},
  url={https://github.com/dnovischi/nhmrs}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on the [PettingZoo](https://pettingzoo.farama.org/) multi-agent RL library
- Rendering utilities inspired by the Multi-Agent Particle Environment (MPE)
