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
nhmrs/                                   # Root package
├── nhmrs/
│   ├── _nhmrs_utils/                    # Core utilities (shared across environments)
│   │   ├── __init__.py
│   │   ├── core.py                      # World, Agent, Landmark classes
│   │   ├── rendering.py                 # Pygame rendering utilities
│   │   ├── reward.py                    # Base reward classes
│   │   ├── kinematics.py                # Unicycle, DiffDrive, Ackermann models
│   │   └── scenario.py                  # BaseScenario abstract class
│   └── simple_assignment/               # Simple assignment environment
│       ├── __init__.py
│       ├── simple_assignment.py         # Main environment (raw_env, Scenario)
│       └── simple_assignment_reward.py  # Reward computation
│           ├── SimpleAssignmentReward   # Hungarian-based task assignment
│           └── SimpleSpreadReward       # MPE2-style cooperative spreading
├── tests/                               # Test suite
│   ├── test_collision.py                # Collision detection tests
│   ├── test_kinematics.py               # Kinematic model tests
│   ├── test_simple_assignment.py        # Environment API tests
│   └── test_spread_reward.py            # Spread reward tests
├── demo.py                              # Basic demo (3 robots, circle policy)
├── demo_kinematics.py                   # Kinematics comparison demo
├── demo_spread.py                       # Spread reward mode demo
├── example_training_skeleton.py         # MADDPG training scaffold
├── example_inference_skeleton.py        # MADDPG inference scaffold
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

### Key Components

**Core Utilities (`_nhmrs_utils/`)**:

- `core.py`: Fundamental data structures (World, Agent, Landmark) used across all environments
- `kinematics.py`: Pluggable kinematic models (Unicycle, DifferentialDrive, Ackermann) with consistent interface
- `rendering.py`: Pygame-based visualization with auto-zoom camera and geometry primitives
- `scenario.py`: Abstract base class for environment scenarios
- `reward.py`: Base classes and utilities for reward computation

**Simple Assignment Environment (`simple_assignment/`)**:

- `simple_assignment.py`: PettingZoo ParallelEnv implementation with `raw_env` class and `Scenario` class
- `simple_assignment_reward.py`: Two reward modes:
  - `SimpleAssignmentReward`: Hungarian algorithm-based task assignment with 5 components (assignment, coverage, collision, idleness, efficiency)
  - `SimpleSpreadReward`: MPE2-inspired occupancy + collision reward for cooperative spreading

**Example Scripts**:

- `demo.py`: Basic 3-robot demonstration with circle motion policy
- `demo_kinematics.py`: Side-by-side comparison of unicycle, differential drive, and Ackermann models
- `demo_spread.py`: Demonstration of the spread reward mode
- `example_training_skeleton.py`: Framework-agnostic MADDPG training scaffold with TODOs
- `example_inference_skeleton.py`: MADDPG inference scaffold for loading trained policies

**Tests (`tests/`)**:

- Complete unit test coverage (23 tests)
- Tests for kinematics, collision detection, reward computation, and PettingZoo API compliance
- Run with: `python -m pytest tests/ -v`

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

# Simple (fast learning: assignment + collision only)
scenario = Scenario(reward_mode='simple')
env = simple_assignment_v0.env(scenario=scenario)

# Balanced (default: full multi-component assignment reward)
scenario = Scenario(reward_mode='balanced')
env = simple_assignment_v0.env(scenario=scenario)

# Patrol (for N < M: emphasizes coverage/idleness)
scenario = Scenario(reward_mode='patrol')
env = simple_assignment_v0.env(scenario=scenario)

# Spread (MPE2-style: occupancy + collision, no explicit assignment)
scenario = Scenario(reward_mode='spread')
env = simple_assignment_v0.env(scenario=scenario)
```

**Reward Mode Comparison:**

| Mode | Reward Class | Use Case | Components |
|------|--------------|----------|------------|
| `simple` | SimpleAssignmentReward | Fast learning, basic task allocation | Assignment + Collision |
| `balanced` | SimpleAssignmentReward | General-purpose multi-component | Assignment + Coverage + Collision + Idleness + Efficiency |
| `patrol` | SimpleAssignmentReward | Persistent coverage (N < M) | Balanced weights with high coverage/idleness |
| `spread` | SimpleSpreadReward | Cooperative spreading without assignments | Occupancy (distance to landmarks) + Collision |

**When to use `spread` mode:**

- Symmetric reward structure (all agents receive same reward signal)
- No need for explicit task assignment (Hungarian algorithm avoided)
- Inspired by MPE2's `simple_spread_v3` environment
- Agents naturally spread to cover landmarks via occupancy gradient

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
