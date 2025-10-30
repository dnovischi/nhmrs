# NHMRS - Non-Holonomic Mobile Robot Systems Environments

A collection of [PettingZoo](https://pettingzoo.farama.org/) environments for multi-agent reinforcement learning with Non-Holonomic Mobile Robot Systems (NHMRS). These environments focus on task allocation problems where agents must coordinate to efficiently assign themselves to tasks.

## Environments

### Simple Assignment (`simple_assignment_v0`)

A basic task allocation environment where non-holonomic agents (unicycle model) must be assigned to task locations to minimize total cost (e.g., travel distance or time).

## Features

- **Non-holonomic dynamics**: Unicycle kinematic model with continuous control `[v, ω]`
- **Task allocation**: Simple assignment problem with n agents and m tasks
- **PettingZoo compatibility**: Standard parallel environment interface
- **Visual rendering**: Real-time visualization with auto-zoom camera
- **Configurable scenarios**: Customizable agent counts, task locations, and world setup

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/nhmrs.git
cd nhmrs

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Via pip (when published)

```bash
pip install nhmrs
```

## Quick Start

```python
from nhmrs import simple_assignment_v0

# Create environment
env = simple_assignment_v0.env(render_mode="human")

# Run episode
obs, info = env.reset(seed=42)
for _ in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    if all(terminated.values()) or all(truncated.values()):
        break
env.close()
```

### Using Different Kinematic Models

```python
from nhmrs import simple_assignment_v0
from nhmrs._nhmrs_utils.kinematics import DifferentialDriveKinematics

# Create environment with differential drive kinematics
kinematics = DifferentialDriveKinematics(wheelbase=0.2, dt=0.1)
env = simple_assignment_v0.env(kinematics=kinematics, render_mode="human")

obs, info = env.reset(seed=42)
# ... rest of code
```

## Demos

Run the included demos to see robots with different kinematic models:

```bash
# Basic demo with 3 robots in circular motion
python docs/demo.py

# Compare different kinematic models (unicycle, differential drive, ackermann)
python docs/demo_kinematics.py
```

The demos show agents (colored arrowheads) moving toward task locations (red circles).

## Project Structure

```
nhmrs/
├── nhmrs/                          # Main package
│   ├── __init__.py                 # Package-level exports
│   ├── _nhmrs_utils/               # Shared utilities (MPE2-inspired)
│   │   ├── __init__.py
│   │   ├── core.py                 # Core abstractions (Entity, Agent, World)
│   │   ├── scenario.py             # Base scenario interface
│   │   ├── rendering.py            # Pygame-based rendering utilities
│   │   └── kinematics/             # Kinematic models
│   │       ├── __init__.py
│   │       ├── base.py             # Abstract base class
│   │       ├── unicycle.py         # Unicycle model
│   │       ├── differential_drive.py  # Differential drive model
│   │       └── ackermann.py        # Ackermann steering model
│   ├── simple_assignment/          # Simple assignment environment
│   │   ├── __init__.py
│   │   └── simple_assignment.py    # Environment + scenario implementation
│   └── simple_assignment_v0.py     # Top-level convenience import
├── demo.py                         # Basic demo script
├── tests/                          # Test suite (14 tests, all passing)
│   ├── test_kinematics.py          # Kinematic model tests (5 tests)
│   └── test_simple_assignment.py   # Environment tests (9 tests)
├── requirements.txt                # Dependencies
├── setup.py                        # Setup script
├── pyproject.toml                  # Modern Python config (PEP 518)
├── ABSTRACTIONS.md                 # Core abstractions documentation
├── STRUCTURE.md                    # Design decisions
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Core Abstractions

NHMRS follows an MPE2-inspired design with core abstractions adapted for non-holonomic robotics:

- **Entity**: Base class for all world objects (agents, landmarks)
- **Agent**: Controllable entity with kinematics model
- **Landmark**: Static entity representing task locations
- **World**: Container managing all entities and stepping simulation
- **Scenario**: Defines task-specific world configuration and rewards

For detailed documentation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).

### Key Differences from MPE2

NHMRS uses **kinematic models** instead of force-based physics:

| Aspect | MPE2 (Point Mass) | NHMRS (Non-Holonomic) |
|--------|-------------------|------------------------|
| State | `[x, y, vx, vy]` | `[x, y, θ]` |
| Action | Forces `[fx, fy]` | Kinematic controls `[v, ω]` |
| Dynamics | Force integration | Kinematic stepping |
| Constraints | None (holonomic) | Non-holonomic (orientation) |

### Kinematics Models

The `nhmrs._nhmrs_utils.kinematics` package provides reusable kinematic models:

```python
from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics, DifferentialDriveKinematics, AckermannKinematics
from nhmrs import simple_assignment_v0
import numpy as np

# Use with any environment
unicycle = UnicycleKinematics(dt=0.1, v_max=2.0, omega_max=np.pi)
env = simple_assignment_v0.env(kinematics=unicycle)
```

### Adding New Environments

To add a new environment to the package:

1. Create a new directory under `nhmrs/`: `nhmrs/new_environment/`
2. Structure it following PettingZoo conventions:
   ```
   nhmrs/new_environment/
   ├── __init__.py
   ├── new_environment_v0.py
   └── env/
       ├── __init__.py
       ├── new_environment.py
       └── scenario.py
   ```
3. Import kinematics from `nhmrs.kinematics`
4. Update `pyproject.toml` to include the new packages
5. Add documentation and demo in `docs/`
6. Follow the MPE2-style structure for consistency

## Configuration

### Scenario

Edit `simple_assignment/env/scenario.py` to customize:
- Number of agents and landmarks
- Initial positions and spawn patterns
- World configuration

### Environment Parameters

```python
env(
    scenario=scenario,      # Scenario object (default: Scenario())
    render_mode="human",    # "human", "rgb_array", or None
    max_steps=500          # Maximum steps per episode
)
```

### Agent Parameters (in simple_assignment/env/simple_assignment.py)

- `dt`: Time step (default: 0.1)
- `v_max`: Maximum velocity (default: 2.0)
- `omega_max`: Maximum angular velocity (default: π)
- `agent_size`: Visual size (default: 0.075, MPE-like scale)

## Observation Space

Per agent observation (flat vector):
- Own state: `[x, y, θ]` (3 values)
- All landmarks: `[x₁, y₁, x₂, y₂, ...]` (2M values)
- Other agents: `[x₁, y₁, θ₁, x₂, y₂, θ₂, ...]` (3(N-1) values)

Total dimension: `3 + 2M + 3(N-1)` where N = agents, M = landmarks

## Action Space

Per agent: `Box([v, ω])` where:
- `v ∈ [-v_max, v_max]`: Forward/backward velocity
- `ω ∈ [-ω_max, ω_max]`: Angular velocity (rotation rate)

## Dynamics

The agents follow unicycle kinematics:

```
dx/dt = v * cos(θ)
dy/dt = v * sin(θ)
dθ/dt = ω
```

Where:
- `(x, y)`: Agent position in world coordinates
- `θ`: Agent orientation (radians)
- `v`: Linear velocity (control input)
- `ω`: Angular velocity (control input)

## Rewards

Default reward function:
```
reward = -min(||agent_position - task_position||)
```

This encourages agents to minimize distance to their nearest task location, providing a learning signal for the assignment problem.

## Rendering

The environment includes visual rendering with:
- **Agents**: Colored arrowheads indicating position and orientation
- **Tasks**: Red circles showing task locations
- **Auto-zoom camera**: View adjusts to fit all entities
- **Render modes**: `"human"` (window) or `"rgb_array"` (for recording)

Rendering uses pygame for visualization.

## Dependencies

- pettingzoo >= 1.24.0
- gymnasium >= 0.29.0
- numpy >= 1.23
- scipy >= 1.10
- pygame >= 2.0

## Citation

If you use these environments in your research, please cite:

```bibtex
@software{nhmrs2025,
  title={NHMRS: Non-Holonomic Mobile Robot Systems Environments for Multi-Agent Reinforcement Learning},
  author={Dan Novischi},
  year={2025},
  url={https://github.com/dnovischi/nhmrs}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using [PettingZoo](https://pettingzoo.farama.org/) framework
- Rendering inspired by Multi-Agent Particle Environment (MPE)
