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
from simple_assignment.simple_assignment_v0 import env
from simple_assignment.env.scenario import Scenario

# Create environment
scenario = Scenario()
e = env(scenario=scenario, render_mode="human")

# Run episode
obs, info = e.reset(seed=42)
for _ in range(100):
    actions = {agent: e.action_space(agent).sample() for agent in e.agents}
    obs, rewards, terminated, truncated, info = e.step(actions)
    e.render()
    if all(terminated.values()) or all(truncated.values()):
        break
e.close()
```

## Demo

Run the included demo to see 3 robots with circular motion policies:

```bash
python demo.py
```

The demo shows agents (colored arrowheads) moving toward task locations (red circles).

## Project Structure

```
nhmrs/
├── simple_assignment/              # Simple assignment environment
│   ├── __init__.py
│   ├── simple_assignment_v0.py    # Environment entry point
│   └── env/
│       ├── __init__.py
│       ├── simple_assignment.py   # Core environment logic
│       ├── rendering.py           # Visualization
│       └── scenario.py            # Initial configuration
├── demo.py                        # Demo script
├── requirements.txt               # Dependencies
├── setup.py                       # Setup script
├── pyproject.toml                 # Modern Python config
└── README.md                      # This file
```

### Adding New Environments

To add a new environment to the package:

1. Create a new directory: `nhmrs/new_environment/`
2. Structure it following PettingZoo conventions:
   ```
   new_environment/
   ├── __init__.py
   ├── new_environment_v0.py
   └── env/
       ├── __init__.py
       ├── new_environment.py
       └── scenario.py
   ```
3. Update `pyproject.toml` to include the new package
4. Add documentation and demo

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

Rendering requires `pyglet >= 1.5, < 2.0`

## Dependencies

- pettingzoo >= 1.24.0
- gymnasium >= 0.29.0
- numpy >= 1.23
- scipy >= 1.10
- pyglet >= 1.5, < 2.0

## Citation

If you use these environments in your research, please cite:

```bibtex
@software{nhmrs2025,
  title={NHMRS: Non-Holonomic Mobile Robot Systems Environments for Multi-Agent Reinforcement Learning},
  author={NHMRS Project},
  year={2025},
  url={https://github.com/yourusername/nhmrs}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using [PettingZoo](https://pettingzoo.farama.org/) framework
- Rendering inspired by Multi-Agent Particle Environment (MPE)
