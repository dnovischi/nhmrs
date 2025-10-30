# Multi-Unicycle Environment

A custom PettingZoo environment for multi-agent unicycle robots with continuous control.

## Features

- **Unicycle dynamics**: Agents move with continuous actions `[v, ω]` (velocity, angular velocity)
- **Arrowhead rendering**: Robots visualized as oriented arrows (similar to MPE)
- **Auto-zoom camera**: View automatically adjusts to fit all entities
- **Scenario-based**: Configurable initial positions and world setup
- **Unbounded world**: No boundaries, camera follows agents

## Installation

```bash
pip install -r requirements.txt
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

Run the included demo to see 3 robots moving in circles:

```bash
python demo.py
```

## Project Structure

```
nhmrs/
├── simple_assignment/
│   ├── env/
│   │   ├── __init__.py
│   │   ├── simple_assignment.py  # Main environment
│   │   ├── rendering.py          # Custom pyglet renderer
│   │   └── scenario.py           # Scenario configuration
│   ├── __init__.py
│   └── simple_assignment_v0.py   # Versioned entry point
├── demo.py                        # Example demo
├── README.md
└── requirements.txt
```

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

Unicycle kinematics:
```
dx/dt = v * cos(θ)
dy/dt = v * sin(θ)
dθ/dt = ω
```

## Rewards

Default: Negative distance to nearest landmark
- `reward = -min(||agent_position - landmark_position||)`
- Encourages agents to approach landmarks

## Rendering

- Custom pyglet-based renderer (MPE-style)
- Arrowheads show agent orientation
- Red circles show landmarks
- Camera auto-zooms to fit all entities
- Requires pyglet >= 1.5, < 2.0 (legacy OpenGL)

## Dependencies

- pettingzoo >= 1.24.0
- gymnasium >= 0.29.0
- numpy >= 1.23
- scipy >= 1.10
- pyglet >= 1.5, < 2.0

## License

MIT
