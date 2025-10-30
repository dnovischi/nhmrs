"""NHMRS core data model and simulation loop (kinematics-first).

This module defines the minimal, reusable building blocks every NHMRS
environment uses:
- State containers (EntityState, AgentState)
- Action container (Action)
- World objects (Entity, Landmark, Agent)
- The World simulation container and its one-step update

Key concepts:
- Coordinate frame: 2D world coordinates, origin at (0, 0), x→right, y→up
- Units: abstract distance (world.dt sets the fixed time step, default 0.1)
- Agents are updated by kinematic models (no force/physics engine here)
- Each Agent owns a `kinematics` object that interprets `action.u`
  and advances its state [x, y, theta] over one world step.

What this module DOES NOT do:
- No collision resolution or contact dynamics
- No reward computation
- No rendering (only provides size/color fields others can use)

Typical per-step flow (handled by environments using this World):
1) Environment sets/collects each agent's Action (action.u, action.c)
2) World.step():
   - Calls scripted `action_callback` if present (to fill `action`)
   - Advances all movable agents via their `kinematics.step(...)`
   - Updates communication state (passes through action.c by default)
3) Environment increments `world.t` (timestep) and computes rewards/obs
"""

import numpy as np


class EntityState:
    """Position-only state shared by all entities.

    Attributes:
        p_pos (np.ndarray | None): 2D position [x, y] in world coordinates.
            Shape (2,), dtype float32/float64. None until initialized.
    """
    
    def __init__(self):
        # Physical position [x, y] in world coordinates
        self.p_pos = None


class AgentState(EntityState):
    """Extended state for agents (adds orientation and optional comms).

    Attributes:
        p_orient (float | None): Heading angle theta in radians, CCW from +x.
            Typically in range [-π, π]. None until initialized.
        c (np.ndarray | None): Optional communication/utterance vector with
            length equal to World.dim_c. None until set.
    """
    
    def __init__(self):
        super().__init__()
        # Orientation (heading angle in radians, range typically [-pi, pi])
        self.p_orient = None
        # Communication utterance (size = world.dim_c), if used
        self.c = None


class Action:
    """Container for agent actions.

    Semantics are defined by the agent's `kinematics` model:
      - Unicycle: u = [v, omega]
      - Differential drive: u = [v_left, v_right]
      - Ackermann: u = [v, delta]
    The environment sets this per step before World.step().

    Attributes:
        u (np.ndarray | list | None): Kinematic controls for this step.
        c (np.ndarray | None): Optional communication action (size = world.dim_c).
    """
    
    def __init__(self):
        # Physical action (kinematic controls: [v, ω], [v_l, v_r], etc.)
        self.u = None
        # Communication action (passed into AgentState.c by World.update_agent_state)
        self.c = None


class Entity:
    """Base class for anything placed in the world.

    Attributes:
        name (str): Identifier (stable across an episode).
        size (float): Visual/collision radius in world units (used by renderers
            and downstream collision checks, if any).
        movable (bool): If True, World.step() may change its state.
        color (np.ndarray | list | None): [r, g, b] in [0, 1] for rendering.
        state (EntityState): Position-only state by default.
    """
    
    def __init__(self):
        # Name identifier
        self.name = ""
        # Entity size (radius used for rendering/collision visualization)
        self.size = 0.050
        # Whether the entity can move
        self.movable = False
        # RGB color for rendering (values in [0, 1]); None = renderer picks default
        self.color = None
        # State container (position only by default)
        self.state = EntityState()


class Landmark(Entity):
    """Static task/location entity.

    Landmarks do not move by default (movable = False).
    Environments typically assign `.size` and `.color` and place them once.
    """
    
    def __init__(self):
        super().__init__()
        # Landmarks are not movable by default
        self.movable = False


class Agent(Entity):
    """Controllable non-holonomic robot.

    Adds:
        - `state` with orientation
        - `action` container
        - `kinematics` model responsible for state updates
        - optional `action_callback` for scripted agents

    Attributes:
        silent (bool): If True, communication is suppressed (state.c = zeros).
        action (Action): Actions to apply this step (set by env or callback).
        kinematics: Object with `step(state: [x,y,theta], u) -> [x,y,theta]`.
        action_callback (callable | None): If set, called each step as
            `action_callback(agent, world)` to produce an Action for scripted agents.
    """
    
    def __init__(self):
        super().__init__()
        # Agents are movable by default
        self.movable = True
        # If True, communication is silenced (overridden in update_agent_state)
        self.silent = False
        # Orientation-enabled state
        self.state = AgentState()
        # Action container (kinematic + communication)
        self.action = Action()
        # Kinematic model (must provide `step([x,y,theta], u) -> [x,y,theta]`)
        self.kinematics = None
        # Optional scripted policy hook
        self.action_callback = None


class World:
    """Container for all entities and the simulation step.

    Responsibilities:
        - Hold agents and landmarks
        - Provide a single `step()` that advances all movable agents by
          delegating to their `kinematics` models
        - Handle communication pass-through each step

    Attributes:
        agents (list[Agent]): All agents in the world.
        landmarks (list[Landmark]): All landmarks in the world.
        dim_c (int): Communication channel width (size of action.c/state.c).
        dim_p (int): Position dimension (fixed at 2).
        dim_color (int): Color channels (3 = RGB).
        dt (float): Simulation step in time units.
        t (int): Current timestep counter (environments should increment).
    """
    
    def __init__(self):
        # Entities
        self.agents = []
        self.landmarks = []
        # Communication channel dimensionality (0 = no comms)
        self.dim_c = 0
        # Position dimension (2D)
        self.dim_p = 2
        # Color dimension (RGB)
        self.dim_color = 3
        # Simulation timestep (environment chooses physical meaning)
        self.dt = 0.1
        # Current timestep (environment increments after step if desired)
        self.t = 0
    
    @property
    def entities(self):
        """All entities (agents first, then landmarks)."""
        return self.agents + self.landmarks
    
    @property
    def policy_agents(self):
        """Agents controlled externally (no action_callback)."""
        return [agent for agent in self.agents if agent.action_callback is None]
    
    @property
    def scripted_agents(self):
        """Agents using a scripted policy via `action_callback`."""
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    def step(self):
        """Advance the world by one time step using kinematics.

        Order of operations:
            1) For each scripted agent, call its `action_callback(agent, world)`
               to populate `agent.action`.
            2) For each movable agent with a kinematics model:
               - Build current kinematic state [x, y, theta]
               - Call `kinematics.step(state, action.u)`
               - Write the returned state back to the agent (pos + orient)
            3) Update per-agent communication state:
               - If `agent.silent` -> zeros
               - Else pass through `agent.action.c`

        Notes:
            - No collision handling or physics are performed here.
            - `world.dt` is available for kinematics models; this loop does not
              enforce it beyond being a shared parameter.
            - `world.t` is NOT incremented here to keep the loop side-effect free
              for environments that need tighter control; envs typically do
              `world.t += 1` after calling `world.step()`.
        """
        # 1) Scripted actions
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # 2) Kinematic updates (no forces/accelerations here)
        for agent in self.agents:
            if agent.movable and agent.kinematics is not None:
                # Current state as [x, y, theta]
                kin_state = np.array(
                    [agent.state.p_pos[0], agent.state.p_pos[1], agent.state.p_orient],
                    dtype=np.float32,
                )
                # Next state from the agent's kinematic model
                new_kin_state = agent.kinematics.step(kin_state, agent.action.u)
                # Commit updates
                agent.state.p_pos = new_kin_state[:2].copy()
                agent.state.p_orient = new_kin_state[2]
        
        # 3) Communication updates
        for agent in self.agents:
            self.update_agent_state(agent)
    
    def update_agent_state(self, agent):
        """Update communication state for a single agent.

        If `agent.silent` is True:
            - `agent.state.c` is set to a zero vector of length `dim_c`.
        Else:
            - `agent.state.c` directly mirrors `agent.action.c`.

        This function does not modify physical state.
        """
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c) if self.dim_c > 0 else None
        else:
            # Communication action passes through directly (may be None)
            agent.state.c = agent.action.c
