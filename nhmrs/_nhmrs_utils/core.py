"""Core classes for NHMRS environments.

Adapted from MPE2's core but modified for non-holonomic kinematic models
instead of point-mass physics.
"""

import numpy as np


class EntityState:
    """Physical state of all entities (agents and landmarks)."""
    
    def __init__(self):
        # Physical position [x, y]
        self.p_pos = None


class AgentState(EntityState):
    """State of agents including orientation and communication."""
    
    def __init__(self):
        super().__init__()
        # Orientation (heading angle in radians)
        self.p_orient = None
        # Communication utterance
        self.c = None


class Action:
    """Action of the agent."""
    
    def __init__(self):
        # Physical action (kinematic controls: [v, ω] or [v_left, v_right], etc.)
        self.u = None
        # Communication action
        self.c = None


class Entity:
    """Properties and state of physical world entity."""
    
    def __init__(self):
        # Name identifier
        self.name = ""
        # Entity size (radius for collision/rendering)
        self.size = 0.050
        # Entity can move / be pushed
        self.movable = False
        # Color for rendering [r, g, b]
        self.color = None
        # State
        self.state = EntityState()


class Landmark(Entity):
    """Properties of landmark entities (static task locations)."""
    
    def __init__(self):
        super().__init__()
        # Landmarks are not movable by default
        self.movable = False


class Agent(Entity):
    """Properties of agent entities (non-holonomic robots)."""
    
    def __init__(self):
        super().__init__()
        # Agents are movable by default
        self.movable = True
        # Cannot send communication signals by default
        self.silent = False
        # State with orientation
        self.state = AgentState()
        # Action
        self.action = Action()
        # Kinematic model for this agent
        self.kinematics = None
        # Script behavior to execute (for scripted agents)
        self.action_callback = None


class World:
    """Multi-agent world for non-holonomic robots."""
    
    def __init__(self):
        # List of agents and landmarks (can change at execution-time)
        self.agents = []
        self.landmarks = []
        # Communication channel dimensionality
        self.dim_c = 0
        # Position dimensionality
        self.dim_p = 2
        # Color dimensionality
        self.dim_color = 3
        # Simulation timestep
        self.dt = 0.1
        # Current timestep (for reward computation)
        self.t = 0
    
    @property
    def entities(self):
        """Return all entities in the world."""
        return self.agents + self.landmarks
    
    @property
    def policy_agents(self):
        """Return all agents controllable by external policies."""
        return [agent for agent in self.agents if agent.action_callback is None]
    
    @property
    def scripted_agents(self):
        """Return all agents controlled by world scripts."""
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    def step(self):
        """Update state of the world using kinematic models.
        
        Unlike MPE2 which uses force-based physics, this directly updates
        agent positions using their kinematic models.
        """
        # Set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # Update agent positions via kinematics (not physics!)
        for agent in self.agents:
            if agent.movable and agent.kinematics is not None:
                # Get current state as kinematic state [x, y, θ]
                kin_state = np.array([
                    agent.state.p_pos[0],
                    agent.state.p_pos[1],
                    agent.state.p_orient
                ], dtype=np.float32)
                
                # Step kinematics with action
                new_kin_state = agent.kinematics.step(kin_state, agent.action.u)
                
                # Update agent state
                agent.state.p_pos = new_kin_state[:2].copy()
                agent.state.p_orient = new_kin_state[2]
        
        # Update agent communication state
        for agent in self.agents:
            self.update_agent_state(agent)
    
    def update_agent_state(self, agent):
        """Update agent communication state (from MPE2)."""
        # Set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            # Communication action passes through directly
            agent.state.c = agent.action.c
