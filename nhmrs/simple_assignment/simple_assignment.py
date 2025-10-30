"""Simple Assignment v0 - NHMRS Task Allocation Environment.

A PettingZoo environment where non-holonomic mobile robots must be assigned
to tasks/landmarks to minimize total cost. Supports multiple kinematic models.

Example:
    >>> from nhmrs import simple_assignment_v0
    >>> env = simple_assignment_v0.env(render_mode="human")
    >>> obs, info = env.reset(seed=42)
"""

import numpy as np
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo import AECEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle
from scipy.optimize import linear_sum_assignment

# Core NHMRS imports
from nhmrs._nhmrs_utils.core import World, Agent, Landmark
from nhmrs._nhmrs_utils.scenario import BaseScenario
from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics

# Simple assignment specific imports
from .simple_assignment_reward import SimpleAssignmentReward

# Rendering imports (optional)
try:
    from nhmrs._nhmrs_utils.rendering import (
        Viewer, Transform, make_circle, make_polygon, make_agent_geom
    )
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False


def make_env(raw_env):
    """Wrap raw environment with standard PettingZoo wrappers."""
    def env_fn(**kwargs):
        return raw_env(**kwargs)
    return env_fn


class raw_env(ParallelEnv, EzPickle):
    """NHMRS Simple Assignment Environment (raw).
    
    A task allocation environment where non-holonomic agents must be assigned
    to task locations to minimize total travel cost.
    
    Args:
        scenario: Scenario object for world configuration (optional)
        render_mode: "human", "rgb_array", or None
        max_steps: Maximum steps per episode (default: 500)
        kinematics: KinematicsModel instance (default: UnicycleKinematics)
        agent_size: Visual size of agents for rendering (default: 0.075)
    """
    
    metadata = {
        "name": "simple_assignment_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, scenario=None, render_mode=None, max_steps=500, 
                 kinematics=None, agent_size=0.075):
        EzPickle.__init__(
            self,
            scenario=scenario,
            render_mode=render_mode,
            max_steps=max_steps,
            kinematics=kinematics,
            agent_size=agent_size,
        )
        super().__init__()
        
        # Create scenario and world using core abstractions
        self.scenario = scenario if scenario is not None else Scenario()
        self.world = self.scenario.make_world()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.agent_size = agent_size
        
        # Set default kinematics if not provided
        default_kinematics = kinematics if kinematics is not None else UnicycleKinematics()
        
        # Ensure all agents have kinematics
        for agent in self.world.agents:
            if agent.kinematics is None:
                agent.kinematics = default_kinematics
        
        # PettingZoo attributes
        self.possible_agents = [agent.name for agent in self.world.agents]
        self.agents = []
        
        # Spaces (based on first agent's kinematics model)
        first_agent = self.world.agents[0]
        action_low, action_high = first_agent.kinematics.get_action_bounds()
        self._action_space_template = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Observation: own state + landmarks + other agents
        state_dim = first_agent.kinematics.state_dim
        obs_dim = state_dim + 2 * len(self.world.landmarks) + state_dim * (len(self.world.agents) - 1)
        self._observation_space_template = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # State
        self._rng = np.random.RandomState()
        self.t = 0
        
        # Rendering
        self.viewer = None
        self._agent_geoms = []
        self._agent_transforms = []
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.action_spaces = {a: self._action_space_template for a in self.agents}
        self.observation_spaces = {a: self._observation_space_template for a in self.agents}
        
        # Reset world using core abstractions
        self.scenario.reset_world(self.world, self._rng)
        self.t = 0
        
        obs = self._get_obs()
        info = {a: {} for a in self.agents}
        return obs, info
    
    def step(self, actions):
        # Set actions for each agent
        for agent_obj in self.world.agents:
            if agent_obj.name in actions:
                agent_obj.action.u = actions[agent_obj.name]
        
        # Step the world (applies kinematics for all agents)
        self.world.step()
        
        # Update timesteps
        self.t += 1
        self.world.t = self.t  # Sync world timestep for reward computation
        
        # Compute rewards using scenario
        rewards = {}
        for agent_obj in self.world.agents:
            rewards[agent_obj.name] = self.scenario.reward(agent_obj, self.world)
        
        # Check termination
        terminated = {a: False for a in self.agents}
        truncated = {a: self.t >= self.max_steps for a in self.agents}
        
        # Clear agents if episode done
        if all(truncated.values()):
            self.agents = []
        
        obs = self._get_obs()
        info = {a: {} for a in self.agents if a in self.agents}
        
        return obs, rewards, terminated, truncated, info
    
    def _get_obs(self):
        obs = {}
        for agent_obj in self.world.agents:
            if agent_obj.name in self.agents:
                obs[agent_obj.name] = self.scenario.observation(agent_obj, self.world)
        return obs
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.viewer is None:
            self.viewer = Viewer(700, 700)
            self._setup_rendering()
        
        # Auto-zoom to fit all entities (similar to MPE2)
        all_pos = []
        for agent in self.world.agents:
            all_pos.append(agent.state.p_pos)
        for landmark in self.world.landmarks:
            all_pos.append(landmark.state.p_pos)
        
        if all_pos:
            all_pos = np.array(all_pos)
            min_x, min_y = all_pos.min(axis=0)
            max_x, max_y = all_pos.max(axis=0)
            
            # Add padding
            margin = 1.0
            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin
            
            # Ensure aspect ratio
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            w, h = max_x - min_x, max_y - min_y
            size = max(w, h) / 2
            
            self.viewer.set_bounds(cx - size, cx + size, cy - size, cy + size)
        
        # Update agent transforms
        for i, agent in enumerate(self.world.agents):
            pos = agent.state.p_pos
            theta = agent.state.p_orient
            self._agent_transforms[i].set_translation(pos[0], pos[1])
            self._agent_transforms[i].set_rotation(theta)
        
        # Draw landmarks
        for landmark in self.world.landmarks:
            circ = make_circle(landmark.size)
            circ.set_color(*landmark.color)
            transform = Transform()
            transform.set_translation(landmark.state.p_pos[0], landmark.state.p_pos[1])
            circ.add_attr(transform)
            self.viewer.add_onetime(circ)
        
        return self.viewer.render(return_rgb_array=(self.render_mode == "rgb_array"))
    
    def _setup_rendering(self):
        """Create persistent agent geometries using core rendering utilities."""
        self._agent_geoms = []
        self._agent_transforms = []
        self._agent_circles = []
        
        for agent in self.world.agents:
            # Use shared agent geometry creator from rendering utilities
            collision_circle, arrow, transform = make_agent_geom(
                agent.size, agent.color
            )
            
            # Add geometries to viewer (circle first, then arrow - draw order matters)
            self.viewer.add_geom(collision_circle)
            self.viewer.add_geom(arrow)
            
            # Store references for updates
            self._agent_circles.append(collision_circle)
            self._agent_geoms.append(arrow)
            self._agent_transforms.append(transform)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


env = make_env(raw_env)
parallel_env = raw_env


class Scenario(BaseScenario):
    """Simple Assignment scenario using core NHMRS abstractions with modular rewards."""
    
    def __init__(self, reward_mode='simple'):
        """Initialize scenario with reward configuration.
        
        Args:
            reward_mode: 'simple', 'balanced', or 'patrol'
                - 'simple': Assignment + collision only (fast learning)
                - 'balanced': Full reward with all components (default weights)
                - 'patrol': Optimized for N < M persistent coverage
        """
        self.reward_mode = reward_mode
        self.reward_computer = None
    
    def make_world(self, n_agents=3, n_landmarks=4):
        """Create world with agents and landmarks."""
        world = World()
        
        # Create agents
        agent_colors = [
            (0.25, 0.50, 0.85),  # blue
            (0.85, 0.25, 0.25),  # red
            (0.25, 0.85, 0.25),  # green
            (0.85, 0.85, 0.25),  # yellow
            (0.85, 0.25, 0.85),  # magenta
            (0.25, 0.85, 0.85),  # cyan
        ]
        
        for i in range(n_agents):
            agent = Agent()
            agent.name = f'agent_{i}'
            agent.size = 0.075
            agent.color = agent_colors[i % len(agent_colors)]
            agent.kinematics = None  # Will be set in environment init
            world.agents.append(agent)
        
        # Create landmarks (tasks)
        for i in range(n_landmarks):
            landmark = Landmark()
            landmark.name = f'landmark_{i}'
            landmark.size = 0.25
            landmark.color = (0.9, 0.1, 0.1)
            world.landmarks.append(landmark)
        
        # Initialize reward computer based on mode
        if self.reward_mode == 'simple':
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 1.0,
                    'coverage': 0.0,
                    'collision': 5.0,
                    'idleness': 0.0,
                    'efficiency': 0.0,
                },
                safety_radius=0.3
            )
        elif self.reward_mode == 'patrol':
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 0.5,
                    'coverage': 1.0,
                    'collision': 5.0,
                    'idleness': 1.0,
                    'efficiency': 0.1,
                },
                safety_radius=0.3,
                visit_threshold=0.5
            )
        else:  # 'balanced'
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 1.0,
                    'coverage': 0.5,
                    'collision': 5.0,
                    'idleness': 0.3,
                    'efficiency': 0.1,
                },
                safety_radius=0.3,
                visit_threshold=0.5
            )
        
        return world
    
    def reset_world(self, world, np_random):
        """Initialize positions and orientations."""
        # Reset reward computer state
        if self.reward_computer is not None:
            self.reward_computer.reset()
        
        # Initialize agent positions randomly
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, 2).astype(np.float32)
            agent.state.p_orient = np_random.uniform(-np.pi, np.pi)
        
        # Place landmarks in a circle
        for i, landmark in enumerate(world.landmarks):
            angle = 2 * np.pi * i / len(world.landmarks)
            radius = 3.0
            landmark.state.p_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ], dtype=np.float32)
    
    def reward(self, agent, world):
        """Compute reward using modular reward computer."""
        if self.reward_computer is None:
            # Fallback to simple distance-based reward
            dists = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) 
                     for lm in world.landmarks]
            return -min(dists)
        
        # Update timestep in reward computer
        if hasattr(world, 't'):
            self.reward_computer.update_timestep(world.t)
        
        # Use modular reward computation
        return self.reward_computer.compute_reward(agent, world)
    
    def observation(self, agent, world):
        """Own state + landmark positions + other agent states."""
        # Own state: [x, y, theta]
        own_state = np.array([
            agent.state.p_pos[0],
            agent.state.p_pos[1],
            agent.state.p_orient
        ], dtype=np.float32)
        
        # Landmark positions
        landmark_pos = np.concatenate([lm.state.p_pos for lm in world.landmarks])
        
        # Other agent states
        other_states = []
        for other in world.agents:
            if other is not agent:
                other_states.extend([
                    other.state.p_pos[0],
                    other.state.p_pos[1],
                    other.state.p_orient
                ])
        other_states = np.array(other_states, dtype=np.float32)
        
        return np.concatenate([own_state, landmark_pos, other_states])
