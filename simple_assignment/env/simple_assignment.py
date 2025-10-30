"""Multi-agent unicycle environment with PettingZoo ParallelEnv interface."""
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from scipy.optimize import linear_sum_assignment

from .rendering import Viewer, make_polygon, make_circle, Transform
from .scenario import Scenario


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "simple_assignment_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, scenario=None, render_mode=None, max_steps=500):
        super().__init__()
        self.scenario = scenario if scenario is not None else Scenario()
        self.world = self.scenario.make_world()
        self.n_agents = self.world['n_agents']
        self.n_landmarks = self.world['n_landmarks']
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Agent parameters
        self.dt = 0.1
        self.v_max = 2.0
        self.omega_max = np.pi
        self.agent_size = 0.075  # Size similar to MPE agents
        
        # PettingZoo attributes
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = []
        
        # Spaces (will be populated on reset)
        self._action_space_template = spaces.Box(
            low=np.array([-self.v_max, -self.omega_max], dtype=np.float32),
            high=np.array([self.v_max, self.omega_max], dtype=np.float32),
            dtype=np.float32
        )
        obs_dim = 3 + 2 * self.n_landmarks + 3 * (self.n_agents - 1)
        self._observation_space_template = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # State
        self._rng = np.random.RandomState()
        self.agent_states = None  # (N, 3): [x, y, theta]
        self.landmark_positions = None  # (M, 2): [x, y]
        self.t = 0
        
        # Rendering
        self.viewer = None
        self._agent_geoms = []
        self._agent_transforms = []
        self._agent_colors = [
            (0.25, 0.50, 0.85),  # blue
            (0.85, 0.25, 0.25),  # red
            (0.25, 0.85, 0.25),  # green
            (0.85, 0.85, 0.25),  # yellow
            (0.85, 0.25, 0.85),  # magenta
            (0.25, 0.85, 0.85),  # cyan
        ]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.action_spaces = {a: self._action_space_template for a in self.agents}
        self.observation_spaces = {a: self._observation_space_template for a in self.agents}
        
        self.agent_states, self.landmark_positions = self.scenario.reset_world(
            self.n_agents, self.n_landmarks, self._rng
        )
        self.t = 0
        
        obs = self._get_obs()
        info = {a: {} for a in self.agents}
        return obs, info
    
    def step(self, actions):
        # Apply unicycle dynamics
        for i, agent in enumerate(self.agents):
            v, omega = actions[agent]
            v = np.clip(v, -self.v_max, self.v_max)
            omega = np.clip(omega, -self.omega_max, self.omega_max)
            
            x, y, theta = self.agent_states[i]
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += omega * self.dt
            theta = np.arctan2(np.sin(theta), np.cos(theta))  # normalize to [-pi, pi]
            
            self.agent_states[i] = [x, y, theta]
        
        # Compute rewards (negative distance to nearest landmark)
        rewards = {}
        for i, agent in enumerate(self.agents):
            pos = self.agent_states[i, :2]
            dists = np.linalg.norm(self.landmark_positions - pos, axis=1)
            rewards[agent] = -np.min(dists)
        
        # Check termination
        self.t += 1
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
        for i, agent in enumerate(self.agents):
            # Own state
            own_state = self.agent_states[i].copy()
            # Landmark positions
            landmarks_flat = self.landmark_positions.flatten()
            # Other agents
            other_agents = np.delete(self.agent_states, i, axis=0).flatten()
            
            obs[agent] = np.concatenate([own_state, landmarks_flat, other_agents]).astype(np.float32)
        return obs
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.viewer is None:
            self.viewer = Viewer(700, 700)
            self._setup_rendering()
        
        # Auto-zoom to fit all entities (similar to MPE2)
        all_pos = []
        if self.agent_states is not None:
            all_pos.append(self.agent_states[:, :2])
        if self.landmark_positions is not None:
            all_pos.append(self.landmark_positions)
        
        if all_pos:
            all_pos = np.vstack(all_pos)
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
        for i in range(self.n_agents):
            x, y, theta = self.agent_states[i]
            self._agent_transforms[i].set_translation(x, y)
            self._agent_transforms[i].set_rotation(theta)
        
        # Draw landmarks
        for lm_pos in self.landmark_positions:
            circ = make_circle(self.agent_size * 0.5)
            circ.set_color(0.9, 0.1, 0.1)
            transform = Transform()
            transform.set_translation(lm_pos[0], lm_pos[1])
            circ.add_attr(transform)
            self.viewer.add_onetime(circ)
        
        return self.viewer.render(return_rgb_array=(self.render_mode == "rgb_array"))
    
    def _setup_rendering(self):
        """Create persistent agent geometries."""
        self._agent_geoms = []
        self._agent_transforms = []
        
        for i in range(self.n_agents):
            # Create arrowhead (pointing right, similar to MPE2)
            arrow_length = self.agent_size * 2.0
            arrow_width = self.agent_size
            vertices = [
                (arrow_length, 0),  # tip
                (-arrow_length * 0.5, arrow_width),  # back top
                (-arrow_length, 0),  # back center
                (-arrow_length * 0.5, -arrow_width),  # back bottom
            ]
            
            arrow = make_polygon(vertices)
            color = self._agent_colors[i % len(self._agent_colors)]
            arrow.set_color(*color)
            
            transform = Transform()
            arrow.add_attr(transform)
            
            self.viewer.add_geom(arrow)
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
