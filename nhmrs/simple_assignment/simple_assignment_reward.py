"""Reward computation for simple assignment (task allocation) scenarios.

This module implements reward functions for MARL task allocation environments
where N agents must visit M landmarks.

Available reward classes:
    - SimpleAssignmentReward: Modular multi-component reward with assignment,
      coverage, collision, idleness, and efficiency components. Handles N < M,
      N = M, and N > M regimes.
    - SimpleSpreadReward: MPE2-inspired reward focusing on spreading agents to
      cover all landmarks while avoiding collisions. Simple and effective for
      cooperative coverage tasks.

Components in SimpleAssignmentReward:
    1. Assignment: Negative distance to assigned landmark (Hungarian algorithm)
    2. Coverage: Global measure of how well all landmarks are covered
    3. Collision: Soft penalty when agents are within safety_radius
    4. Idleness: Urgency-weighted reward for visiting neglected landmarks (N < M)
    5. Efficiency: Penalty for moving when already at target

SimpleSpreadReward (MPE2-style):
    1. Occupancy: Reward for agents being close to landmarks (one agent per landmark)
    2. Collision: Penalty for agent-agent proximity
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from nhmrs._nhmrs_utils.reward import BaseReward


class SimpleAssignmentReward(BaseReward):
    """Multi-component reward function for task allocation scenarios.
    
    Computes a weighted sum of five components per agent:
        r_total = w_assign * r_assign + w_cover * r_cover + w_coll * r_coll +
                  w_idle * r_idle + w_eff * r_eff
    
    The reward adapts to different N/M regimes:
        - Assignment and collision are always active
        - Coverage provides global coordination signal
        - Idleness only applies when N < M (patrol scenarios)
        - Efficiency discourages motion when agent is at target
    
    Attributes:
        weights (dict): Component weights with keys:
            'assignment', 'coverage', 'collision', 'idleness', 'efficiency'
        safety_radius (float): Distance threshold for collision penalty
        visit_threshold (float): Distance threshold to consider landmark "visited"
        landmark_last_visit (dict): Maps landmark.name -> timestep of last visit
        current_timestep (int): Current episode timestep
    """
    
    def __init__(self, weights=None, safety_radius=0.3, visit_threshold=0.5):
        """Initialize reward with configurable weights and thresholds.
        
        Args:
            weights: Dictionary of component weights. If None, uses balanced defaults:
                {'assignment': 1.0, 'coverage': 0.5, 'collision': 5.0,
                 'idleness': 0.3, 'efficiency': 0.1}
            safety_radius: Collision avoidance radius in world units (default 0.3)
            visit_threshold: Distance to consider landmark visited (default 0.5)
        """
        self.weights = weights or {
            'assignment': 1.0,
            'coverage': 0.5,
            'collision': 5.0,
            'idleness': 0.3,
            'efficiency': 0.1,
        }
        self.safety_radius = safety_radius
        self.visit_threshold = visit_threshold
        self.landmark_last_visit = {}
        self.current_timestep = 0

    def reset(self):
        """Reset state tracking at episode start."""
        self.landmark_last_visit = {}
        self.current_timestep = 0

    def update_timestep(self, t: int):
        """Update current timestep for idleness tracking."""
        self.current_timestep = t

    def compute_reward(self, agent, world) -> float:
        """Compute total weighted reward for a single agent.
        
        Steps:
            1) Update landmark visit tracking
            2) Compute each component independently
            3) Return weighted sum
        
        All components except coverage are agent-specific. Coverage is a global
        measure but broadcast to all agents for cooperative learning.
        
        Args:
            agent: Agent object with state.p_pos and action.u
            world: World object with agents and landmarks lists
            
        Returns:
            float: Total reward (typically negative, minimizing distance/penalty)
        """
        self._update_landmark_visits(world)
        
        r_assignment = self.assignment_component(agent, world)
        r_coverage = self.coverage_component(world)
        r_collision = self.collision_penalty(agent, world)
        r_idleness = self.idleness_component(agent, world)
        r_efficiency = self.efficiency_bonus(agent, world)
        
        total = (
            self.weights['assignment'] * r_assignment +
            self.weights['coverage'] * r_coverage +
            self.weights['collision'] * r_collision +
            self.weights['idleness'] * r_idleness +
            self.weights['efficiency'] * r_efficiency
        )
        return total

    def assignment_component(self, agent, world) -> float:
        """Negative distance to agent's assigned landmark.
        
        Assignment algorithm:
            - N <= M: Solve optimal assignment via Hungarian algorithm
                      (scipy.optimize.linear_sum_assignment on distance matrix)
            - N > M: Greedy assignment - each landmark claims its nearest agent
                     Unassigned agents get distance to nearest landmark
        
        This component drives agents to reach their targets while avoiding
        conflicts with other agents competing for the same landmarks.
        
        Args:
            agent: Agent to compute reward for
            world: World with agents and landmarks
            
        Returns:
            float: -distance to assigned landmark (higher is better/closer)
        """
        n_agents = len(world.agents)
        n_landmarks = len(world.landmarks)
        
        # Build distance matrix
        D = np.zeros((n_agents, n_landmarks))
        for i, ag in enumerate(world.agents):
            for j, lm in enumerate(world.landmarks):
                D[i, j] = np.linalg.norm(ag.state.p_pos - lm.state.p_pos)
        
        agent_idx = world.agents.index(agent)
        
        if n_agents <= n_landmarks:
            # Standard assignment
            row_ind, col_ind = linear_sum_assignment(D)
            assigned_landmark_idx = col_ind[agent_idx]
            agent_distance = D[agent_idx, assigned_landmark_idx]
        else:
            # N > M: More agents than landmarks
            assigned_landmark_idx = None
            agent_distance = float('inf')
            
            for lm_idx, landmark in enumerate(world.landmarks):
                distances_to_lm = D[:, lm_idx]
                closest_agent_idx = np.argmin(distances_to_lm)
                
                if closest_agent_idx == agent_idx:
                    assigned_landmark_idx = lm_idx
                    agent_distance = distances_to_lm[closest_agent_idx]
                    break
            
            if assigned_landmark_idx is None:
                agent_distance = D[agent_idx, :].min()
        
        return -agent_distance

    def coverage_component(self, world) -> float:
        """Global coverage quality metric (broadcast to all agents).
        
        Measures how well the team covers all landmarks:
            - N < M: Returns -max(min_dist_to_each_landmark)
                     (worst-case coverage, encourages spreading out)
            - N >= M: Returns -mean(min_dist_to_each_landmark)
                      (average coverage, all landmarks should be reached)
        
        This component is identical for all agents, providing a cooperative
        team-level signal that encourages load balancing.
        
        Args:
            world: World with agents and landmarks
            
        Returns:
            float: Negative coverage metric (higher is better/closer)
        """
        n_agents = len(world.agents)
        n_landmarks = len(world.landmarks)
        
        if n_agents < n_landmarks:
            landmark_coverages = []
            for landmark in world.landmarks:
                min_dist = min(
                    np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                    for agent in world.agents
                )
                landmark_coverages.append(min_dist)
            worst_coverage = max(landmark_coverages)
            return -worst_coverage
        else:
            total_coverage = 0
            for landmark in world.landmarks:
                min_dist = min(
                    np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                    for agent in world.agents
                )
                total_coverage += min_dist
            avg_coverage = total_coverage / n_landmarks
            return -avg_coverage

    def collision_penalty(self, agent, world) -> float:
        """Soft distance-based collision penalty between agents.
        
        For each other agent within safety_radius, adds a penalty proportional
        to how close they are:
            penalty += (safety_radius - distance) / safety_radius
        
        This creates a smooth gradient (unlike hard contact) for RL training.
        No agent-landmark collision is considered.
        
        Args:
            agent: Agent to compute penalty for
            world: World with all agents
            
        Returns:
            float: -penalty (0 if no agents within safety_radius, negative otherwise)
        """
        penalty = 0.0
        for other in world.agents:
            if other is agent:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            if dist < self.safety_radius:
                penalty += (self.safety_radius - dist) / self.safety_radius
        return -penalty

    def idleness_component(self, agent, world) -> float:
        """Urgency-weighted reward for visiting neglected landmarks.
        
        Only active when N < M (patrol scenarios). Tracks how long since each
        landmark was last visited and weights distances by urgency:
            urgency_factor = 1.0 + (timesteps_since_visit) / 100.0
        
        Returns the maximum urgency-weighted reward across all landmarks,
        encouraging agents to visit the most neglected targets.
        
        Returns 0.0 when N >= M (all landmarks can be simultaneously covered).
        
        Args:
            agent: Agent to compute reward for
            world: World with agents and landmarks
            
        Returns:
            float: Max of -distance * urgency_factor over all landmarks, or 0.0
        """
        n_agents = len(world.agents)
        n_landmarks = len(world.landmarks)
        
        if n_agents >= n_landmarks:
            return 0.0
        
        urgencies = {}
        for landmark in world.landmarks:
            last_visit = self.landmark_last_visit.get(landmark.name, 0)
            idleness = self.current_timestep - last_visit
            urgencies[landmark.name] = idleness
        
        agent_rewards = []
        for landmark in world.landmarks:
            dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            urgency = urgencies[landmark.name]
            urgency_factor = 1.0 + urgency / 100.0
            agent_rewards.append(-dist * urgency_factor)
        
        return max(agent_rewards)

    def efficiency_bonus(self, agent, world) -> float:
        """Penalty for moving when already at target.
        
        If the agent is within visit_threshold of the nearest landmark,
        penalize action magnitude (||u||) to encourage stopping/hovering.
        Otherwise, return 0.0 (movement is fine when far from targets).
        
        This helps prevent oscillation around targets and reduces energy waste.
        
        Args:
            agent: Agent with action.u (kinematic controls)
            world: World with landmarks
            
        Returns:
            float: -||action.u|| if near target, else 0.0
        """
        min_dist_to_landmark = min(
            np.linalg.norm(agent.state.p_pos - lm.state.p_pos)
            for lm in world.landmarks
        )
        
        if hasattr(agent.action, 'u') and agent.action.u is not None:
            action_magnitude = np.linalg.norm(agent.action.u)
        else:
            action_magnitude = 0.0
        
        if min_dist_to_landmark < self.visit_threshold:
            return -action_magnitude
        else:
            return 0.0

    def _update_landmark_visits(self, world):
        """Update visit timestamps for landmarks within visit_threshold.
        
        For each landmark, if any agent is within visit_threshold, record
        the current timestep as the last visit time. Used by idleness_component.
        
        Args:
            world: World with agents and landmarks
        """
        for landmark in world.landmarks:
            min_dist = min(
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for agent in world.agents
            )
            if min_dist < self.visit_threshold:
                self.landmark_last_visit[landmark.name] = self.current_timestep


class SimpleSpreadReward(BaseReward):
    """MPE2-inspired reward for spreading agents across landmarks.
    
    This reward encourages agents to cooperatively spread out and occupy all
    landmarks while avoiding collisions with each other. It is simpler than
    SimpleAssignmentReward and well-suited for cooperative coverage tasks.
    
    Inspired by MPE2's simple_spread_v3 environment, this reward:
        - Encourages each agent to get close to at least one landmark
        - Penalizes agent-agent collisions
        - Is symmetric (same reward structure for all agents)
        - Does not use explicit assignment algorithms
    
    The total reward per agent is:
        r_total = r_occupancy + r_collision
    
    Where:
        - r_occupancy: -min(distance to all landmarks) for this agent
        - r_collision: -sum(collision penalties with other agents)
    
    Attributes:
        collision_weight (float): Weight for collision penalty component
        agent_radius (float): Collision radius for agents (default 0.15)
    """
    
    def __init__(self, collision_weight=1.0, agent_radius=0.15):
        """Initialize spread reward with collision parameters.
        
        Args:
            collision_weight: Multiplier for collision penalty (default 1.0)
            agent_radius: Radius for agent collision detection (default 0.15)
        """
        self.collision_weight = collision_weight
        self.agent_radius = agent_radius

    def reset(self):
        """Reset state tracking (no state needed for this reward)."""
        pass

    def update_timestep(self, t: int):
        """Update timestep (not used by this reward)."""
        pass

    def compute_reward(self, agent, world) -> float:
        """Compute total reward for a single agent.
        
        The reward consists of two components:
            1) Occupancy: Negative distance to nearest landmark (encourages
               agents to get close to landmarks)
            2) Collision: Negative penalty for being too close to other agents
        
        This formulation naturally encourages spreading behavior: agents want
        to minimize their distance to landmarks while maintaining separation
        from other agents.
        
        Args:
            agent: Agent object with state.p_pos
            world: World object with agents and landmarks lists
            
        Returns:
            float: Total reward (typically negative, higher is better)
        """
        # Occupancy component: negative distance to nearest landmark
        r_occupancy = self._occupancy_reward(agent, world)
        
        # Collision component: penalty for being near other agents
        r_collision = self._collision_penalty(agent, world)
        
        total = r_occupancy + self.collision_weight * r_collision
        return total

    def _occupancy_reward(self, agent, world) -> float:
        """Reward for being close to landmarks.
        
        Returns the negative distance to the nearest landmark. This encourages
        each agent to move toward and occupy at least one landmark. Combined
        with collision avoidance, this naturally produces spreading behavior.
        
        Args:
            agent: Agent to compute reward for
            world: World with landmarks
            
        Returns:
            float: -min_distance_to_landmarks (higher is better/closer)
        """
        if len(world.landmarks) == 0:
            return 0.0
        
        distances = [
            np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            for landmark in world.landmarks
        ]
        min_distance = min(distances)
        return -min_distance

    def _collision_penalty(self, agent, world) -> float:
        """Penalty for proximity to other agents.
        
        For each pair of agents within 2*agent_radius distance, applies a
        penalty proportional to how close they are. This creates a smooth
        gradient for collision avoidance learning.
        
        The penalty formula when dist < 2*agent_radius:
            penalty += 1 - (dist / (2 * agent_radius))
        
        This gives:
            - penalty = 1.0 when agents are at the same position
            - penalty = 0.0 when dist >= 2*agent_radius
        
        Args:
            agent: Agent to compute penalty for
            world: World with all agents
            
        Returns:
            float: -total_penalty (0 if no collisions, negative otherwise)
        """
        penalty = 0.0
        collision_threshold = 2.0 * self.agent_radius
        
        for other in world.agents:
            if other is agent:
                continue
            
            dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            if dist < collision_threshold:
                # Linear penalty that increases as agents get closer
                penalty += 1.0 - (dist / collision_threshold)
        
        return -penalty