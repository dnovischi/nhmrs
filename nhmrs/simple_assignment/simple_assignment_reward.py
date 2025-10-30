"""Reward functions for Simple Assignment environment."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from nhmrs._nhmrs_utils.reward import BaseReward


class SimpleAssignmentReward(BaseReward):
    """Concrete reward for simple assignment scenario covering all use cases."""
    
    def __init__(self, weights=None, safety_radius=0.3, visit_threshold=0.5):
        """Initialize reward with configurable weights and thresholds.
        
        Args:
            weights: Dictionary of component weights (assignment, coverage, collision, idleness, efficiency)
            safety_radius: Collision avoidance radius
            visit_threshold: Distance threshold to consider landmark visited
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
        """Compute total weighted reward for an agent.
        
        Combines: assignment, coverage, collision, idleness, efficiency.
        Handles N < M, N = M, N > M cases.
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
        """Optimal assignment reward using Hungarian algorithm.
        
        Handles:
        - N <= M: Each agent assigned to unique landmark
        - N > M: Greedy assignment, surplus agents get nearest landmark
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
        """Global coverage quality (same for all agents).
        
        - N < M: Worst-case coverage (minimize max distance to any landmark)
        - N >= M: Average coverage
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
        """Soft collision penalty for smooth gradients."""
        penalty = 0.0
        for other in world.agents:
            if other is agent:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            if dist < self.safety_radius:
                penalty += (self.safety_radius - dist) / self.safety_radius
        return -penalty

    def idleness_component(self, agent, world) -> float:
        """Reward visiting unvisited landmarks (N < M patrol case)."""
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
        """Penalize unnecessary movement when near targets."""
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
        """Track landmark visits for idleness computation."""
        for landmark in world.landmarks:
            min_dist = min(
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for agent in world.agents
            )
            if min_dist < self.visit_threshold:
                self.landmark_last_visit[landmark.name] = self.current_timestep
