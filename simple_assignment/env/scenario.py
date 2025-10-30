"""Scenario: defines initial positions and setup for NHMRS task allocation environment.

This module provides the Scenario class which handles the initialization and
configuration of the multi-agent task allocation environment for Non-Holonomic
Mobile Robot Systems (NHMRS). It defines the spatial layout, agent spawn 
locations, and task landmark positions for the simple assignment problem.

Units:
    - Position (x, y): Abstract distance units (conceptually meters)
    - Orientation (theta): Radians, range [-π, π]
    - World scale: Approximately 6-8 units in diameter
    
Coordinate System:
    - Origin (0, 0) at center of world
    - Positive x-axis points right
    - Positive y-axis points up
    - Angles measured counter-clockwise from positive x-axis

Task Allocation:
    This scenario sets up a simple assignment problem where each agent must be
    assigned to exactly one task (landmark), and the goal is to minimize the
    total cost (e.g., travel distance or time) across all assignments.
"""
import numpy as np


class Scenario:
    """Base scenario for NHMRS task allocation environment.
    
    This class defines the initial configuration of the environment for the
    simple assignment problem, including:
    - Number of non-holonomic agents (unicycle model)
    - Number of task locations (landmarks)
    - Initial positions and orientations of agents
    - Fixed positions of task landmarks
    
    The scenario creates a setup where agents spawn in a cluster near the origin
    and task landmarks are evenly distributed in a circle around them, representing
    a balanced task allocation scenario.
    
    Attributes:
        None (stateless class, uses parameters passed to methods)
    
    Notes:
        - Designed for simple assignment: n_agents ≤ n_landmarks
        - Each agent should be assigned to exactly one task
        - Optimal assignment minimizes total travel cost
    """
    
    def make_world(self, n_agents=3, n_landmarks=4):
        """Create world configuration for task allocation scenario.
        
        This method defines the high-level structure of the NHMRS task allocation
        environment without initializing actual positions. It sets up the parameters
        that will be used during reset to create the assignment problem.
        
        Args:
            n_agents (int, optional): Number of non-holonomic agents (robots) in 
                the environment. Default is 3. Must be positive integer. For simple
                assignment, should be ≤ n_landmarks.
            n_landmarks (int, optional): Number of task locations to be assigned.
                Default is 4. Must be positive integer. Represents the number of
                tasks available for allocation.
        
        Returns:
            dict: World configuration containing:
                - 'n_agents' (int): Number of agents (robots)
                - 'n_landmarks' (int): Number of tasks (landmarks)
        
        Example:
            >>> scenario = Scenario()
            >>> world = scenario.make_world(n_agents=5, n_landmarks=5)
            >>> print(world)
            {'n_agents': 5, 'n_landmarks': 5}
        
        Notes:
            - For simple assignment: typically n_agents == n_landmarks
            - If n_agents < n_landmarks: some tasks remain unassigned
            - If n_agents > n_landmarks: problem becomes infeasible
        """
        return {
            'n_agents': n_agents,
            'n_landmarks': n_landmarks,
        }
    
    def reset_world(self, n_agents, n_landmarks, rng):
        """Initialize agent and task positions for a new assignment episode.
        
        This method generates random initial positions for all NHMRS agents and
        places task landmarks in a fixed circular pattern. Agents are spawned in
        a small cluster near the origin to create an initial unassigned state,
        while task locations are evenly distributed at a fixed radius to create
        clear assignment targets.
        
        Spatial Layout:
            - Agents: Randomly placed in [-1, 1] × [-1, 1] square (2×2 units)
            - Tasks: Evenly spaced on circle of radius 3.0 units
            - Total world span: Approximately 6 units diameter
        
        This layout ensures that:
            - All agents start at similar distances from tasks (fair initial state)
            - Task distribution is symmetric (no bias toward any agent)
            - Assignment problem has clear spatial structure
        
        Args:
            n_agents (int): Number of agents to initialize. Must match world config.
            n_landmarks (int): Number of task locations to place. Must match world config.
            rng (np.random.Generator): NumPy random number generator for reproducibility.
                Used to generate consistent random positions across episodes with
                same seed.
        
        Returns:
            tuple: A tuple containing:
                - agent_states (np.ndarray): Shape (n_agents, 3), dtype float32
                    Each row is [x, y, theta] where:
                    - x, y: Agent position in world coordinates (units)
                    - theta: Agent orientation in radians, range [-π, π]
                
                - landmark_positions (np.ndarray): Shape (n_landmarks, 2), dtype float32
                    Each row is [x, y] task location in world coordinates (units)
        
        Example:
            >>> import numpy as np
            >>> scenario = Scenario()
            >>> rng = np.random.default_rng(seed=42)
            >>> agents, tasks = scenario.reset_world(3, 4, rng)
            >>> print(f"Agent states shape: {agents.shape}")  # (3, 3)
            >>> print(f"Task positions shape: {tasks.shape}")  # (4, 2)
            >>> print(f"Agent 0 position: {agents[0, :2]}")  # Random in [-1, 1]
            >>> print(f"Task 0 position: {tasks[0]}")  # [3.0, 0.0]
        
        Notes:
            - Agent orientations are uniformly random over full circle
            - Task angles start at 0 (positive x-axis) and increase counter-clockwise
            - First task is always at (3.0, 0.0) if n_landmarks > 0
            - Uses fixed radius of 3.0 for all task locations
            - Tasks are stationary throughout the episode
        """
        # Initialize agent states array: position (x, y) + orientation (theta)
        agent_states = np.zeros((n_agents, 3), dtype=np.float32)
        
        # Spawn agents in a small cluster near origin (unassigned initial state)
        agent_states[:, 0] = rng.uniform(-1, 1, n_agents)  # x position in [-1, 1]
        agent_states[:, 1] = rng.uniform(-1, 1, n_agents)  # y position in [-1, 1]
        agent_states[:, 2] = rng.uniform(-np.pi, np.pi, n_agents)  # theta in [-π, π]
        
        # Initialize task locations array: only (x, y), tasks have no orientation
        landmark_positions = np.zeros((n_landmarks, 2), dtype=np.float32)
        
        # Distribute tasks evenly around a circle at fixed radius
        for i in range(n_landmarks):
            # Calculate evenly spaced angles: 0, 2π/n, 4π/n, ..., 2π(n-1)/n
            angle = 2 * np.pi * i / n_landmarks
            radius = 3.0  # Fixed distance from origin (units)
            
            # Convert polar to Cartesian coordinates for task location
            landmark_positions[i, 0] = radius * np.cos(angle)  # x = r cos(θ)
            landmark_positions[i, 1] = radius * np.sin(angle)  # y = r sin(θ)
        
        return agent_states, landmark_positions
