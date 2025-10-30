import numpy as np

from nhmrs.simple_assignment.simple_assignment import Scenario
from nhmrs.simple_assignment.simple_assignment_reward import SimpleAssignmentReward


def test_collision_penalty_within_radius():
    # Build a minimal world via Scenario
    scenario = Scenario(reward_mode='simple')
    world = scenario.make_world(n_agents=2, n_landmarks=1)

    # Place both agents at the same position (guaranteed collision)
    world.agents[0].state.p_pos = np.array([0.0, 0.0], dtype=np.float32)
    world.agents[1].state.p_pos = np.array([0.0, 0.0], dtype=np.float32)

    reward = SimpleAssignmentReward(safety_radius=0.3)

    # Penalty is negative when within safety radius
    p0 = reward.collision_penalty(world.agents[0], world)
    p1 = reward.collision_penalty(world.agents[1], world)

    assert p0 < 0.0
    assert p1 < 0.0


def test_collision_penalty_outside_radius():
    scenario = Scenario(reward_mode='simple')
    world = scenario.make_world(n_agents=2, n_landmarks=1)

    # Place agents far apart (no collision)
    world.agents[0].state.p_pos = np.array([0.0, 0.0], dtype=np.float32)
    world.agents[1].state.p_pos = np.array([1.0, 0.0], dtype=np.float32)  # > safety_radius

    reward = SimpleAssignmentReward(safety_radius=0.3)

    p0 = reward.collision_penalty(world.agents[0], world)
    p1 = reward.collision_penalty(world.agents[1], world)

    assert p0 == 0.0
    assert p1 == 0.0
