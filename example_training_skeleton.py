"""MADDPG-style training skeleton for NHMRS simple_assignment environment.

Goal
====
Provide a very explicit, implementation-focused scaffold that can be
followed to build a working agent on top of the NHMRS
``simple_assignment_v0`` PettingZoo ParallelEnv. This file prioritizes
structure, contracts, shapes, and flow over neural network details.

What you will learn here
========================
1) How to instantiate and interact with the NHMRS ``simple_assignment_v0`` env
     using the PettingZoo ParallelEnv API (headless for training).
2) How to structure a MADDPG training loop:
     - Per-agent actors (decentralized execution)
     - Centralized critics (conditioned on all agents' observations and actions)
     - A shared replay buffer that stores joint transitions
     - Target networks and soft updates (where to add them)
3) Exactly where to plug in deep learning code (left as TODOs) and what the
     expected tensor shapes/concats should look like.

Important: This file intentionally contains NO deep learning framework imports
or model code so it runs without extra dependencies. Actions are chosen
randomly via the environment's action spaces to demonstrate the control flow.

Quickstart
==========
Run this file headlessly to see the scaffold execute end-to-end:

        python example_training_skeleton.py

PettingZoo ParallelEnv contract (applied by simple_assignment)
=============================================================
- reset(seed) -> (obs, info):
    - obs: Dict[str, np.ndarray], keyed by agent names like "agent_0".
    - info: Dict[str, dict], auxiliary data (empty here).
- step(actions) -> (next_obs, rewards, terminated, truncated, info):
    - actions: Dict[str, np.ndarray], one continuous action vector per agent.
    - rewards: Dict[str, float]
    - terminated: Dict[str, bool] (always False here; no early termination)
    - truncated: Dict[str, bool] (True at episode limit max_steps)
    - info: Dict[str, dict]

simple_assignment specifics (what you need to know)
==================================================
- Agent names: env.agents (current active) and env.possible_agents (all)
- Observation (per agent):
    [own_x, own_y, own_theta, lm1_x, lm1_y, ..., lmM_x, lmM_y,
     other1_x, other1_y, other1_theta, ..., otherN_x, otherN_y, otherN_theta]
    Size = 3 + 2*M + 3*(N-1)
- Action (per agent): continuous Box from the agent's kinematics, e.g. unicycle
    uses [v, omega]. Use env.action_space(agent).low/high to clip.
- Episodes end by truncation at "max_steps". No early termination on collision.
- Rendering is optional; keep training headless (render_mode=None) for speed.

MADDPG at a glance (how to wire your code)
==========================================
- Actors: one deterministic policy per agent: a_i = pi_i(o_i)
- Centralized critics: one critic per agent: Q_i(O, A) where
    O = [o_1 || o_2 || ... || o_N] and A = [a_1 || a_2 || ... || a_N]
- ReplayBuffer stores joint transitions T = (O, A, R, O', D) in a dict form.
- Targets: target_actor_i and target_critic_i for stability.
- Updates per agent i:
    1) Critic target: y = r_i + gamma * Q_i'(O', A') * (1 - done_i)
                where A' = [a'_1 .. a'_N], a'_j = target_actor_j(o'_j)
    2) Critic loss: L_Q = MSE(Q_i(O, A), y)
    3) Actor loss:  L_pi = -mean(Q_i(O, [a_1 .. pi_i(o_i) .. a_N]))
                with other agents' actions detached from graph
    4) Soft update: target <- (1 - tau) * target + tau * online

Where to plug your code
=======================
- MADDPGAgent.__init__: define actor/critic/targets/optimizers (framework of choice)
- MADDPGAgent.select_action: forward obs through actor + exploration noise
- MADDPGAgent.update: implement the MADDPG losses and backprop
- MADDPGAgent.soft_update: implement target parameter updates

Tip: Shapes and concatenation
-----------------------------
Assuming homogeneous agents, each obs has dim Do and action has dim Da.
For batch size B and number of agents N:
    - O_batch shape: (B, N*Do)
    - A_batch shape: (B, N*Da)
    - Q_i output shape: (B, 1)
Build O and A by concatenating per-agent tensors along the last dimension in
the fixed agent order env.possible_agents.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np

from nhmrs import simple_assignment_v0


# =============================================================================
# Replay Buffer (joint transitions)
# =============================================================================

@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    actions: Dict[str, np.ndarray]
    rewards: Dict[str, float]
    next_obs: Dict[str, np.ndarray]
    dones: Dict[str, bool]


class ReplayBuffer:
    """Simple list-based buffer storing joint transitions for all agents.

    For production, implement a ring buffer with pre-allocation and fast
    vectorized sampling. Here we keep it simple and framework-agnostic.
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, tr: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(tr)
        else:
            self.storage[self.idx] = tr
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.storage, batch_size)


# =============================================================================
# MADDPG Agents (placeholders for your NN models and optimizers)
# =============================================================================

class MADDPGAgent:
    """One agent in MADDPG with its own actor and a centralized critic.

    Replace the placeholders with your DL framework code (e.g., PyTorch):
      - actor:        pi_i(o_i) -> a_i
      - critic:       Q_i(o_1..N, a_1..N)
      - target_actor, target_critic for stable Q-learning
    """

    def __init__(self, name: str, obs_dim: int, action_dim: int):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # TODO: Initialize actor, critic, target networks, and optimizers
        # self.actor = ...
        # self.critic = ...
        # self.target_actor = ...
        # self.target_critic = ...

    def select_action(self, obs: np.ndarray, noise_scale: float = 0.1,
                       action_space=None) -> np.ndarray:
        """Select action for this agent given its own observation.

        For now, returns a random valid action so the scaffold runs headless.
        Replace with: a = actor(obs) + exploration_noise
        """
        if action_space is not None:
            action = action_space.sample()
        else:
            action = np.zeros(self.action_dim, dtype=np.float32)

        # Optional: add Gaussian/OU noise here if you have a deterministic actor
        # action = action + noise_scale * np.random.randn(self.action_dim)
        return action

    def update(self, batch: List[Transition], all_agents: Dict[str, "MADDPGAgent"],
               gamma: float, tau: float) -> Dict[str, float]:
        """One gradient update step for this agent.

        Pseudo-code (PyTorch-like), not executed here:
            1) Build centralized critic inputs by concatenating all obs/actions
               in the sampled batch: O = [o_1..o_N], A = [a_1..a_N]
            2) Critic target:
                 a'_j = target_actor_j(o'_j) for all agents j
                 y = r_i + gamma * target_critic_i(O', A') * (1 - done_i)
            3) Critic loss:
                 L_Q = MSE(critic_i(O, A), y)
            4) Actor loss (policy gradient via deterministic policy gradient):
                 a_i = actor_i(o_i)
                 A_pi = [a_1..a_i..a_N]  # with other agents' a_j detached
                 L_pi = -critic_i(O, A_pi).mean()
            5) Soft update targets:
                 soft_update(target, online, tau)

        Returns a dict of scalars to log.

       Example batch assembly (framework-agnostic):
          # batch: List[Transition] of length B
          agent_order = list(all_agents.keys())  # fixed order

          # Stack per-agent obs/actions into joint arrays
          O  = []   # (B, N*Do)
          O2 = []   # (B, N*Do) for next_obs
          A  = []   # (B, N*Da)
          R_i = []  # (B, 1)
          D_i = []  # (B, 1)

          for tr in batch:
             O.append(np.concatenate([tr.obs[a]      for a in agent_order], axis=-1))
             O2.append(np.concatenate([tr.next_obs[a] for a in agent_order], axis=-1))
             A.append(np.concatenate([tr.actions[a]  for a in agent_order], axis=-1))
             R_i.append([tr.rewards[self.name]])
             D_i.append([float(tr.dones[self.name])])

          # Convert O, O2, A, R_i, D_i to tensors in your framework
          # Compute A' with target actors and form target y
          # Backprop L_Q and L_pi, then soft-update targets
        """
        # TODO: Implement with your DL framework
        return {"loss/critic": np.nan, "loss/actor": np.nan}

    def soft_update(self, tau: float) -> None:
        """Soft-update target networks: target = (1-tau)*target + tau*online.

        Implement for your DL framework; here it's a placeholder.
        """
        # TODO: Implement soft update
        pass


class MADDPG:
    """Coordinator for all agents: action selection, storage, updates."""

    def __init__(self, agent_names: List[str], obs_dims: Dict[str, int],
                 action_dims: Dict[str, int]):
        self.agents: Dict[str, MADDPGAgent] = {
            name: MADDPGAgent(name, obs_dims[name], action_dims[name])
            for name in agent_names
        }
        self.buffer = ReplayBuffer(capacity=200_000)

    def select_actions(self, obs: Dict[str, np.ndarray], noise_scale: float,
                        action_spaces: Dict[str, Any]) -> Dict[str, np.ndarray]:
        actions = {}
        for name, agent in self.agents.items():
            actions[name] = agent.select_action(obs[name], noise_scale,
                                                action_space=action_spaces[name])
        return actions

    def store(self, tr: Transition) -> None:
        self.buffer.add(tr)

    def can_update(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def update(self, batch_size: int, gamma: float, tau: float) -> Dict[str, Dict[str, float]]:
        batch = self.buffer.sample(batch_size)
        logs: Dict[str, Dict[str, float]] = {}
        for name, agent in self.agents.items():
            logs[name] = agent.update(batch, self.agents, gamma, tau)
            agent.soft_update(tau)
        return logs


# =============================================================================
# Training loop (headless)
# =============================================================================

def main():
    # ----------------------
    # Hyperparameters (edit)
    # ----------------------
    seed = 42
    n_episodes = 50
    max_steps = 500
    warmup_steps = 5_000          # collect random experience before updates
    batch_size = 256
    gamma = 0.95
    tau = 0.01
    update_every = 100            # environment steps per update round
    updates_per_round = 1         # how many gradient updates per round
    exploration_noise = 0.1       # used with deterministic actors

    # ----------------------
    # Environment (headless)
    # ----------------------
    env = simple_assignment_v0.env(
        render_mode=None,  # headless for training
        max_steps=max_steps,
    )
    # Reset returns dicts keyed by agent names
    obs, info = env.reset(seed=seed)

    agent_names = list(env.agents)
    action_spaces = {a: env.action_space(a) for a in agent_names}
    obs_dims = {a: env.observation_space(a).shape[0] for a in agent_names}
    action_dims = {a: env.action_space(a).shape[0] for a in agent_names}

    maddpg = MADDPG(agent_names, obs_dims, action_dims)

    # Logging helpers
    episode_returns = {a: 0.0 for a in agent_names}
    global_step = 0

    print("\nStarting MADDPG-style training (skeleton)...")
    print("=" * 60)

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            for a in agent_names:
                episode_returns[a] = 0.0

            for t in range(max_steps):
                global_step += 1

                # 1) Action selection
                #    obs: Dict[str, np.ndarray], per-agent observation
                #    actions must be: Dict[str, np.ndarray] matching action spaces
                actions = maddpg.select_actions(
                    obs,
                    noise_scale=exploration_noise,
                    action_spaces=action_spaces,
                )

                # 2) Env step
                next_obs, rewards, terminated, truncated, info = env.step(actions)

                # 3) Store joint transition in replay buffer
                #    We store the full dicts to later construct centralized
                #    (O, A) inputs for each critic during updates.
                maddpg.store(Transition(obs=obs,
                                        actions=actions,
                                        rewards=rewards,
                                        next_obs=next_obs,
                                        dones={k: terminated[k] or truncated[k] for k in agent_names}))

                # 4) Book-keeping
                for a in agent_names:
                    episode_returns[a] += rewards[a]
                obs = next_obs

                # 5) Updates (after warmup and at a set frequency)
                #    After warmup_steps of random collection, periodically sample
                #    batches from ReplayBuffer and update each agent's actor/critic.
                if global_step > warmup_steps and global_step % update_every == 0:
                    for _ in range(updates_per_round):
                        if maddpg.can_update(batch_size):
                            logs = maddpg.update(batch_size, gamma, tau)
                            # TODO: write logs to tensorboard or console

                # 6) Episode termination
                if all(terminated.values()) or all(truncated.values()):
                    break

            # Episode summary
            avg_ret = float(np.mean([episode_returns[a] for a in agent_names]))
            print(f"Episode {ep:04d} | Steps: {t+1:03d} | AvgReturn: {avg_ret:8.3f}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("\nTraining skeleton finished.")


if __name__ == "__main__":
    main()


# =============================================================================
# Where to plug in your DL code (PyTorch/TF)
# =============================================================================
# - In MADDPGAgent.__init__: define actor, critic, target_actor, target_critic,
#   and their optimizers. Use obs_dim and action_dim to size networks.
# - In MADDPGAgent.select_action: forward obs through actor, add exploration
#   noise, and clip to env.action_space bounds.
# - In MADDPGAgent.update: implement the MADDPG update for this agent using
#   batches sampled from the shared ReplayBuffer. Key points:
#     * Build centralized inputs by concatenating all agents' obs/actions.
#     * Compute target actions with target actors and next observations.
#     * Critic target: r_i + gamma * Q_i'(O', A')
#     * Critic loss: MSE(Q_i(O, A), target)
#     * Actor loss: maximize Q_i(O, [a_1..pi_i(o_i)..a_N])
#     * Soft-update targets with tau.
# - Use a proper prioritized or uniform replay buffer and consider
#   normalization of observations/actions.
# - Add logging (TensorBoard) and checkpointing.
