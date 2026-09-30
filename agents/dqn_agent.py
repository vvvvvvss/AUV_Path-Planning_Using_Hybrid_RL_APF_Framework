"""
dqn_agent.py  —  Step 3a
=========================
Deep Q-Network agent for AUV path planning (discrete action space).

Algorithm: DQN  (Mnih et al., 2015)
  • Double DQN       — reduces Q-value overestimation bias
  • Replay buffer    — breaks temporal correlation between samples
  • Target network   — stabilises Q-target via periodic hard update

Network architecture  (MlpPolicy)
-----------------------------------
  Input(16) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(15)
                                                          ↑ Q-value/action

Hyperparameter rationale
------------------------
  learning_rate=1e-4       DQN is off-policy; smaller LR than PPO prevents
                           Q-target instability from replay variance
  buffer_size=100_000      ~500 episodes of history — enough for diverse
                           coverage of 3D env without excessive RAM
  learning_starts=5_000    fill buffer before first gradient step;
                           avoids exploding gradients on correlated samples
  batch_size=128           larger than SB3 default (64) — smoother gradients
                           in a 16-D state space
  gamma=0.99               high discount — goal may be 100+ steps away
  train_freq=4             1 gradient step per 4 env steps; balances
                           sample efficiency vs wall-clock time
  target_update_interval=1000  too frequent → unstable; too rare → slow
  exploration_fraction=0.3     ε: 1.0 → 0.05 over 30% of training budget
  exploration_final_eps=0.05   5% random actions forever after decay —
                               prevents premature convergence to local policy
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.auv_environment import AUVEnvironmentDiscrete
from agents.training_callbacks import (
    AUVMetricsCallback, EarlyStopCallback, BestModelCallback,
)

# ─────────────────────────────────── Paper-spec defaults ──────────────────────────────────────────────────────
DQN_DEFAULTS = dict(
    policy                 = "MlpPolicy",
    learning_rate          = 1e-4,
    buffer_size            = 100_000,
    learning_starts        = 5_000,
    batch_size             = 128,
    gamma                  = 0.99,
    train_freq             = 4,
    target_update_interval = 1_000,
    exploration_fraction   = 0.30,
    exploration_final_eps  = 0.05,
    policy_kwargs          = dict(net_arch=[256, 256]),
    verbose                = 0,
)


def make_dqn_env(seed: int = 0, n_envs: int = 1) -> VecMonitor:
    """
    Vectorised + monitored environment factory.

    VecMonitor injects per-episode stats (reward, length, reached_goal,
    energy_used, collided) into the info dict that our callback reads.
    DQN uses n_envs=1 — off-policy replay doesn't benefit from parallelism
    the way on-policy PPO does.
    """
    def _make_fn(rank: int):
        def _init():
            return AUVEnvironmentDiscrete(seed=seed + rank)
        return _init

    vec = DummyVecEnv([_make_fn(i) for i in range(n_envs)])
    return VecMonitor(vec)


def build_dqn(
    env,
    seed: int = 0,
    device: str = "auto",
    **overrides,
) -> DQN:
    """
    Construct a DQN model with paper-spec hyperparameters.

    Parameters
    ----------
    env      : VecEnv with Discrete(15) action space
    seed     : RNG seed for reproducibility
    device   : 'auto' → CUDA if available, else CPU
    overrides: any key in DQN_DEFAULTS to override for ablations
    """
    kwargs = {**DQN_DEFAULTS, **overrides}
    model  = DQN(
        env             = env,
        seed            = seed,
        device          = device,
        tensorboard_log = "./results/tb_logs/dqn",
        **kwargs,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  DQN built  |  params={n_params:,}  device={model.device}  "
          f"arch={kwargs['policy_kwargs']['net_arch']}")
    return model


def train_dqn(
    total_timesteps : int   = 500_000,
    seed            : int   = 0,
    save_dir        : str   = "./results/models",
    log_freq        : int   = 50,
    target_success  : float = 0.88,
    device          : str   = "auto",
    verbose         : int   = 1,
) -> tuple[DQN, AUVMetricsCallback, dict]:
    """
    Full DQN training run.

    Returns
    -------
    model      : trained DQN (best checkpoint loaded automatically)
    metrics_cb : callback with full training history for plotting
    summary    : paper Table-1 metrics dict
    """
    print("\n" + "=" * 60)
    print("  DQN TRAINING")
    print(f"  timesteps={total_timesteps:,}  seed={seed}  device={device}")
    print("=" * 60)

    env   = make_dqn_env(seed=seed)
    model = build_dqn(env, seed=seed, device=device)

    # ── Callbacks ────────────────────────────────────────────────────────
    metrics_cb = AUVMetricsCallback(log_freq=log_freq, verbose=verbose)
    early_stop = EarlyStopCallback(
        target_success = target_success,
        patience       = 4,
        check_freq     = 3_000,
        metrics_cb     = metrics_cb,
        verbose        = verbose,
    )
    saver = BestModelCallback(
        save_path  = save_dir,
        model_name = "dqn_auv",
        metrics_cb = metrics_cb,
        check_freq = 5_000,
        verbose    = verbose,
    )

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = CallbackList([metrics_cb, early_stop, saver]),
        reset_num_timesteps = True,
        tb_log_name         = "dqn_run",
        progress_bar        = False,
    )

    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "dqn_auv_final")
    model.save(final_path)
    if verbose:
        print(f"\n  Saved → {final_path}.zip")

    summary = metrics_cb.get_results_dict()
    summary["converged_episode"]  = getattr(early_stop, "converged_at_ep",  None)
    summary["converged_timestep"] = getattr(early_stop, "converged_at_ts",  None)
    summary["method"] = "Pure DQN"
    if verbose:
        _print_summary(summary)

    env.close()
    return model, metrics_cb, summary


def evaluate_dqn(
    model     : DQN,
    n_episodes: int  = 100,
    seed      : int  = 1000,
    verbose   : bool = False,
) -> dict:
    """
    Deterministic evaluation — ε-greedy is disabled (no exploration).
    Returns paper Table-1 metrics over n_episodes.
    """
    successes, energies, collisions, steps_all = [], [], [], []

    for ep in range(n_episodes):
        env = AUVEnvironmentDiscrete(seed=seed + ep)
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        successes.append(int(info["reached_goal"]))
        energies.append(env.episode_energy / 1000.0)
        collisions.append(int(env.episode_collided))
        steps_all.append(info["step"])
        env.close()

        if verbose:
            print(f"  ep {ep:3d}: {'✓' if info['reached_goal'] else '✗'} "
                  f"E={env.episode_energy/1000:.3f}kJ "
                  f"steps={info['step']} col={env.episode_collided}")

    return {
        "method"        : "Pure DQN",
        "success_rate"  : float(np.mean(successes)),
        "mean_energy_kj": float(np.mean(energies)),
        "collision_rate": float(np.mean(collisions)),
        "mean_steps"    : float(np.mean(steps_all)),
        "n_episodes"    : n_episodes,
    }


def load_dqn(path: str, device: str = "auto") -> DQN:
    """Load a saved DQN. Appends .zip if missing."""
    if not path.endswith(".zip"):
        path += ".zip"
    model = DQN.load(path, device=device)
    print(f"  Loaded DQN ← {path}")
    return model


def _print_summary(s: dict) -> None:
    print("\n  ── DQN Summary " + "─" * 40)
    for k, v in s.items():
        line = f"  {k:<24}: {v:.4f}" if isinstance(v, float) else \
               f"  {k:<24}: {v}"
        print(line)
    print("  " + "─" * 54)