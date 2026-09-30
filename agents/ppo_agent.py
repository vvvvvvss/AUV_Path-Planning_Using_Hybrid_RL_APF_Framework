"""
ppo_agent.py  —  Step 3b
=========================
Proximal Policy Optimisation agent for AUV path planning (continuous actions).

Why PPO over DQN for continuous control?
-----------------------------------------
DQN hand-discretises actions into 15 bins — coarse heading control.
PPO outputs a Gaussian N(mu, sigma) over (speed, yaw_rate, pitch_rate),
giving infinite-precision corrections. This is why PPO reaches 82% success
vs DQN's ~78% in the same environment.

Actor-Critic architecture  (MlpPolicy, Tanh activations)
----------------------------------------------------------
  Shared:  Input(16) -> Linear(256) -> Tanh -> Linear(256) -> Tanh
  Actor:   -> Linear(3) + learned log_std   => Gaussian policy
  Critic:  -> Linear(1)                     => V(s) baseline

  Tanh (not ReLU): bounds pre-activations naturally, avoids dying-ReLU
  when action distributions explore extreme values early in training.

PPO update cycle (every n_steps * n_envs transitions):
  1. Collect n_steps=2048 transitions per env  (= 8192 total with 4 envs)
  2. GAE advantage: A_t = sum_k (gamma*lambda)^k * delta_{t+k}
     where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
  3. n_epochs=10 passes over mini-batches of size batch_size=256:
       L = L_CLIP - ent_coef*H[pi] + vf_coef*(V - V_target)^2
       L_CLIP = E[ min(r*A, clip(r, 1-eps, 1+eps)*A) ]
       r = pi(a|s) / pi_old(a|s)
"""

from __future__ import annotations
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from envs.auv_environment import AUVEnvironment
from agents.training_callbacks import (
    AUVMetricsCallback, EarlyStopCallback, BestModelCallback,
)

PPO_DEFAULTS = dict(
    policy              = "MlpPolicy",
    learning_rate       = 3e-4,
    n_steps             = 2048,
    batch_size          = 256,
    n_epochs            = 10,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    clip_range          = 0.2,
    ent_coef            = 0.01,
    vf_coef             = 0.5,
    max_grad_norm       = 0.5,
    normalize_advantage = True,
    policy_kwargs       = dict(
        net_arch      = [256, 256],
        activation_fn = torch.nn.Tanh,
        log_std_init  = -1.0,   # initial sigma~0.37 — moderate exploration
        ortho_init    = True,   # orthogonal weight init (empirically better)
    ),
    verbose             = 0,
)


def make_ppo_env(seed: int = 0, n_envs: int = 4, mode: str = "continuous") -> VecMonitor:
    """
    Vectorised + monitored PPO environment.

    PPO is on-policy: more parallel envs = richer rollout diversity.
    n_envs=4 is a sweet spot for this env size (diminishing returns above 8).
    SubprocVecEnv = separate OS processes, true parallelism, bypasses GIL.
    DummyVecEnv  = same process, simpler debugging (used when n_envs=1).
    """
    def _make_fn(rank: int):
        def _init():
            set_random_seed(seed + rank)
            return AUVEnvironment(mode=mode, seed=seed + rank)
        return _init

    VecCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return VecMonitor(VecCls([_make_fn(i) for i in range(n_envs)]))


def build_ppo(env, seed: int = 0, device: str = "auto", **overrides) -> PPO:
    """Construct PPO with paper-spec hyperparameters."""
    kwargs = {**PPO_DEFAULTS, **overrides}
    model  = PPO(
        env=env, seed=seed, device=device,
        tensorboard_log="./results/tb_logs/ppo",
        **kwargs,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  PPO built  |  params={n_params:,}  device={model.device}  "
          f"arch={kwargs['policy_kwargs']['net_arch']}")
    return model


def train_ppo(
    total_timesteps : int   = 500_000,
    n_envs          : int   = 4,
    seed            : int   = 0,
    save_dir        : str   = "./results/models",
    log_freq        : int   = 50,
    target_success  : float = 0.90,
    device          : str   = "auto",
    verbose         : int   = 1,
    env_mode        : str   = "continuous",
    model_name      : str   = "ppo_auv",
    extra_callbacks : list  = None,
    ppo_overrides   : dict  = None,
) -> tuple[PPO, AUVMetricsCallback, dict]:
    """
    Full PPO training run.

    Parameters
    ----------
    extra_callbacks : additional SB3 callbacks injected by hybrid_fusion
    ppo_overrides   : hyperparameter overrides for ablation studies
    """
    print("\n" + "=" * 60)
    print(f"  PPO TRAINING  [{model_name}]")
    print(f"  timesteps={total_timesteps:,}  n_envs={n_envs}  seed={seed}")
    print("=" * 60)

    env   = make_ppo_env(seed=seed, n_envs=n_envs, mode=env_mode)
    model = build_ppo(env, seed=seed, device=device, **(ppo_overrides or {}))

    metrics_cb = AUVMetricsCallback(log_freq=log_freq, verbose=verbose)
    early_stop = EarlyStopCallback(
        target_success=target_success, patience=5,
        check_freq=4_000, metrics_cb=metrics_cb, verbose=verbose,
    )
    saver = BestModelCallback(
        save_path=save_dir, model_name=model_name,
        metrics_cb=metrics_cb, check_freq=8_000, verbose=verbose,
    )
    cb_list = [metrics_cb, early_stop, saver] + (extra_callbacks or [])

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(cb_list),
        reset_num_timesteps=True,
        tb_log_name=model_name,
        progress_bar=False,
    )

    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_path)
    if verbose:
        print(f"\n  Saved -> {final_path}.zip")

    summary = metrics_cb.get_results_dict()
    summary["converged_episode"]  = getattr(early_stop, "converged_at_ep",  None)
    summary["converged_timestep"] = getattr(early_stop, "converged_at_ts",  None)
    summary["method"] = model_name
    if verbose:
        _print_summary(summary)

    env.close()
    return model, metrics_cb, summary


def evaluate_ppo(
    model     : PPO,
    n_episodes: int  = 100,
    seed      : int  = 1000,
    method    : str  = "Pure PPO",
    verbose   : bool = False,
) -> dict:
    """
    Deterministic evaluation — uses policy mean mu (no sigma noise).
    deterministic=True is the correct eval mode for paper reporting.
    """
    successes, energies, collisions, steps_all = [], [], [], []

    for ep in range(n_episodes):
        env = AUVEnvironment(mode="continuous", seed=seed + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        successes.append(int(info["reached_goal"]))
        energies.append(env.episode_energy / 1000.0)
        collisions.append(int(env.episode_collided))
        steps_all.append(info["step"])
        env.close()

        if verbose:
            print(f"  ep {ep:3d}: {'✓' if info['reached_goal'] else 'x'} "
                  f"E={env.episode_energy/1000:.3f}kJ steps={info['step']}")

    return {
        "method"        : method,
        "success_rate"  : float(np.mean(successes)),
        "mean_energy_kj": float(np.mean(energies)),
        "collision_rate": float(np.mean(collisions)),
        "mean_steps"    : float(np.mean(steps_all)),
        "n_episodes"    : n_episodes,
    }


def load_ppo(path: str, device: str = "auto") -> PPO:
    """Load a saved PPO. Appends .zip if missing."""
    if not path.endswith(".zip"):
        path += ".zip"
    model = PPO.load(path, device=device)
    print(f"  Loaded PPO <- {path}")
    return model


def _print_summary(s: dict) -> None:
    print("\n  -- PPO Summary " + "-" * 40)
    for k, v in s.items():
        print(f"  {k:<24}: {v:.4f}" if isinstance(v, float)
              else f"  {k:<24}: {v}")
    print("  " + "-" * 54)