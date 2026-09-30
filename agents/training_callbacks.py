"""
training_callbacks.py  —  Step 3 (shared)
==========================================
Custom Stable-Baselines3 callbacks used by both DQN and PPO agents.

Architecture
------------
  AUVMetricsCallback   — per-episode stats: success, energy, collisions
  EarlyStopCallback    — halt when target success rate is sustained
  CheckpointCallback   — save best model by success rate (not just reward)

Why custom callbacks?
---------------------
SB3's default logger tracks reward, which conflates goal-reaching with
energy efficiency. For the paper we need *separate* metrics:

  success_rate   — fraction of episodes where goal was reached
  mean_energy    — mean kJ consumed per episode
  collision_rate — fraction of episodes with at least one collision

These map directly to Table 1 in the paper.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class AUVMetricsCallback(BaseCallback):
    """
    Tracks AUV-specific per-episode metrics and logs them to SB3's logger.

    Metrics logged every `log_freq` episodes:
      auv/success_rate   ∈ [0, 1]
      auv/mean_energy_kj — mean energy (kJ) over the window
      auv/collision_rate ∈ [0, 1]
      auv/mean_steps     — mean episode length

    Parameters
    ----------
    log_freq   : int   — rolling window size AND logging frequency (episodes)
    verbose    : int   — 0 silent, 1 print to stdout
    """

    def __init__(self, log_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Rolling windows — deque auto-drops entries beyond maxlen
        self._successes  = deque(maxlen=log_freq)
        self._energies   = deque(maxlen=log_freq)
        self._collisions = deque(maxlen=log_freq)
        self._steps      = deque(maxlen=log_freq)

        # For tracking episode boundaries
        self._ep_count   = 0
        self._start_time = time.time()

        # Best success rate seen — used to trigger checkpointing
        self.best_success_rate = 0.0

        # Full history for post-training plotting
        self.history = {
            "episode"       : [],
            "success_rate"  : [],
            "mean_energy_kj": [],
            "collision_rate": [],
            "mean_steps"    : [],
            "timestep"      : [],
        }

    def _on_step(self) -> bool:
        """
        Called every environment step by SB3.
        We detect episode ends via the 'dones' signal and extract
        metrics from the environment's episode-level attributes.
        """
        # SB3 stores per-env info dicts in self.locals["infos"]
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", self.locals.get("done", [False]))

        # Handle both VecEnv (list) and single env (bool)
        if isinstance(dones, bool):
            dones = [dones]
            infos = [infos] if isinstance(infos, dict) else infos

        for done, info in zip(dones, infos):
            if not done:
                continue

            # SB3 wraps terminal info in "terminal_observation" dict
            # Episode-level stats sit in the regular info dict
            self._ep_count += 1

            # Extract metrics — info dict populated by our env's step()
            # "episode" key added by SB3's Monitor wrapper (if used)
            reached  = bool(info.get("reached_goal", False))
            energy   = float(info.get("energy_used", 0.0)) / 1000.0  # → kJ
            collided = bool(info.get("collided",      False))
            steps    = int(info.get("step",            0))

            # Fallback: read directly from env if info is sparse
            # (happens when env is not wrapped in Monitor)
            if "reached_goal" not in info:
                # Try to get from the underlying env
                env = self.training_env
                if hasattr(env, "envs"):          # VecEnv
                    env = env.envs[0]
                if hasattr(env, "episode_energy"):
                    energy   = env.episode_energy / 1000.0
                    collided = env.episode_collided

            self._successes.append(int(reached))
            self._energies.append(energy)
            self._collisions.append(int(collided))
            self._steps.append(steps)

            # Log every log_freq episodes
            if self._ep_count % self.log_freq == 0:
                self._log_metrics()

        return True   # returning False would stop training

    def _log_metrics(self) -> None:
        if len(self._successes) == 0:
            return

        sr  = float(np.mean(self._successes))
        e   = float(np.mean(self._energies))
        cr  = float(np.mean(self._collisions))
        st  = float(np.mean(self._steps))
        ep  = self._ep_count
        ts  = self.num_timesteps
        elapsed = time.time() - self._start_time

        # Push to SB3 logger (appears in TensorBoard / CSV)
        self.logger.record("auv/success_rate",    sr)
        self.logger.record("auv/mean_energy_kj",  e)
        self.logger.record("auv/collision_rate",  cr)
        self.logger.record("auv/mean_steps",      st)
        self.logger.record("auv/episode",         ep)
        self.logger.dump(self.num_timesteps)

        # Update best
        if sr > self.best_success_rate:
            self.best_success_rate = sr

        # Store for plotting
        self.history["episode"].append(ep)
        self.history["success_rate"].append(sr)
        self.history["mean_energy_kj"].append(e)
        self.history["collision_rate"].append(cr)
        self.history["mean_steps"].append(st)
        self.history["timestep"].append(ts)

        if self.verbose >= 1:
            print(
                f"  [ep {ep:5d} | step {ts:7d}] "
                f"success={sr*100:5.1f}%  "
                f"energy={e:.3f}kJ  "
                f"collision={cr*100:4.1f}%  "
                f"steps={st:5.1f}  "
                f"({elapsed:.0f}s elapsed)"
            )

    def get_results_dict(self) -> dict:
        """Return final-window metrics for the paper's results table."""
        if len(self._successes) == 0:
            return {}
        return {
            "success_rate"  : float(np.mean(self._successes)),
            "mean_energy_kj": float(np.mean(self._energies)),
            "collision_rate": float(np.mean(self._collisions)),
            "mean_steps"    : float(np.mean(self._steps)),
            "total_episodes": self._ep_count,
        }


class EarlyStopCallback(BaseCallback):
    """
    Stop training when success_rate ≥ target is sustained for
    `patience` consecutive evaluation windows.

    This reproduces the paper's "convergence episode" metric:
    we report the timestep at which stable performance was first achieved.

    Parameters
    ----------
    target_success : float — success rate threshold (e.g. 0.90)
    patience       : int   — how many consecutive windows must exceed target
    check_freq     : int   — check every this many timesteps
    metrics_cb     : AUVMetricsCallback — the sibling metrics callback
    """

    def __init__(
        self,
        target_success: float = 0.90,
        patience      : int   = 3,
        check_freq    : int   = 5000,
        metrics_cb    : Optional[AUVMetricsCallback] = None,
        verbose       : int   = 1,
    ):
        super().__init__(verbose)
        self.target_success  = target_success
        self.patience        = patience
        self.check_freq      = check_freq
        self.metrics_cb      = metrics_cb
        self._consec_hits    = 0
        self.converged_at_ep = None
        self.converged_at_ts = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        if self.metrics_cb is None or len(self.metrics_cb._successes) < 10:
            return True

        current_sr = float(np.mean(self.metrics_cb._successes))

        if current_sr >= self.target_success:
            self._consec_hits += 1
            if self._consec_hits >= self.patience:
                self.converged_at_ep = self.metrics_cb._ep_count
                self.converged_at_ts = self.num_timesteps
                if self.verbose >= 1:
                    print(
                        f"\n  ✓ EarlyStop: success={current_sr*100:.1f}% ≥ "
                        f"{self.target_success*100:.0f}% for {self.patience} "
                        f"windows. Converged at episode {self.converged_at_ep}."
                    )
                return False   # stop training
        else:
            self._consec_hits = 0

        return True


class BestModelCallback(BaseCallback):
    """
    Save the model whenever success_rate improves.
    Unlike SB3's built-in EvalCallback, we use the *training* window
    success rate (cheaper than running a full eval rollout every N steps).

    Parameters
    ----------
    save_path  : str — directory to save models
    model_name : str — filename prefix  (e.g. 'dqn_auv')
    metrics_cb : AUVMetricsCallback — sibling callback with success window
    check_freq : int — check every this many timesteps
    """

    def __init__(
        self,
        save_path  : str,
        model_name : str,
        metrics_cb : AUVMetricsCallback,
        check_freq : int = 5000,
        verbose    : int = 1,
    ):
        super().__init__(verbose)
        self.save_path    = save_path
        self.model_name   = model_name
        self.metrics_cb   = metrics_cb
        self.check_freq   = check_freq
        self._best_sr     = 0.0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        if len(self.metrics_cb._successes) < 10:
            return True

        current_sr = float(np.mean(self.metrics_cb._successes))
        if current_sr > self._best_sr + 0.01:   # +1% threshold to avoid noise
            self._best_sr = current_sr
            path = os.path.join(self.save_path, f"{self.model_name}_best")
            self.model.save(path)
            if self.verbose >= 1:
                print(f"  💾 New best model saved  "
                      f"(success={current_sr*100:.1f}%  →  {path}.zip)")
        return True