"""
hybrid_fusion.py  —  Step 4
============================
Adaptive RL-APF hybrid fusion layer — the core novelty of the paper.

Core idea
---------
Neither pure RL nor pure APF is optimal alone:
  • APF  : reactive, safe, but stuck in local minima (~82% success, 1.45 kJ)
  • Pure RL: flexible, but slow to learn safety (~82% success, 1.82 kJ)
  • Hybrid: RL handles strategy, APF handles safety → 96% success, 1.28 kJ

The blended action at each step is:

    a_exec = α · a_RL  +  (1 − α) · a_APF

where α ∈ [0.3, 0.9] is distance-adaptive:

    α = 0.3   if  d_nearest < 15 m   (APF dominates — obstacle avoidance)
    α = 0.7   if  15 ≤ d < 30 m     (shared control)
    α = 0.9   if  d ≥ 30 m          (RL dominates  — energy efficiency)

Alignment bonus reward shaping
-------------------------------
To teach the RL agent to select actions that AGREE with APF's safe heading,
we add the alignment term from the paper's reward function:

    R_align = W_ALIGN · cos(Δyaw_RL_APF)
            = 5 · dot(û_RL, û_APF)

When the RL heading vector aligns with the APF gradient direction,
cos → 1 (+5 reward). When they oppose, cos → −1 (−5 penalty).
This shapes the RL policy to "work with" APF rather than fight it.

Classes
-------
  HybridFusionEnv          — gym.Wrapper for PPO (continuous actions)
  DiscreteHybridFusionEnv  — gym.Wrapper for DQN (discrete → blend → continuous)

Factories
---------
  make_hybrid_ppo_env()    — builds VecEnv for PPO-APF training
  make_hybrid_dqn_env()    — builds VecEnv for DQN-APF training
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from envs.auv_environment import (
    AUVEnvironment, AUVEnvironmentDiscrete,
    V_MIN, V_MAX, YAW_RATE_MAX, PITCH_RATE_MAX, W_ALIGN,
)
from planning.apf_module import APFPlanner


# ─── α schedule constants (paper spec) ───────────────────────────────────────
ALPHA_NEAR   = 0.3    # d < D_NEAR  → APF dominates (safety)
ALPHA_MID    = 0.7    # D_NEAR ≤ d < D_FAR → shared
ALPHA_FAR    = 0.9    # d ≥ D_FAR   → RL dominates (efficiency)
D_NEAR       = 15.0   # m  — near-obstacle threshold
D_FAR        = 30.0   # m  — open-water threshold


# ─── Alpha computation ────────────────────────────────────────────────────────
def compute_alpha(d_nearest: float, mode: str = "adaptive") -> float:
    """
    Compute blending coefficient α ∈ [0.3, 0.9].

    Parameters
    ----------
    d_nearest : float — surface distance to nearest obstacle [m]
    mode      : str
        'adaptive' — paper's three-level step function (default)
        'fixed'    — returns ALPHA_MID (used for DQN-APF fixed-α baseline)
        'smooth'   — sigmoid interpolation (no discontinuities, alternative)

    Returns
    -------
    α : float  — RL weight in blended action
    """
    if mode == "fixed":
        return ALPHA_MID                             # always 0.7 (≈ fixed α=0.5 if desired)

    if mode == "smooth":
        # Sigmoid: α(d) = 0.3 + 0.6 · σ(k·(d − midpoint))
        # Smooth transition centred at 22.5m (midpoint of [15, 30])
        k = 0.25
        sig = 1.0 / (1.0 + math.exp(-k * (d_nearest - 22.5)))
        return float(np.clip(0.3 + 0.6 * sig, ALPHA_NEAR, ALPHA_FAR))

    # Default: adaptive step function (paper spec)
    if d_nearest < D_NEAR:
        return ALPHA_NEAR
    elif d_nearest < D_FAR:
        return ALPHA_MID
    else:
        return ALPHA_FAR


def alpha_schedule_table() -> list[tuple[str, float, float]]:
    """Return the three α zones for display / paper tables."""
    return [
        ("Near obstacle",  f"d < {D_NEAR}m",           ALPHA_NEAR),
        ("Transition zone",f"{D_NEAR}m ≤ d < {D_FAR}m",ALPHA_MID),
        ("Open water",     f"d ≥ {D_FAR}m",            ALPHA_FAR),
    ]


# ─── Action-space conversion utilities ───────────────────────────────────────
def discrete_to_continuous(
    action_idx: int,
    discrete_table: list,
) -> np.ndarray:
    """
    Convert a discrete DQN action index to a normalised continuous vector.

    Mapping (inverse of AUVEnvironment._decode_action in continuous mode):
      speed_n    = 2·(speed − V_MIN)/(V_MAX − V_MIN) − 1    → [−1, 1]
      yaw_n      = yaw_rate / YAW_RATE_MAX                    → [−1, 1]
      pitch_n    = pitch_rate / PITCH_RATE_MAX                → [−1, 1]
    """
    speed, yaw_rate, pitch_rate = discrete_table[int(action_idx)]
    speed_n    = 2.0 * (speed      - V_MIN)        / (V_MAX        - V_MIN) - 1.0
    yaw_n      = float(yaw_rate)   / YAW_RATE_MAX
    pitch_n    = float(pitch_rate) / PITCH_RATE_MAX
    return np.array([speed_n, yaw_n, pitch_n], dtype=np.float32)


def continuous_to_heading_vector(
    action_cont: np.ndarray,
    current_yaw: float,
    current_pitch: float,
) -> np.ndarray:
    """
    Convert a continuous action [speed_n, yaw_n, pitch_n] to a unit
    heading vector in world frame.  Used for alignment bonus computation.

    The RL action specifies heading CHANGES (rates), so the intended
    heading = current_heading + delta_heading.
    """
    yaw_delta   = float(action_cont[1]) * YAW_RATE_MAX
    pitch_delta = float(action_cont[2]) * PITCH_RATE_MAX
    tgt_yaw     = _wrap(current_yaw   + yaw_delta)
    tgt_pitch   = float(np.clip(current_pitch + pitch_delta,
                                -math.pi/4, math.pi/4))
    return np.array([
        math.cos(tgt_yaw)  * math.cos(tgt_pitch),
        math.sin(tgt_yaw)  * math.cos(tgt_pitch),
        math.sin(tgt_pitch),
    ], dtype=np.float64)


def compute_alignment_bonus(
    rl_action_cont : np.ndarray,
    apf_heading_vec: np.ndarray,
    current_yaw    : float,
    current_pitch  : float,
) -> float:
    """
    Compute cos(Δyaw_RL_APF) — the alignment between RL heading intent
    and APF gradient direction.

    Returns
    -------
    cos_sim : float ∈ [−1, 1]
        +1  → RL perfectly agrees with APF  (+5 reward via W_ALIGN)
         0  → perpendicular
        −1  → RL opposes APF               (−5 reward)
    """
    rl_vec = continuous_to_heading_vector(rl_action_cont, current_yaw, current_pitch)
    rl_mag = np.linalg.norm(rl_vec)
    ap_mag = np.linalg.norm(apf_heading_vec)
    if rl_mag < 1e-9 or ap_mag < 1e-9:
        return 0.0
    cos_sim = float(np.dot(rl_vec / rl_mag, apf_heading_vec / ap_mag))
    return float(np.clip(cos_sim, -1.0, 1.0))


# ─── PPO Hybrid Fusion Environment ────────────────────────────────────────────
class HybridFusionEnv(gym.Wrapper):
    """
    Gymnasium wrapper for PPO-APF hybrid training.

    At every step:
      1. Receive continuous RL action a_RL ∈ [−1,1]³
      2. Compute APF action a_APF (continuous)
      3. Compute α from d_nearest (adaptive or fixed)
      4. Execute blended action: a_exec = α·a_RL + (1−α)·a_APF
      5. Add alignment bonus R_align = W_ALIGN·cos(Δyaw_RL_APF) to reward

    The RL agent is trained to produce actions that, when blended with APF,
    yield high cumulative reward.  This implicitly teaches the agent:
      • In open water (α≈0.9): pursue the goal efficiently
      • Near obstacles (α≈0.3): APF handles avoidance; RL can focus on
        angle alignment to stay on-course post-obstacle

    Parameters
    ----------
    env        : AUVEnvironment  (mode='continuous')
    apf        : APFPlanner
    alpha_mode : 'adaptive' | 'fixed' | 'smooth'
    fixed_alpha: float  — used only when alpha_mode='fixed'
    """

    def __init__(
        self,
        env        : AUVEnvironment,
        apf        : APFPlanner,
        alpha_mode : str   = "adaptive",
        fixed_alpha: float = ALPHA_MID,
    ):
        super().__init__(env)
        self.apf         = apf
        self.alpha_mode  = alpha_mode
        self.fixed_alpha = fixed_alpha

        # Episode-level fusion diagnostics
        self._alphas      : list[float] = []
        self._alignments  : list[float] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.apf.reset()
        self._alphas     = []
        self._alignments = []
        return obs, info

    def step(self, rl_action: np.ndarray):
        """
        Blend RL action with APF, execute, augment reward.
        """
        rl_action = np.asarray(rl_action, dtype=np.float32)

        # 1. APF recommended action in continuous [-1,1]³
        apf_action = self.apf.compute_action(self.env)

        # 2. Adaptive alpha
        d_nearest = self.env.get_nearest_obstacle_distance()
        alpha     = (compute_alpha(d_nearest, self.alpha_mode)
                     if self.alpha_mode != "fixed"
                     else self.fixed_alpha)

        # 3. Alignment bonus (computed BEFORE step, using pre-step state)
        apf_vec   = self.apf.get_apf_heading_vector(self.env)
        alignment = compute_alignment_bonus(
            rl_action, apf_vec,
            self.env.auv_yaw, self.env.auv_pitch,
        )

        # 4. Blend and clip
        blended = np.clip(alpha * rl_action + (1.0 - alpha) * apf_action, -1.0, 1.0)

        # 5. Execute blended action in underlying env
        obs, reward, terminated, truncated, info = self.env.step(blended)

        # 6. Add alignment bonus to reward
        reward += W_ALIGN * alignment

        # 7. Store diagnostics
        self._alphas.append(alpha)
        self._alignments.append(alignment)

        info["alpha"]       = alpha
        info["alignment"]   = alignment
        info["apf_action"]  = apf_action
        info["rl_action"]   = rl_action
        info["blended_act"] = blended
        info["d_nearest"]   = d_nearest

        return obs, float(reward), terminated, truncated, info

    # ── Diagnostics ───────────────────────────────────────────────────────
    def episode_alpha_stats(self) -> dict:
        if not self._alphas:
            return {}
        a = np.array(self._alphas)
        return {
            "alpha_mean" : float(a.mean()),
            "alpha_min"  : float(a.min()),
            "alpha_max"  : float(a.max()),
            "alpha_near_frac": float((a <= ALPHA_NEAR + 0.01).mean()),
            "alpha_far_frac" : float((a >= ALPHA_FAR  - 0.01).mean()),
        }

    def episode_alignment_stats(self) -> dict:
        if not self._alignments:
            return {}
        al = np.array(self._alignments)
        return {
            "align_mean" : float(al.mean()),
            "align_min"  : float(al.min()),
            "align_max"  : float(al.max()),
        }


# ─── DQN Hybrid Fusion Environment ───────────────────────────────────────────
class DiscreteHybridFusionEnv(gym.Wrapper):
    """
    Gymnasium wrapper for DQN-APF hybrid training.

    DQN outputs a discrete action index ∈ {0,…,14}.
    This wrapper:
      1. Converts the discrete index → continuous normalised vector
      2. Blends with APF (same α logic as HybridFusionEnv)
      3. Executes the blended continuous action by temporarily switching
         the underlying env's decode mode

    The DQN still sees Discrete(15) action space — no change to its
    network architecture or training loop.  Only the EXECUTED action
    changes via blending.

    Parameters
    ----------
    env        : AUVEnvironmentDiscrete  (mode='discrete')
    apf        : APFPlanner
    alpha_mode : 'fixed' (DQN-APF baseline uses fixed α=0.5)
    fixed_alpha: float  — default 0.5 per paper baseline spec
    """

    def __init__(
        self,
        env        : AUVEnvironmentDiscrete,
        apf        : APFPlanner,
        alpha_mode : str   = "fixed",
        fixed_alpha: float = 0.5,
    ):
        super().__init__(env)
        assert env.mode == "discrete", \
            "DiscreteHybridFusionEnv requires AUVEnvironmentDiscrete"
        self.apf         = apf
        self.alpha_mode  = alpha_mode
        self.fixed_alpha = fixed_alpha
        self._alphas     : list[float] = []
        self._alignments : list[float] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.apf.reset()
        self._alphas     = []
        self._alignments = []
        return obs, info

    def step(self, discrete_action: int):
        discrete_action = int(discrete_action)

        # 1. Convert discrete action → continuous representation
        rl_cont = discrete_to_continuous(
            discrete_action, self.env._discrete_table
        )

        # 2. APF continuous action
        apf_action = self.apf.compute_action(self.env)

        # 3. Alpha
        d_nearest = self.env.get_nearest_obstacle_distance()
        alpha     = (compute_alpha(d_nearest, self.alpha_mode)
                     if self.alpha_mode != "fixed"
                     else self.fixed_alpha)

        # 4. Alignment bonus
        apf_vec   = self.apf.get_apf_heading_vector(self.env)
        alignment = compute_alignment_bonus(
            rl_cont, apf_vec,
            self.env.auv_yaw, self.env.auv_pitch,
        )

        # 5. Blend
        blended = np.clip(
            alpha * rl_cont + (1.0 - alpha) * apf_action, -1.0, 1.0
        )

        # 6. Execute as continuous action:
        #    Temporarily switch mode, step, restore
        original_mode     = self.env.mode
        self.env.mode     = "continuous"
        obs, reward, terminated, truncated, info = self.env.step(blended)
        self.env.mode     = original_mode

        # 7. Alignment bonus
        reward += W_ALIGN * alignment

        self._alphas.append(alpha)
        self._alignments.append(alignment)

        info["alpha"]       = alpha
        info["alignment"]   = alignment
        info["d_nearest"]   = d_nearest

        return obs, float(reward), terminated, truncated, info


# ─── Environment factories ────────────────────────────────────────────────────
def make_hybrid_ppo_env(
    seed       : int   = 0,
    n_envs     : int   = 4,
    alpha_mode : str   = "adaptive",
    fixed_alpha: float = ALPHA_MID,
    apf_seed   : int   = 42,
) -> VecMonitor:
    """
    Vectorised + monitored PPO-APF environment.

    Each parallel env gets its own APFPlanner instance (required because
    each env has independent state — shared APF would cause race conditions).
    """
    def _make_fn(rank: int):
        def _init():
            set_random_seed(seed + rank)
            base_env = AUVEnvironment(mode="continuous", seed=seed + rank)
            apf      = APFPlanner(seed=apf_seed + rank)
            return HybridFusionEnv(
                base_env, apf,
                alpha_mode=alpha_mode,
                fixed_alpha=fixed_alpha,
            )
        return _init

    VecCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return VecMonitor(VecCls([_make_fn(i) for i in range(n_envs)]))


def make_hybrid_dqn_env(
    seed       : int   = 0,
    fixed_alpha: float = 0.5,
    apf_seed   : int   = 42,
) -> VecMonitor:
    """
    Monitored DQN-APF environment (single env — DQN is off-policy).
    Uses fixed α=0.5 per paper's DQN-APF baseline spec.
    """
    def _make_fn():
        def _init():
            base_env = AUVEnvironmentDiscrete(seed=seed)
            apf      = APFPlanner(seed=apf_seed)
            return DiscreteHybridFusionEnv(
                base_env, apf,
                alpha_mode="fixed",
                fixed_alpha=fixed_alpha,
            )
        return _init

    return VecMonitor(DummyVecEnv([_make_fn()]))


# ─── Standalone hybrid evaluation ────────────────────────────────────────────
def evaluate_hybrid(
    model,
    alpha_mode : str   = "adaptive",
    fixed_alpha: float = ALPHA_MID,
    n_episodes : int   = 100,
    seed       : int   = 1000,
    method     : str   = "PPO-APF",
    is_discrete: bool  = False,
    verbose    : bool  = False,
) -> dict:
    """
    Evaluate a trained RL model inside the hybrid fusion wrapper.

    Works for both PPO (continuous) and DQN (discrete) models.

    Parameters
    ----------
    model       : SB3 model (PPO or DQN)
    alpha_mode  : 'adaptive' | 'fixed' | 'smooth'
    is_discrete : True for DQN, False for PPO
    """
    successes, energies, collisions = [], [], []
    steps_all, alphas_all, aligns_all = [], [], []

    for ep in range(n_episodes):
        apf = APFPlanner(seed=seed + ep)
        if is_discrete:
            base = AUVEnvironmentDiscrete(seed=seed + ep)
            env  = DiscreteHybridFusionEnv(
                base, apf,
                alpha_mode=alpha_mode,
                fixed_alpha=fixed_alpha,
            )
        else:
            base = AUVEnvironment(mode="continuous", seed=seed + ep)
            env  = HybridFusionEnv(
                base, apf,
                alpha_mode=alpha_mode,
                fixed_alpha=fixed_alpha,
            )

        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        successes.append(int(info["reached_goal"]))
        energies.append(base.episode_energy / 1000.0)
        collisions.append(int(base.episode_collided))
        steps_all.append(info["step"])

        alpha_stats = env.episode_alpha_stats()
        alphas_all.append(alpha_stats.get("alpha_mean", fixed_alpha))
        align_stats = env.episode_alignment_stats()
        aligns_all.append(align_stats.get("align_mean", 0.0))

        if verbose:
            print(
                f"  ep {ep:3d}: {'✓' if info['reached_goal'] else '✗'}  "
                f"E={base.episode_energy/1000:.3f}kJ  "
                f"steps={info['step']:3d}  "
                f"alpha_mean={alpha_stats.get('alpha_mean',0):.2f}  "
                f"align={align_stats.get('align_mean',0):.2f}"
            )
        env.close()

    return {
        "method"        : method,
        "success_rate"  : float(np.mean(successes)),
        "mean_energy_kj": float(np.mean(energies)),
        "collision_rate": float(np.mean(collisions)),
        "mean_steps"    : float(np.mean(steps_all)),
        "mean_alpha"    : float(np.mean(alphas_all)),
        "mean_alignment": float(np.mean(aligns_all)),
        "n_episodes"    : n_episodes,
    }


# ─── Utility ─────────────────────────────────────────────────────────────────
def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi