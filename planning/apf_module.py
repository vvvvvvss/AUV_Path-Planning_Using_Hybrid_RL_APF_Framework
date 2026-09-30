"""
apf_module.py  —  Step 2
=========================
Artificial Potential Field (APF) path planner for 3-D AUV navigation.

Theory
------
APF treats the environment as a scalar potential field U(q):

    U(q) = U_att(q) + U_rep(q)

The vehicle follows the negative gradient  F = −∇U(q),
which attracts it toward the goal and repels it from obstacles.

Attractive potential  (paraboloid / conic hybrid)
-------------------------------------------------
                 ½ k_att · d²          if d ≤ d_switch
  U_att(q) =
                 d_switch·k_att·d − ½k_att·d_switch²   otherwise

  Gradient:  F_att = −k_att · (q − q_goal)         (capped at d_switch)

Using a pure quadratic near the goal gives smooth, zero-velocity arrival.
Beyond d_switch we switch to a linear (conic) potential so the attractive
force is bounded and doesn't overpower repulsion in large environments.

Repulsive potential  (Khatib 1986 — standard formulation)
---------------------------------------------------------
                 ½ k_rep·(1/d − 1/d₀)²   if d ≤ d₀
  U_rep(q) =
                 0                         if d >  d₀

  Gradient:  F_rep = k_rep·(1/d − 1/d₀)·(1/d²)·(q − q_obs)/d
                   = k_rep·(1/d − 1/d₀)·(q̂_away)/d²

  where d  = surface-to-AUV distance,  d₀ = influence radius.

Singularity issue  (Local-minima fix)
--------------------------------------
Classical APF gets trapped in local minima when repulsive and attractive
forces cancel. We address this with:
  1. Random perturbation if the AUV is nearly stationary for >5 steps.
  2. Obstacle-tangent component (adds a lateral "slide" force).
  3. The RL agent overrides APF in deep local minima via fusion α.

Usage
-----
Standalone (APF-only baseline):
    planner = APFPlanner(env)
    obs, _ = env.reset()
    action  = planner.compute_action(env)   # returns continuous 3-vector

Hybrid (called by hybrid_fusion.py):
    force   = planner.compute_force(env)    # raw 3-D force vector
    heading = planner.force_to_heading(force, env)  # (yaw, pitch) in rad
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from envs.auv_environment import (
    AUVEnvironment,
    V_MIN, V_MAX, YAW_RATE_MAX, PITCH_RATE_MAX, DT,
)


# ─── APF hyperparameters ─────────────────────────────────────────────────────
K_ATT      = 2.0      # Attractive gain        — scales pull toward goal
K_REP      = 800.0    # Repulsive gain          — scales push from obstacles
D_SWITCH   = 30.0     # m — switch paraboloid→conic  (prevents runaway att.)
D0         = 20.0     # m — repulsive influence radius  (paper: 15–20 m)
D0_MIN     = 0.5      # m — clamp: avoid division by zero at obs surface
MAX_FORCE  = 50.0     # N — global force saturation   (numerical stability)

# Local-minima escape
STALL_STEPS     = 8    # consecutive near-zero-movement steps → trigger escape
STALL_THRESHOLD = 0.3  # m — displacement below this → "stalled"
PERTURB_MAG     = 5.0  # N — random perturbation force magnitude


class APFPlanner:
    """
    3-D Artificial Potential Field planner.

    Parameters
    ----------
    k_att     : Attractive force gain.
    k_rep     : Repulsive force gain.
    d0        : Repulsive influence radius [m].
    d_switch  : Distance at which attractive potential switches from
                quadratic to linear  [m].
    max_force : Saturation cap on total force magnitude [N].
    seed      : RNG seed for perturbation.
    """

    def __init__(
        self,
        k_att    : float = K_ATT,
        k_rep    : float = K_REP,
        d0       : float = D0,
        d_switch : float = D_SWITCH,
        max_force: float = MAX_FORCE,
        seed     : Optional[int] = None,
    ):
        self.k_att     = k_att
        self.k_rep     = k_rep
        self.d0        = d0
        self.d_switch  = d_switch
        self.max_force = max_force
        self._rng = np.random.default_rng(seed)

        # Local-minima detection
        self._stall_count  = 0
        self._prev_pos     = np.zeros(3)

    # ── Public API ────────────────────────────────────────────────────────

    def compute_force(self, env: AUVEnvironment) -> np.ndarray:
        """
        Compute the total APF force vector at the AUV's current position.

        Returns
        -------
        force : np.ndarray, shape (3,)
            Net force vector in world frame [N].
        """
        pos  = env.auv_pos
        goal = env.goal_pos

        f_att = self._attractive(pos, goal)
        f_rep = self._repulsive(pos, env)
        f_tan = self._tangential(pos, env)   # local-minima mitigation

        f_total = f_att + f_rep + f_tan

        # Detect stall and perturb
        f_total = self._maybe_perturb(f_total, pos)

        # Saturate
        mag = np.linalg.norm(f_total)
        if mag > self.max_force:
            f_total = f_total * (self.max_force / mag)

        return f_total

    def compute_action(self, env: AUVEnvironment) -> np.ndarray:
        """
        Full APF → action pipeline for standalone use as an env agent.

        Returns
        -------
        action : np.ndarray, shape (3,)  ∈ [-1, 1]
            Continuous action compatible with AUVEnvironment('continuous').
        """
        force = self.compute_force(env)
        return self.force_to_action(force, env)

    def force_to_action(
        self, force: np.ndarray, env: AUVEnvironment
    ) -> np.ndarray:
        """
        Convert a 3-D force vector to a normalised continuous action.

        Steps
        -----
        1. Extract desired heading from force direction (yaw, pitch).
        2. Compute heading error w.r.t. AUV's current heading.
        3. Scale errors into yaw_rate and pitch_rate ∈ [−1, 1].
        4. Set speed proportional to force magnitude (bounded).

        Returns
        -------
        action : np.ndarray [speed_n, yaw_rate_n, pitch_rate_n], all ∈ [-1,1]
        """
        tgt_yaw, tgt_pitch = self.force_to_heading(force)

        # Heading errors (wrapped to (−π, π])
        dyaw   = _wrap(tgt_yaw   - env.auv_yaw)
        dpitch = _wrap(tgt_pitch - env.auv_pitch)

        # Scale errors to [-1,1] rate commands
        yaw_rate_n   = float(np.clip(dyaw   / YAW_RATE_MAX,   -1.0, 1.0))
        pitch_rate_n = float(np.clip(dpitch / PITCH_RATE_MAX, -1.0, 1.0))

        # Speed: scale force magnitude to [V_MIN, V_MAX], then normalise
        fmag   = float(np.linalg.norm(force))
        speed  = float(np.clip(
            V_MIN + (fmag / self.max_force) * (V_MAX - V_MIN),
            V_MIN, V_MAX,
        ))
        speed_n = 2.0 * (speed - V_MIN) / (V_MAX - V_MIN) - 1.0  # → [-1,1]

        return np.array([speed_n, yaw_rate_n, pitch_rate_n], dtype=np.float32)

    @staticmethod
    def force_to_heading(force: np.ndarray) -> Tuple[float, float]:
        """
        Extract (yaw, pitch) from a 3-D force vector.

        yaw   = atan2(Fy, Fx)          — horizontal heading
        pitch = atan2(Fz, ‖F_xy‖)     — vertical inclination
        """
        fx, fy, fz = float(force[0]), float(force[1]), float(force[2])
        yaw   = math.atan2(fy, fx)
        fxy   = math.sqrt(fx**2 + fy**2)
        pitch = math.atan2(fz, fxy + 1e-9)
        return yaw, pitch

    def get_apf_heading_vector(self, env: AUVEnvironment) -> np.ndarray:
        """
        Unit vector in the APF-recommended heading direction.
        Used by hybrid_fusion.py to compute RL–APF alignment angle.
        """
        force = self.compute_force(env)
        mag   = np.linalg.norm(force)
        if mag < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return force / mag

    def reset(self) -> None:
        """Call at episode start to clear stall counter."""
        self._stall_count = 0
        self._prev_pos    = np.zeros(3)

    # ── Force components ──────────────────────────────────────────────────

    def _attractive(
        self, pos: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """
        Attractive force using paraboloid / conic hybrid.

        • d ≤ d_switch  →  F = −k_att · (pos − goal)          [quadratic zone]
        • d > d_switch  →  F = −k_att · d_switch · (pos−goal)/d [conic zone]

        The conic zone caps the attractive force at  k_att × d_switch,
        preventing it from completely dominating repulsion far from the goal.
        """
        diff = goal - pos                              # vector toward goal
        d    = float(np.linalg.norm(diff))
        if d < 1e-9:
            return np.zeros(3)

        if d <= self.d_switch:
            # Quadratic: force ∝ distance  (smooth arrival at goal)
            return self.k_att * diff
        else:
            # Conic: constant-magnitude attractive force
            return self.k_att * self.d_switch * (diff / d)

    def _repulsive(
        self, pos: np.ndarray, env: AUVEnvironment
    ) -> np.ndarray:
        """
        Repulsive force summed over all obstacles (Khatib 1986).

        F_rep_i = k_rep · (1/d_i − 1/d₀) · (1/d_i²) · (pos − obs_i)/d_i
                = k_rep · (1/d_i − 1/d₀) / d_i² · û_away_i

        Only active when d_i ≤ d₀.  Force grows as  1/d² → ∞  near surface,
        but we clamp d_i ≥ D0_MIN to prevent numerical explosion.
        """
        f_rep = np.zeros(3, dtype=np.float64)

        for obs in env.obstacles:
            # Direction from obstacle centre to AUV
            diff_vec = pos - obs.pos
            dist_ctr = float(np.linalg.norm(diff_vec))   # centre-to-centre
            d_surf   = max(dist_ctr - obs.radius, D0_MIN) # surface distance

            if d_surf > self.d0:
                continue   # outside influence radius — no repulsion

            # Unit vector pointing away from obstacle
            u_away = diff_vec / (dist_ctr + 1e-9)

            # Khatib formula
            coeff  = self.k_rep * (1.0 / d_surf - 1.0 / self.d0) / (d_surf ** 2)
            f_rep += coeff * u_away

        return f_rep

    def _tangential(
        self, pos: np.ndarray, env: AUVEnvironment
    ) -> np.ndarray:
        """
        Tangential (obstacle-circumnavigation) force.

        When the AUV is directly between the goal and an obstacle, the
        attractive and repulsive forces are co-linear → local minimum.
        Adding a lateral component perpendicular to the obstacle-goal axis
        breaks this symmetry and helps the vehicle slide around.

        F_tan = k_tan · (u_rep × u_goal) × u_rep
              = tangential component orthogonal to repulsive direction,
                pointing toward the goal half-space.

        This is a lightweight version of the 'tangent bug' idea.
        """
        K_TAN = 0.15 * self.k_rep

        f_tan = np.zeros(3, dtype=np.float64)
        u_goal = env.goal_pos - pos
        d_goal = np.linalg.norm(u_goal)
        if d_goal < 1e-9:
            return f_tan
        u_goal = u_goal / d_goal

        for obs in env.obstacles:
            diff_vec = pos - obs.pos
            dist_ctr = float(np.linalg.norm(diff_vec))
            d_surf   = max(dist_ctr - obs.radius, D0_MIN)
            if d_surf > self.d0:
                continue

            u_away  = diff_vec / (dist_ctr + 1e-9)
            # Component of goal-direction orthogonal to obstacle-away direction
            proj    = np.dot(u_goal, u_away) * u_away
            tangent = u_goal - proj
            t_mag   = np.linalg.norm(tangent)
            if t_mag > 1e-9:
                scale   = K_TAN * (1.0 / d_surf - 1.0 / self.d0) / (d_surf ** 2)
                f_tan  += scale * tangent / t_mag

        return f_tan

    # ── Local-minima escape ───────────────────────────────────────────────

    def _maybe_perturb(
        self, force: np.ndarray, pos: np.ndarray
    ) -> np.ndarray:
        """
        Detect stall (near-zero displacement) and add random perturbation.

        If the AUV hasn't moved > STALL_THRESHOLD in STALL_STEPS consecutive
        steps, we inject a random horizontal force to escape the local minimum.
        The RL agent naturally avoids this issue, but APF-standalone needs it.
        """
        displacement = np.linalg.norm(pos - self._prev_pos)
        self._prev_pos = pos.copy()

        if displacement < STALL_THRESHOLD:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= STALL_STEPS:
            perturb = self._rng.standard_normal(3)
            perturb[2] *= 0.3                             # mostly horizontal
            perturb = PERTURB_MAG * perturb / (np.linalg.norm(perturb) + 1e-9)
            self._stall_count = 0
            return force + perturb

        return force


# ─── Utility ─────────────────────────────────────────────────────────────────

def _wrap(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def run_apf_episode(
    env       : AUVEnvironment,
    planner   : APFPlanner,
    max_steps : int = 500,
    verbose   : bool = False,
) -> dict:
    """
    Run one complete APF-only episode.  Used for the APF baseline evaluation.

    Returns
    -------
    dict with keys: success, energy, collided, steps, trajectory
    """
    obs, _ = env.reset()
    planner.reset()

    trajectory = [env.auv_pos.copy()]
    total_reward = 0.0

    for step in range(max_steps):
        action   = planner.compute_action(env)
        obs, r, terminated, truncated, info = env.step(action)
        trajectory.append(env.auv_pos.copy())
        total_reward += r

        if verbose:
            print(f"  APF step {step:3d}: d_goal={info['d_goal']:6.2f}m  "
                  f"E={info['energy_used']:6.1f}J  "
                  f"speed={info['speed']:.2f}")

        if terminated or truncated:
            break

    # Energy normalised to kJ for paper tables
    energy_kj = env.episode_energy / 1000.0

    return {
        "success"    : info["reached_goal"],
        "energy"     : energy_kj,
        "collided"   : env.episode_collided,
        "steps"      : step + 1,
        "d_final"    : info["d_goal"],
        "total_reward": total_reward,
        "trajectory" : np.array(trajectory),
    } 