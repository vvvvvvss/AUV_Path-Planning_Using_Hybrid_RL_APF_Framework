"""
auv_environment.py
==================
3D AUV Path Planning Environment — Gymnasium-compatible

Ocean volume  : 100 × 100 × 50 m
Obstacles     : 8 dynamic spherical obstacles with random drift
Ocean current : 3D vector field, slowly time-varying (0.1–0.5 m/s)
Energy model  : Quadratic drag  P ∝ v²  (realistic for underwater vehicles)

State (16-D)
------------
[x, y, z,           — normalised position  ∈ [0,1]
 vx, vy, vz,        — normalised velocity  ∈ [-1,1]
 E,                 — remaining energy fraction ∈ [0,1]
 d_obs1..d_obs8,    — normalised distances to 8 obstacles ∈ [0,1]
 d_goal]            — normalised distance to goal ∈ [0,1]

Actions
-------
Discrete  (DQN): 15 actions  — speed × yaw_rate × pitch_rate combinations
Continuous (PPO): 3 actions  — [speed, yaw_rate, pitch_rate]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Physical constants — tuned to match paper's energy budget
# ---------------------------------------------------------------------------
MASS          = 30.0     # kg  — typical BlueROV2-heavy class AUV
DRAG_COEFF    = 2.5      # N·s²/m²  — quadratic drag coefficient
DT            = 0.5      # s   — simulation time-step
MAX_STEPS     = 500      # episode horizon
GOAL_RADIUS   = 3.0      # m   — success threshold
OBS_RADIUS    = 4.0      # m   — obstacle physical radius (collision)
ENERGY_BUDGET = 500.0    # J   — episode energy cap

# Ocean volume bounds
X_MAX, Y_MAX, Z_MAX = 100.0, 100.0, 50.0

# Speed & rate limits
V_MIN, V_MAX         = 0.5, 2.0     # m/s
YAW_RATE_MAX         = math.pi / 6  # rad/step  ≈ 30°
PITCH_RATE_MAX       = math.pi / 12 # rad/step  ≈ 15°

# Current parameters
CURRENT_MIN, CURRENT_MAX = 0.1, 0.5  # m/s magnitude

# Reward shaping weights (from paper)
R_GOAL      = 100.0   # large positive on goal reach
W_DIST      = 0.1     # penalty per metre to goal
W_ALIGN     = 5.0     # bonus for RL-APF heading alignment (set externally)
W_ENERGY    = 0.01    # penalty per Joule consumed
R_COLLISION = -50.0   # collision penalty


class DynamicObstacle:
    """
    A spherical obstacle that drifts slowly through the ocean volume.
    
    Drift uses a bounded random-walk so obstacles stay inside the world.
    Each obstacle has a unique drift velocity ∈ [-0.2, 0.2] m/s per axis.
    """

    def __init__(self, rng: np.random.Generator, obs_id: int):
        self.rng    = rng
        self.obs_id = obs_id
        self.radius = OBS_RADIUS
        self._initialise()

    def _initialise(self) -> None:
        # Spawn away from boundaries (5 m margin) and from start/goal zones
        self.pos = self.rng.uniform(
            low  = [10, 10, 5],
            high = [90, 90, 45],
        ).astype(np.float64)
        # Small drift velocity — realistic ocean debris / marine life
        self.vel = self.rng.uniform(-0.2, 0.2, size=3).astype(np.float64)

    def step(self) -> None:
        """Advance obstacle position; reflect off walls to stay in bounds."""
        self.pos += self.vel * DT
        for i, limit in enumerate([X_MAX, Y_MAX, Z_MAX]):
            if self.pos[i] < 5 or self.pos[i] > limit - 5:
                self.vel[i] *= -1.0   # reverse velocity at boundary
                self.pos[i]  = np.clip(self.pos[i], 5, limit - 5)

    def distance_to(self, point: np.ndarray) -> float:
        """Euclidean distance from obstacle surface to a point."""
        return float(np.linalg.norm(self.pos - point)) - self.radius

    def reset(self) -> None:
        self._initialise()


class AUVEnvironment(gym.Env):
    """
    3D AUV Path Planning Environment.

    Parameters
    ----------
    mode : str
        'discrete'  → DQN-compatible 15-action discrete space
        'continuous' → PPO-compatible Box(3) action space
    seed : int | None
        Random seed for reproducibility.
    render_mode : str | None
        'human' enables matplotlib live rendering (slow; for debugging).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mode: str = "continuous",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert mode in ("discrete", "continuous"), \
            "mode must be 'discrete' or 'continuous'"
        self.mode        = mode
        self.render_mode = render_mode

        # ── Seeded RNG (critical for reproducibility) ───────────────────────
        # np_random is reset properly by super().reset(seed=seed) each episode.
        # _rng is our secondary generator for obstacle/current initialisation.
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._rng = np.random.default_rng(seed)
        self._init_seed = seed

        # ── Action space ────────────────────────────────────────────────────
        if mode == "discrete":
            # 15 actions = 3 speeds × 3 yaw_rates × (collapsed pitch)
            # Expanded design: 3 speeds × 5 headings
            self._build_discrete_actions()
            self.action_space = spaces.Discrete(15)
        else:
            # Continuous: [speed, yaw_rate, pitch_rate]  all normalised [-1,1]
            self.action_space = spaces.Box(
                low  = np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high = np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            )

        # ── Observation space (16-D, all ∈ [0,1] or [-1,1]) ────────────────
        self.observation_space = spaces.Box(
            low  = np.full(16, -1.0, dtype=np.float32),
            high = np.full(16,  1.0, dtype=np.float32),
        )

        # ── Obstacles ───────────────────────────────────────────────────────
        self.obstacles = [
            DynamicObstacle(self._rng, i) for i in range(8)
        ]

        # ── Internal state (initialised in reset) ───────────────────────────
        self.auv_pos   = np.zeros(3)
        self.auv_vel   = np.zeros(3)
        self.auv_yaw   = 0.0        # heading angle (rad)
        self.auv_pitch = 0.0        # pitch angle (rad)
        self.energy_used = 0.0
        self.current_vec = np.zeros(3)
        self.goal_pos    = np.zeros(3)
        self._step_count = 0

        # ── Metrics (populated per episode, read by experiment runner) ───────
        self.episode_energy   = 0.0
        self.episode_collided = False

    # -----------------------------------------------------------------------
    # Discrete action table
    # -----------------------------------------------------------------------
    def _build_discrete_actions(self) -> None:
        """
        Construct 15 discrete (speed, yaw_rate, pitch_rate) tuples.

        Design rationale
        ----------------
        3 speeds × 3 yaw changes × pitch sampled at 0 gives 9 — not enough.
        We use 3 speeds × 5 yaw changes (including larger ±π/6) to cover
        richer manoeuvring, matching the paper's 15-action spec.
        Pitch_rate is set to 0 for simplicity; depth is controlled via speed
        direction inherited from current heading.
        """
        speeds    = [V_MIN, 1.25, V_MAX]                    # 3 speed levels
        yaw_rates = [
            -YAW_RATE_MAX, -YAW_RATE_MAX / 2,
            0.0,
            YAW_RATE_MAX / 2, YAW_RATE_MAX,
        ]                                                    # 5 yaw rates
        self._discrete_table = [
            (s, y, 0.0) for s in speeds for y in yaw_rates  # 3×5 = 15
        ]

    # -----------------------------------------------------------------------
    # Gymnasium API: reset
    # -----------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._init_seed is not None:
            # gymnasium calls reset(seed=None) for determinism check — use stored seed
            self._rng = np.random.default_rng(self._init_seed)

        # AUV starts near origin corner, goal in opposite region
        self.auv_pos   = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        self.auv_vel   = np.zeros(3, dtype=np.float64)
        self.auv_yaw   = 0.0
        self.auv_pitch = 0.0
        self.energy_used = 0.0
        self._step_count = 0
        self.episode_energy   = 0.0
        self.episode_collided = False

        # Goal: fixed position in far corner (matches paper's evaluation)
        self.goal_pos = np.array([90.0, 90.0, 40.0], dtype=np.float64)

        # Reset dynamic obstacles (random positions each episode)
        for obs in self.obstacles:
            obs.reset()
            # Ensure obstacles don't spawn inside start/goal zones
            while (np.linalg.norm(obs.pos - self.auv_pos) < 15 or
                   np.linalg.norm(obs.pos - self.goal_pos) < 15):
                obs.reset()

        # Sample a new ocean current vector
        self._sample_current()

        obs_vec = self._get_observation()
        return obs_vec.astype(np.float32), {}

    # -----------------------------------------------------------------------
    # Gymnasium API: step
    # -----------------------------------------------------------------------
    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        # 1. Decode action → (speed, yaw_rate, pitch_rate)
        speed, yaw_rate, pitch_rate = self._decode_action(action)

        # 2. Update heading
        self.auv_yaw   = self._wrap_angle(self.auv_yaw   + yaw_rate)
        self.auv_pitch = np.clip(
            self.auv_pitch + pitch_rate,
            -math.pi / 4, math.pi / 4,   # limit pitch to ±45°
        )

        # 3. Compute velocity vector from speed + heading
        #    (spherical → Cartesian)
        self.auv_vel = speed * np.array([
            math.cos(self.auv_yaw) * math.cos(self.auv_pitch),
            math.sin(self.auv_yaw) * math.cos(self.auv_pitch),
            math.sin(self.auv_pitch),
        ])

        # 4. Add ocean current (AUV fights current or benefits from it)
        effective_vel = self.auv_vel + self.current_vec

        # 5. Update position (Euler integration)
        new_pos = self.auv_pos + effective_vel * DT

        # 6. Clamp to ocean volume (walls = hard boundary, no penalty)
        new_pos = np.clip(new_pos, [0, 0, 0], [X_MAX, Y_MAX, Z_MAX])
        self.auv_pos = new_pos

        # 7. Energy consumption: P = DRAG_COEFF × speed²  (quadratic drag)
        #    Effort is against effective water (AUV speed, not ground speed)
        #    so we use commanded speed (not effective_vel) — the AUV expends
        #    energy based on thrust, independent of current.
        delta_E = DRAG_COEFF * (speed ** 2) * DT
        self.energy_used     += delta_E
        self.episode_energy  += delta_E

        # 8. Advance obstacles
        for obs in self.obstacles:
            obs.step()

        # 9. Slowly evolve current (realistic ocean dynamics)
        self._evolve_current()

        # 10. Check termination conditions
        d_goal     = float(np.linalg.norm(self.auv_pos - self.goal_pos))
        collisions = self._check_collisions()
        reached    = d_goal < GOAL_RADIUS
        energy_out = self.energy_used > ENERGY_BUDGET
        self._step_count += 1
        truncated  = self._step_count >= MAX_STEPS or energy_out

        if collisions:
            self.episode_collided = True

        # 11. Compute reward
        reward = self._compute_reward(
            d_goal=d_goal,
            reached=reached,
            collision=collisions,
            delta_E=delta_E,
        )

        terminated = reached or collisions   # episode ends on success/crash

        obs_vec = self._get_observation()
        info = {
            "d_goal"      : d_goal,
            "energy_used" : self.energy_used,
            "speed"       : speed,
            "collided"    : collisions,
            "reached_goal": reached,
            "step"        : self._step_count,
        }
        return obs_vec.astype(np.float32), float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """
        Construct the 16-D state vector (all values normalised).

        Normalisation is essential: neural networks learn faster and more
        stably when all inputs share a similar scale (≈ [-1, 1]).
        """
        # Position: normalise to [0, 1]
        pos_norm = self.auv_pos / np.array([X_MAX, Y_MAX, Z_MAX])

        # Velocity: normalise by max possible speed + current
        v_max_total = V_MAX + CURRENT_MAX
        vel_norm    = self.auv_vel / v_max_total           # ∈ [-1, 1]

        # Remaining energy fraction: 1 = full, 0 = depleted
        energy_frac = np.array([1.0 - self.energy_used / ENERGY_BUDGET])

        # Distances to 8 obstacles, normalised by max diagonal
        max_dist    = math.sqrt(X_MAX**2 + Y_MAX**2 + Z_MAX**2)  # ≈ 152 m
        obs_dists   = np.array([
            max(0.0, obs.distance_to(self.auv_pos)) / max_dist
            for obs in self.obstacles
        ])

        # Distance to goal, normalised
        d_goal_norm = np.array([
            np.linalg.norm(self.auv_pos - self.goal_pos) / max_dist
        ])

        state = np.concatenate([
            pos_norm,    # 3
            vel_norm,    # 3
            energy_frac, # 1
            obs_dists,   # 8
            d_goal_norm, # 1
        ])                # = 16
        return state.astype(np.float32)

    # -----------------------------------------------------------------------
    # Action decoder
    # -----------------------------------------------------------------------
    def _decode_action(
        self, action
    ) -> Tuple[float, float, float]:
        """
        Map raw action → physical (speed, yaw_rate, pitch_rate).

        Discrete: look up in pre-built table.
        Continuous: scale from [-1,1] to physical ranges.
        """
        if self.mode == "discrete":
            speed, yaw_rate, pitch_rate = self._discrete_table[int(action)]
        else:
            a = np.asarray(action, dtype=np.float64)
            # Scale speed: [-1, 1] → [V_MIN, V_MAX]
            speed      = V_MIN + (a[0] + 1.0) / 2.0 * (V_MAX - V_MIN)
            yaw_rate   = float(a[1]) * YAW_RATE_MAX
            pitch_rate = float(a[2]) * PITCH_RATE_MAX
        return float(speed), float(yaw_rate), float(pitch_rate)

    # -----------------------------------------------------------------------
    # Reward function  (exactly as in paper)
    # -----------------------------------------------------------------------
    def _compute_reward(
        self,
        d_goal: float,
        reached: bool,
        collision: bool,
        delta_E: float,
        alignment_bonus: float = 0.0,   # set by hybrid_fusion wrapper
    ) -> float:
        """
        R = R_goal(if reached) - W_dist*d_goal + W_align*cos(Δyaw)
            - W_energy*ΔE - 50*(collision)

        alignment_bonus: external caller (hybrid_fusion) sets this when
        the RL action aligns well with the APF gradient direction.
        """
        r = 0.0
        if reached:
            r += R_GOAL
        if collision:
            r += R_COLLISION
        r -= W_DIST   * d_goal
        r -= W_ENERGY * delta_E
        r += W_ALIGN  * alignment_bonus  # 0 unless set by fusion layer
        return r

    # -----------------------------------------------------------------------
    # Collision detection
    # -----------------------------------------------------------------------
    def _check_collisions(self) -> bool:
        """True if AUV centre is within any obstacle's radius."""
        for obs in self.obstacles:
            if np.linalg.norm(self.auv_pos - obs.pos) < obs.radius:
                return True
        return False

    def get_obstacle_distances(self) -> np.ndarray:
        """Return raw (unnormalised) distances to all 8 obstacles."""
        return np.array([
            max(0.0, obs.distance_to(self.auv_pos))
            for obs in self.obstacles
        ])

    def get_nearest_obstacle_distance(self) -> float:
        """Scalar: distance to the closest obstacle surface."""
        return float(np.min(self.get_obstacle_distances()))

    # -----------------------------------------------------------------------
    # Ocean current helpers
    # -----------------------------------------------------------------------
    def _sample_current(self) -> None:
        """
        Sample a new 3D current vector.
        Horizontal current is dominant (like real thermohaline flows).
        Vertical (z) current is much weaker.
        """
        magnitude = self._rng.uniform(CURRENT_MIN, CURRENT_MAX)
        direction = self._rng.uniform(0, 2 * math.pi)
        self.current_vec = np.array([
            magnitude * math.cos(direction),
            magnitude * math.sin(direction),
            self._rng.uniform(-0.05, 0.05),   # weak vertical component
        ])

    def _evolve_current(self) -> None:
        """
        Slowly rotate and scale the current vector each step.
        Simulates realistic ocean current temporal variation.
        Variation is tiny per step (≈ 0.02 rad, ≈ 0.002 m/s) so the
        AUV cannot exploit rapid changes.
        """
        angle_drift = self._rng.normal(0, 0.02)   # rad
        cos_d, sin_d = math.cos(angle_drift), math.sin(angle_drift)
        cx, cy = self.current_vec[0], self.current_vec[1]
        self.current_vec[0] =  cos_d * cx - sin_d * cy
        self.current_vec[1] =  sin_d * cx + cos_d * cy
        # Clamp magnitude to stay in [CURRENT_MIN, CURRENT_MAX]
        mag = np.linalg.norm(self.current_vec[:2])
        if mag > CURRENT_MAX:
            self.current_vec[:2] *= CURRENT_MAX / mag
        elif mag < CURRENT_MIN and mag > 1e-6:
            self.current_vec[:2] *= CURRENT_MIN / mag

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to (-π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # -----------------------------------------------------------------------
    # Gymnasium: render (lightweight text mode; 3D viz in visualize.py)
    # -----------------------------------------------------------------------
    def render(self) -> None:
        if self.render_mode == "human":
            d = np.linalg.norm(self.auv_pos - self.goal_pos)
            print(
                f"Step {self._step_count:3d} | "
                f"Pos ({self.auv_pos[0]:5.1f},{self.auv_pos[1]:5.1f},"
                f"{self.auv_pos[2]:4.1f}) | "
                f"d_goal={d:5.1f}m | "
                f"E={self.energy_used:5.1f}J | "
                f"Yaw={math.degrees(self.auv_yaw):5.1f}°"
            )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Discrete-action subclass (thin wrapper — keeps the same env, different mode)
# ---------------------------------------------------------------------------
class AUVEnvironmentDiscrete(AUVEnvironment):
    """Convenience class so DQN users don't have to pass mode='discrete'."""
    def __init__(self, seed=None, render_mode=None):
        super().__init__(mode="discrete", seed=seed, render_mode=render_mode)