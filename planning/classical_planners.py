"""
classical_planners.py  —  Step 5 (support module)
==================================================
A* and RRT* path planners for 3-D AUV navigation.

Both planners receive a STATIC snapshot of obstacle positions at episode
start (standard practice for classical planners benchmarked against RL),
produce a waypoint path, then follow it in the dynamic environment using
a proportional heading controller.

A* (grid-based)
---------------
  Grid  : 5 m resolution  →  20×20×10 cells covering 100×100×50 m
  Graph : 26-connected (diagonal moves allowed in 3-D)
  h(n)  : Euclidean distance to goal  (admissible, consistent)
  Inflation: obstacle cells expanded by (obs_radius + 2m safety margin)

RRT* (sampling-based)
---------------------
  Space     : continuous [0,100]×[0,100]×[0,50]
  Step size : 8 m
  Near radius: 25 m  (for rewire)
  Max iters : 2000
  Goal bias : 10%  (sample goal directly 10% of the time)
  Path smoothing: line-of-sight shortcutting on final path
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from envs.auv_environment import (
    AUVEnvironment, V_MIN, V_MAX,
    YAW_RATE_MAX, PITCH_RATE_MAX, OBS_RADIUS, GOAL_RADIUS,
)

# ─── Shared waypoint controller ───────────────────────────────────────────────
def _action_toward_waypoint(
    auv_pos   : np.ndarray,
    auv_yaw   : float,
    auv_pitch : float,
    waypoint  : np.ndarray,
    speed_n   : float = 0.6,       # normalised speed ∈ [-1,1]  → ~1.4 m/s
) -> np.ndarray:
    """
    Proportional controller: compute continuous action toward a waypoint.
    Returns normalised action [speed_n, yaw_rate_n, pitch_rate_n].
    """
    diff      = waypoint - auv_pos
    dist      = float(np.linalg.norm(diff))
    if dist < 1e-3:
        return np.array([speed_n, 0.0, 0.0], dtype=np.float32)

    tgt_yaw   = math.atan2(diff[1], diff[0])
    tgt_pitch = math.atan2(diff[2], math.sqrt(diff[0]**2 + diff[1]**2))

    dyaw   = _wrap(tgt_yaw   - auv_yaw)
    dpitch = _wrap(tgt_pitch - auv_pitch)

    yaw_n   = float(np.clip(dyaw   / YAW_RATE_MAX,   -1.0, 1.0))
    pitch_n = float(np.clip(dpitch / PITCH_RATE_MAX, -1.0, 1.0))

    return np.array([speed_n, yaw_n, pitch_n], dtype=np.float32)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def follow_path(
    env      : AUVEnvironment,
    waypoints: List[np.ndarray],
    max_steps: int = 500,
    wp_thresh: float = 6.0,
) -> dict:
    """
    Execute a pre-planned waypoint list in the dynamic environment.

    Parameters
    ----------
    waypoints : list of 3-D world-frame positions
    wp_thresh : advance to next waypoint when within this radius [m]

    Returns
    -------
    Standard episode result dict.
    """
    wp_idx = 0
    steps  = 0
    total_r = 0.0

    while steps < max_steps and wp_idx < len(waypoints):
        wp      = waypoints[wp_idx]
        dist_wp = np.linalg.norm(env.auv_pos - wp)

        if dist_wp < wp_thresh:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                break
            continue

        action = _action_toward_waypoint(
            env.auv_pos, env.auv_yaw, env.auv_pitch, wp
        )
        _, r, terminated, truncated, info = env.step(action)
        total_r += r
        steps   += 1

        if terminated or truncated:
            break

    return {
        "success"     : info.get("reached_goal", False),
        "energy"      : env.episode_energy / 1000.0,
        "collided"    : env.episode_collided,
        "steps"       : steps,
        "d_final"     : info.get("d_goal", float("inf")),
        "total_reward": total_r,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  A*  (3-D occupancy grid)
# ─────────────────────────────────────────────────────────────────────────────
GRID_RES    = 5.0          # metres per cell
INFLATE_R   = OBS_RADIUS + 2.0    # obstacle inflation radius [m]
GRID_X      = int(100 / GRID_RES)  # 20
GRID_Y      = int(100 / GRID_RES)  # 20
GRID_Z      = int(50  / GRID_RES)  # 10

# 26-connected neighbourhood offsets
_NEIGHBOURS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def _pos_to_grid(pos: np.ndarray) -> Tuple[int, int, int]:
    gx = int(np.clip(pos[0] / GRID_RES, 0, GRID_X - 1))
    gy = int(np.clip(pos[1] / GRID_RES, 0, GRID_Y - 1))
    gz = int(np.clip(pos[2] / GRID_RES, 0, GRID_Z - 1))
    return gx, gy, gz


def _grid_to_pos(cell: Tuple[int, int, int]) -> np.ndarray:
    return np.array([
        (cell[0] + 0.5) * GRID_RES,
        (cell[1] + 0.5) * GRID_RES,
        (cell[2] + 0.5) * GRID_RES,
    ])


def _build_occupancy(obstacle_positions: List[np.ndarray]) -> np.ndarray:
    """Mark cells within INFLATE_R of any obstacle centre as occupied."""
    grid = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=bool)
    for obs_pos in obstacle_positions:
        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                for gz in range(GRID_Z):
                    cell_centre = _grid_to_pos((gx, gy, gz))
                    if np.linalg.norm(cell_centre - obs_pos) < INFLATE_R:
                        grid[gx, gy, gz] = True
    return grid


def _heuristic(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def astar_plan(
    start_pos       : np.ndarray,
    goal_pos        : np.ndarray,
    obstacle_positions: List[np.ndarray],
) -> Optional[List[np.ndarray]]:
    """
    A* search on 3-D occupancy grid.

    Returns
    -------
    waypoints : list of world-frame positions, or None if no path found.
    """
    occupancy = _build_occupancy(obstacle_positions)
    start = _pos_to_grid(start_pos)
    goal  = _pos_to_grid(goal_pos)

    # If start or goal is inside an obstacle, clear it
    occupancy[start] = False
    occupancy[goal]  = False

    open_heap = [(0.0 + _heuristic(start, goal), 0.0, start)]
    came_from : dict = {}
    g_score   : dict = {start: 0.0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(_grid_to_pos(node))
                node = came_from[node]
            path.reverse()
            path.append(goal_pos.copy())
            return path

        for dx, dy, dz in _NEIGHBOURS_26:
            nx, ny, nz = current[0]+dx, current[1]+dy, current[2]+dz
            if not (0 <= nx < GRID_X and 0 <= ny < GRID_Y and 0 <= nz < GRID_Z):
                continue
            nb = (nx, ny, nz)
            if occupancy[nb]:
                continue
            move_cost = math.sqrt(dx*dx + dy*dy + dz*dz)
            tentative_g = g + move_cost
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb]  = current
                g_score[nb]    = tentative_g
                h              = _heuristic(nb, goal)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, nb))

    return None   # no path found — fallback to straight line


def run_astar_episode(env: AUVEnvironment, max_steps: int = 500) -> dict:
    """Run one A* episode: plan on static snapshot, follow in dynamic env."""
    obs, _ = env.reset()
    obs_positions = [o.pos.copy() for o in env.obstacles]

    t0   = time.perf_counter()
    path = astar_plan(env.auv_pos.copy(), env.goal_pos.copy(), obs_positions)
    plan_ms = (time.perf_counter() - t0) * 1000

    if path is None or len(path) == 0:
        # Fallback: straight line via midpoints
        path = [
            env.auv_pos + i / 5 * (env.goal_pos - env.auv_pos)
            for i in range(1, 6)
        ]

    result = follow_path(env, path, max_steps=max_steps)
    result["plan_ms"] = plan_ms
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  RRT*  (continuous 3-D)
# ─────────────────────────────────────────────────────────────────────────────
BOUNDS      = np.array([[0, 100], [0, 100], [0, 50]], dtype=float)
STEP_SIZE   = 8.0     # m — maximum extension step
NEAR_RADIUS = 20.0    # m — rewire neighbourhood
GOAL_BIAS   = 0.10    # probability of sampling goal directly
MAX_ITER    = 400


class _RRTNode:
    __slots__ = ("pos", "parent", "cost")
    def __init__(self, pos, parent=None, cost=0.0):
        self.pos    = np.asarray(pos, dtype=float)
        self.parent = parent
        self.cost   = cost


def _obs_free(p1: np.ndarray, p2: np.ndarray,
              obs_positions: List[np.ndarray],
              n_checks: int = 10) -> bool:
    """Line-of-sight check via discrete sampling along the segment."""
    for k in range(n_checks + 1):
        t   = k / n_checks
        pt  = p1 + t * (p2 - p1)
        for op in obs_positions:
            if np.linalg.norm(pt - op) < INFLATE_R:
                return False
    return True


def rrtstar_plan(
    start_pos        : np.ndarray,
    goal_pos         : np.ndarray,
    obstacle_positions: List[np.ndarray],
    rng              : Optional[np.random.Generator] = None,
) -> Optional[List[np.ndarray]]:
    """
    RRT* path planning in continuous 3-D space.

    Returns
    -------
    waypoints : smoothed list of world-frame positions, or None.
    """
    if rng is None:
        rng = np.random.default_rng()

    root  = _RRTNode(start_pos, parent=None, cost=0.0)
    nodes : List[_RRTNode] = [root]

    goal_node = None

    for _ in range(MAX_ITER):
        # Sample
        if rng.random() < GOAL_BIAS:
            q_rand = goal_pos.copy()
        else:
            q_rand = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])

        # Nearest
        dists = [np.linalg.norm(n.pos - q_rand) for n in nodes]
        nearest = nodes[int(np.argmin(dists))]

        # Steer
        diff = q_rand - nearest.pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            continue
        q_new_pos = (nearest.pos + min(STEP_SIZE, dist) / dist * diff)
        q_new_pos = np.clip(q_new_pos, BOUNDS[:, 0], BOUNDS[:, 1])

        if not _obs_free(nearest.pos, q_new_pos, obstacle_positions):
            continue

        # Near set
        near_nodes = [
            n for n in nodes
            if np.linalg.norm(n.pos - q_new_pos) < NEAR_RADIUS
        ]

        # Choose best parent from near set
        best_parent = nearest
        best_cost   = nearest.cost + np.linalg.norm(nearest.pos - q_new_pos)
        for n in near_nodes:
            c = n.cost + np.linalg.norm(n.pos - q_new_pos)
            if c < best_cost and _obs_free(n.pos, q_new_pos, obstacle_positions):
                best_parent = n
                best_cost   = c

        q_new = _RRTNode(q_new_pos, parent=best_parent, cost=best_cost)
        nodes.append(q_new)

        # Rewire
        for n in near_nodes:
            new_cost = q_new.cost + np.linalg.norm(q_new.pos - n.pos)
            if (new_cost < n.cost and
                    _obs_free(q_new.pos, n.pos, obstacle_positions)):
                n.parent = q_new
                n.cost   = new_cost

        # Goal check
        if np.linalg.norm(q_new_pos - goal_pos) < STEP_SIZE:
            if _obs_free(q_new_pos, goal_pos, obstacle_positions):
                total_cost = q_new.cost + np.linalg.norm(q_new_pos - goal_pos)
                if goal_node is None or total_cost < goal_node.cost:
                    goal_node = _RRTNode(goal_pos, parent=q_new,
                                         cost=total_cost)

    if goal_node is None:
        return None

    # Reconstruct
    path = []
    node = goal_node
    while node is not None:
        path.append(node.pos.copy())
        node = node.parent
    path.reverse()

    # Line-of-sight path smoothing
    path = _smooth_path(path, obstacle_positions)
    return path


def _smooth_path(
    path: List[np.ndarray],
    obs_positions: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Shortcut smoothing: try to skip intermediate waypoints when
    the direct line-of-sight is obstacle-free.
    """
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _obs_free(path[i], path[j], obs_positions, n_checks=15):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def run_rrtstar_episode(
    env     : AUVEnvironment,
    seed    : int = 0,
    max_steps: int = 500,
) -> dict:
    """Run one RRT* episode: plan on static snapshot, follow in dynamic env."""
    obs, _ = env.reset()
    obs_positions = [o.pos.copy() for o in env.obstacles]
    rng = np.random.default_rng(seed)

    t0   = time.perf_counter()
    path = rrtstar_plan(env.auv_pos.copy(), env.goal_pos.copy(),
                        obs_positions, rng=rng)
    plan_ms = (time.perf_counter() - t0) * 1000

    if path is None or len(path) < 2:
        path = [
            env.auv_pos + i / 5 * (env.goal_pos - env.auv_pos)
            for i in range(1, 6)
        ]

    result = follow_path(env, path, max_steps=max_steps)
    result["plan_ms"] = plan_ms
    return result