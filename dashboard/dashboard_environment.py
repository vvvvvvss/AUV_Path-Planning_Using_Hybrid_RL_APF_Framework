from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.auv_environment import AUVEnvironment, DT, ENERGY_BUDGET, GOAL_RADIUS, MAX_STEPS, OBS_RADIUS, X_MAX, Y_MAX, Z_MAX


env = AUVEnvironment(mode='continuous', seed=42)
env.reset()
print(json.dumps({
    'seed': 42,
    'world': {
        'x_max': X_MAX,
        'y_max': Y_MAX,
        'z_max': Z_MAX,
        'dt': DT,
        'max_steps': MAX_STEPS,
        'goal_radius': GOAL_RADIUS,
        'obstacle_radius': OBS_RADIUS,
        'energy_budget_j': ENERGY_BUDGET,
        'current': env.current_vec.tolist(),
    },
    'start': env.auv_pos.tolist(),
    'goal': env.goal_pos.tolist(),
    'obstacles': [
        {'x': float(obstacle.pos[0]), 'y': float(obstacle.pos[1]), 'z': float(obstacle.pos[2]), 'radius': float(obstacle.radius)}
        for obstacle in env.obstacles
    ],
}))
