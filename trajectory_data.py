from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.auv_environment import AUVEnvironment, AUVEnvironmentDiscrete
from planning.apf_module import APFPlanner
from planning.hybrid_fusion import DiscreteHybridFusionEnv, HybridFusionEnv

METHODS = ('APF', 'Pure PPO', 'Pure DQN', 'PPO-APF (adaptive α)', 'DQN-APF (fixed α)')
MODEL_PATHS = {
    'Pure PPO': os.path.join(ROOT, 'results', 'models', 'ppo_auv_final.zip'),
    'Pure DQN': os.path.join(ROOT, 'results', 'models', 'dqn_auv_final.zip'),
    'PPO-APF (adaptive α)': os.path.join(ROOT, 'results', 'models', 'ppo_apf_best.zip'),
    'DQN-APF (fixed α)': os.path.join(ROOT, 'results', 'models', 'dqn_apf_best.zip'),
}


def _snapshot(seed: int):
    reference = AUVEnvironment(mode='continuous', seed=seed)
    reference.reset()
    obstacles = [(obstacle.pos.copy(), obstacle.radius) for obstacle in reference.obstacles]
    return reference.auv_pos.copy(), reference.goal_pos.copy(), obstacles, reference.current_vec.copy()


def _apply_snapshot(env, obstacles, current):
    for obstacle, (position, _) in zip(env.obstacles, obstacles):
        obstacle.pos = position.copy()
        obstacle.vel[:] = 0.0
    env.current_vec = current.copy()


def _run(env, action_fn: Callable, obstacles, current, base_env=None):
    base = base_env or env
    obs, _ = env.reset()
    _apply_snapshot(base, obstacles, current)
    points = [base.auv_pos.copy()]
    telemetry = []
    done = False
    info = {}
    while not done:
        action = action_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        points.append(base.auv_pos.copy())
        telemetry.append({
            'alpha': info.get('alpha'),
            'alignment': info.get('alignment'),
            'd_nearest': info.get('d_nearest'),
            'energy_kj': base.episode_energy / 1000.0,
        })
        done = terminated or truncated
    env.close()
    return {
        'positions': np.asarray(points),
        'telemetry': telemetry,
        'success': bool(info.get('reached_goal', False)),
        'collided': bool(base.episode_collided),
        'energy_kj': float(base.episode_energy / 1000.0),
        'steps': len(points) - 1,
    }


def collect_trajectories(seed: int = 3, methods: tuple[str, ...] = METHODS):
    start, goal, obstacles, current = _snapshot(seed)
    trajectories = {}

    if 'APF' in methods:
        env = AUVEnvironment(mode='continuous', seed=seed)
        planner = APFPlanner(seed=seed)
        planner.reset()
        trajectories['APF'] = _run(env, lambda obs: planner.compute_action(env), obstacles, current)

    model_cache = {}
    for method in methods:
        if method == 'APF':
            continue
        path = MODEL_PATHS[method]
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing checkpoint for {method}: {path}')
        if method in ('Pure DQN', 'DQN-APF (fixed α)'):
            from stable_baselines3 import DQN
            model_cache[method] = DQN.load(path, device='cpu')
        else:
            from stable_baselines3 import PPO
            model_cache[method] = PPO.load(path, device='cpu')

    if 'Pure PPO' in methods:
        env = AUVEnvironment(mode='continuous', seed=seed)
        model = model_cache['Pure PPO']
        trajectories['Pure PPO'] = _run(env, lambda obs: model.predict(obs, deterministic=True)[0], obstacles, current)

    if 'Pure DQN' in methods:
        env = AUVEnvironmentDiscrete(seed=seed)
        model = model_cache['Pure DQN']
        trajectories['Pure DQN'] = _run(env, lambda obs: int(model.predict(obs, deterministic=True)[0]), obstacles, current)

    if 'PPO-APF (adaptive α)' in methods:
        base = AUVEnvironment(mode='continuous', seed=seed)
        wrapper = HybridFusionEnv(base, APFPlanner(seed=seed), alpha_mode='adaptive')
        model = model_cache['PPO-APF (adaptive α)']
        trajectories['PPO-APF (adaptive α)'] = _run(wrapper, lambda obs: model.predict(obs, deterministic=True)[0], obstacles, current, base)

    if 'DQN-APF (fixed α)' in methods:
        base = AUVEnvironmentDiscrete(seed=seed)
        wrapper = DiscreteHybridFusionEnv(base, APFPlanner(seed=seed), alpha_mode='fixed', fixed_alpha=0.5)
        model = model_cache['DQN-APF (fixed α)']
        trajectories['DQN-APF (fixed α)'] = _run(wrapper, lambda obs: int(model.predict(obs, deterministic=True)[0]), obstacles, current, base)

    environment = {
        'start': start,
        'goal': goal,
        'obstacles': obstacles,
        'current': current,
    }
    return trajectories, environment
