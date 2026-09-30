from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.auv_environment import AUVEnvironment, AUVEnvironmentDiscrete
from planning.apf_module import APFPlanner
from planning.hybrid_fusion import DiscreteHybridFusionEnv, HybridFusionEnv


def snapshot(env):
    return [(obstacle.pos.copy(), obstacle.radius) for obstacle in env.obstacles], env.current_vec.copy()


def apply_snapshot(env, obstacles, current):
    for obstacle, (position, _) in zip(env.obstacles, obstacles):
        obstacle.pos = position.copy()
    env.current_vec = current.copy()


def run_apf(seed):
    env = AUVEnvironment(mode='continuous', seed=seed)
    env.reset()
    planner = APFPlanner(seed=seed)
    planner.reset()
    return run_loop(env, lambda obs: planner.compute_action(env))


def run_model(method, seed):
    if method in ('Pure DQN', 'DQN-APF (fixed α)'):
        from agents.dqn_agent import load_dqn
        env = AUVEnvironmentDiscrete(seed=seed)
        checkpoint = 'dqn_auv_final.zip' if method == 'Pure DQN' else 'dqn_apf_best.zip'
        model = load_dqn(os.path.join(ROOT, 'results', 'models', checkpoint), device='cpu')
        if method == 'Pure DQN':
            return run_loop(env, lambda obs: int(model.predict(obs, deterministic=True)[0]))
        wrapper = DiscreteHybridFusionEnv(env, APFPlanner(seed=seed), alpha_mode='fixed', fixed_alpha=0.5)
        return run_loop(wrapper, lambda obs: int(model.predict(obs, deterministic=True)[0]), base_env=env)

    from agents.ppo_agent import load_ppo
    env = AUVEnvironment(mode='continuous', seed=seed)
    if method == 'Pure PPO':
        model = load_ppo(os.path.join(ROOT, 'results', 'models', 'ppo_auv_final.zip'), device='cpu')
        return run_loop(env, lambda obs: model.predict(obs, deterministic=True)[0])

    model = load_ppo(os.path.join(ROOT, 'results', 'models', 'ppo_apf_best.zip'), device='cpu')
    planner = APFPlanner(seed=seed)
    wrapped = HybridFusionEnv(env, planner, alpha_mode='adaptive')
    return run_loop(wrapped, lambda obs: model.predict(obs, deterministic=True)[0], base_env=env)


def run_loop(env, action_fn, base_env=None):
    base = base_env or env
    obs, _ = env.reset()
    points = [base.auv_pos.copy().tolist()]
    telemetry = []
    done = False
    info = {}
    while not done:
        action = action_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        points.append(base.auv_pos.copy().tolist())
        telemetry.append({
            'step': int(info.get('step', len(points) - 1)),
            'position': points[-1],
            'energy_kj': float(base.episode_energy / 1000.0),
            'alpha': info.get('alpha'),
            'alignment': info.get('alignment'),
            'd_nearest': info.get('d_nearest'),
        })
        done = terminated or truncated

    output = {
        'trajectory': points,
        'telemetry': telemetry,
        'success': bool(info.get('reached_goal', False)),
        'collided': bool(base.episode_collided),
        'energy_kj': float(base.episode_energy / 1000.0),
        'steps': len(points) - 1,
    }
    env.close()
    return output


if __name__ == '__main__':
    method = sys.argv[1]
    seed = int(sys.argv[2])
    if method == 'APF':
        result = run_apf(seed)
    else:
        result = run_model(method, seed)
    print(json.dumps(result))
