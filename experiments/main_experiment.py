"""
main_experiment.py  —  Step 5
==============================
Full 7-method experiment runner for the paper:
  "Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning"

Methods (in paper Table 1 order)
----------------------------------
  1. APF           — classical, no training
  2. A*            — classical, grid-based
  3. RRT*          — classical, sampling-based
  4. Pure DQN      — RL only, discrete actions
  5. Pure PPO      — RL only, continuous actions
  6. DQN-APF       — DQN + APF, fixed α=0.5
  7. PPO-APF       — PPO + APF, adaptive α  ← KEY METHOD

Usage
-----
  # Full run (paper-quality, ~hours depending on hardware):
  python experiments/main_experiment.py

  # Fast mode (smoke test, ~minutes):
  python experiments/main_experiment.py --fast

  # Skip training, evaluate saved models:
  python experiments/main_experiment.py --eval-only

  # Single method:
  python experiments/main_experiment.py --method ppo_apf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.auv_environment       import AUVEnvironment
from planning.apf_module        import APFPlanner, run_apf_episode
from planning.classical_planners import run_astar_episode, run_rrtstar_episode
from planning.hybrid_fusion     import (
    make_hybrid_ppo_env, make_hybrid_dqn_env, evaluate_hybrid,
)
from agents.dqn_agent           import (
    train_dqn, evaluate_dqn, load_dqn, make_dqn_env,
)
from agents.ppo_agent           import (
    train_ppo, evaluate_ppo, load_ppo,
)


# ─── Experiment configuration ─────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """
    Single source of truth for all hyperparameters.
    Override via CLI flags or direct instantiation for ablations.
    """
    # Evaluation
    n_eval_episodes    : int   = 100
    eval_seed          : int   = 1000

    # RL training budgets  (paper values)
    ppo_timesteps      : int   = 500_000
    dqn_timesteps      : int   = 500_000
    n_envs_ppo         : int   = 4        # parallel PPO envs

    # Seeds
    train_seed         : int   = 0

    # I/O
    save_dir           : str   = "./results/models"
    results_dir        : str   = "./results"
    device             : str   = "auto"

    # Training verbosity
    verbose            : int   = 1
    log_freq           : int   = 50       # episodes between metric logs

    # Fast mode: reduced budget for quick testing
    fast_mode          : bool  = False
    fast_timesteps     : int   = 30_000
    fast_eval_episodes : int   = 20

    # Which methods to run  (None = all 7)
    methods            : list  = field(default_factory=lambda: [
        "apf", "astar", "rrtstar",
        "pure_dqn", "pure_ppo",
        "dqn_apf", "ppo_apf",
    ])

    def __post_init__(self):
        if self.fast_mode:
            self.ppo_timesteps   = self.fast_timesteps
            self.dqn_timesteps   = self.fast_timesteps
            self.n_eval_episodes = self.fast_eval_episodes
            self.n_envs_ppo      = 1
            self.log_freq        = 10
        os.makedirs(self.save_dir,   exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def n_steps(self) -> int:
        return self.ppo_timesteps

    def model_path(self, name: str) -> str:
        return os.path.join(self.save_dir, f"{name}_final")


# ─── Classical baselines ──────────────────────────────────────────────────────
def run_apf_baseline(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [1/7]  APF  —  classical, no training")
    print("━" * 60)
    t0 = time.perf_counter()
    successes, energies, collisions, steps_all = [], [], [], []

    for ep in range(cfg.n_eval_episodes):
        env    = AUVEnvironment(mode="continuous", seed=cfg.eval_seed + ep)
        planner = APFPlanner(seed=cfg.eval_seed + ep)
        r      = run_apf_episode(env, planner)
        successes.append(int(r["success"]))
        energies.append(r["energy"])
        collisions.append(int(r["collided"]))
        steps_all.append(r["steps"])
        if cfg.verbose >= 2:
            print(f"  ep {ep:3d}: {'✓' if r['success'] else '✗'}  "
                  f"E={r['energy']:.3f}kJ  steps={r['steps']}")

    result = _aggregate("APF", successes, energies, collisions, steps_all,
                         elapsed=time.perf_counter()-t0,
                         converged_ep=None)
    _print_result(result)
    return result


def run_astar_baseline(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [2/7]  A*  —  3-D grid search, 5m resolution")
    print("━" * 60)
    t0 = time.perf_counter()
    successes, energies, collisions, steps_all = [], [], [], []

    for ep in range(cfg.n_eval_episodes):
        env = AUVEnvironment(mode="continuous", seed=cfg.eval_seed + ep)
        r   = run_astar_episode(env)
        successes.append(int(r["success"]))
        energies.append(r["energy"])
        collisions.append(int(r["collided"]))
        steps_all.append(r["steps"])
        if cfg.verbose >= 2:
            print(f"  ep {ep:3d}: {'✓' if r['success'] else '✗'}  "
                  f"E={r['energy']:.3f}kJ  plan={r['plan_ms']:.0f}ms")

    result = _aggregate("A*", successes, energies, collisions, steps_all,
                         elapsed=time.perf_counter()-t0, converged_ep=None)
    _print_result(result)
    return result


def run_rrtstar_baseline(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [3/7]  RRT*  —  sampling-based, 400 iterations")
    print("━" * 60)
    t0 = time.perf_counter()
    successes, energies, collisions, steps_all = [], [], [], []

    for ep in range(cfg.n_eval_episodes):
        env = AUVEnvironment(mode="continuous", seed=cfg.eval_seed + ep)
        r   = run_rrtstar_episode(env, seed=cfg.eval_seed + ep)
        successes.append(int(r["success"]))
        energies.append(r["energy"])
        collisions.append(int(r["collided"]))
        steps_all.append(r["steps"])
        if cfg.verbose >= 2:
            print(f"  ep {ep:3d}: {'✓' if r['success'] else '✗'}  "
                  f"E={r['energy']:.3f}kJ  plan={r['plan_ms']:.0f}ms")

    result = _aggregate("RRT*", successes, energies, collisions, steps_all,
                         elapsed=time.perf_counter()-t0, converged_ep=None)
    _print_result(result)
    return result


# ─── RL baselines ─────────────────────────────────────────────────────────────
def run_pure_dqn(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [4/7]  Pure DQN  —  discrete, no APF")
    print("━" * 60)
    model_path = cfg.model_path("dqn_auv_final")

    if os.path.exists(model_path + ".zip"):
        print(f"  Loading saved model: {model_path}.zip")
        model = load_dqn(model_path, device=cfg.device)
        converged_ep = None
    else:
        model, cb, summary = train_dqn(
            total_timesteps = cfg.dqn_timesteps,
            seed            = cfg.train_seed,
            save_dir        = cfg.save_dir,
            log_freq        = cfg.log_freq,
            target_success  = 0.85,
            device          = cfg.device,
            verbose         = cfg.verbose,
        )
        converged_ep = summary.get("converged_episode")

    t0  = time.perf_counter()
    res = evaluate_dqn(model, n_episodes=cfg.n_eval_episodes,
                        seed=cfg.eval_seed)
    res["converged_episode"] = converged_ep
    res["elapsed_s"]         = time.perf_counter() - t0
    res["method"]            = "Pure DQN"
    _print_result(res)
    return res


def run_pure_ppo(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [5/7]  Pure PPO  —  continuous, no APF")
    print("━" * 60)
    model_path = cfg.model_path("ppo_auv_final")

    if os.path.exists(model_path + ".zip"):
        print(f"  Loading saved model: {model_path}.zip")
        model = load_ppo(model_path, device=cfg.device)
        converged_ep = None
    else:
        model, cb, summary = train_ppo(
            total_timesteps = cfg.ppo_timesteps,
            n_envs          = cfg.n_envs_ppo,
            seed            = cfg.train_seed,
            save_dir        = cfg.save_dir,
            log_freq        = cfg.log_freq,
            target_success  = 0.87,
            device          = cfg.device,
            verbose         = cfg.verbose,
            model_name      = "ppo_auv",
        )
        converged_ep = summary.get("converged_episode")

    t0  = time.perf_counter()
    res = evaluate_ppo(model, n_episodes=cfg.n_eval_episodes,
                        seed=cfg.eval_seed, method="Pure PPO")
    res["converged_episode"] = converged_ep
    res["elapsed_s"]         = time.perf_counter() - t0
    _print_result(res)
    return res


def run_dqn_apf(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [6/7]  DQN-APF  —  discrete, fixed α=0.5")
    print("━" * 60)
    model_path = cfg.model_path("dqn_apf_final")

    if os.path.exists(model_path + ".zip"):
        print(f"  Loading saved model: {model_path}.zip")
        model = load_dqn(model_path, device=cfg.device)
        converged_ep = None
    else:
        from agents.dqn_agent import build_dqn
        from stable_baselines3.common.callbacks import CallbackList
        from agents.training_callbacks import (
            AUVMetricsCallback, EarlyStopCallback, BestModelCallback,
        )

        print("  Building DQN-APF environment (fixed α=0.5) ...")
        env = make_hybrid_dqn_env(seed=cfg.train_seed, fixed_alpha=0.5)
        model = build_dqn(env, seed=cfg.train_seed, device=cfg.device)

        metrics_cb = AUVMetricsCallback(log_freq=cfg.log_freq,
                                         verbose=cfg.verbose)
        early_stop = EarlyStopCallback(target_success=0.88, patience=4,
                                        check_freq=3_000,
                                        metrics_cb=metrics_cb,
                                        verbose=cfg.verbose)
        saver = BestModelCallback(save_path=cfg.save_dir,
                                   model_name="dqn_apf",
                                   metrics_cb=metrics_cb,
                                   check_freq=5_000,
                                   verbose=cfg.verbose)
        model.learn(
            total_timesteps     = cfg.dqn_timesteps,
            callback            = CallbackList([metrics_cb, early_stop, saver]),
            reset_num_timesteps = True,
            progress_bar        = False,
        )
        model.save(model_path)
        converged_ep = getattr(early_stop, "converged_at_ep", None)
        env.close()

    t0  = time.perf_counter()
    res = evaluate_hybrid(
        model, alpha_mode="fixed", fixed_alpha=0.5,
        n_episodes=cfg.n_eval_episodes,
        seed=cfg.eval_seed,
        method="DQN-APF (fixed α)",
        is_discrete=True,
    )
    res["converged_episode"] = converged_ep
    res["elapsed_s"]         = time.perf_counter() - t0
    _print_result(res)
    return res


def run_ppo_apf(cfg: ExperimentConfig) -> dict:
    print("\n" + "━" * 60)
    print("  [7/7]  PPO-APF  —  adaptive α ∈ [0.3, 0.9]  ← KEY METHOD")
    print("━" * 60)
    model_path = cfg.model_path("ppo_apf_final")

    if os.path.exists(model_path + ".zip"):
        print(f"  Loading saved model: {model_path}.zip")
        model = load_ppo(model_path, device=cfg.device)
        converged_ep = None
    else:
        model, cb, summary = train_ppo(
            total_timesteps = cfg.ppo_timesteps,
            n_envs          = cfg.n_envs_ppo,
            seed            = cfg.train_seed,
            save_dir        = cfg.save_dir,
            log_freq        = cfg.log_freq,
            target_success  = 0.93,
            device          = cfg.device,
            verbose         = cfg.verbose,
            env_mode        = "continuous",   # HybridFusionEnv wraps this
            model_name      = "ppo_apf",
            ppo_overrides   = {},
        )
        # Re-train inside hybrid wrapper if pure-PPO env was used above
        # (train_ppo with env_mode='continuous' + hybrid wrapper set externally)
        # For cleaner integration: build hybrid env directly
        print("\n  Re-training PPO inside HybridFusionEnv wrapper ...")
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from agents.training_callbacks import (
            AUVMetricsCallback, EarlyStopCallback, BestModelCallback,
        )
        from agents.ppo_agent import PPO_DEFAULTS

        h_env = make_hybrid_ppo_env(
            seed=cfg.train_seed, n_envs=cfg.n_envs_ppo,
            alpha_mode="adaptive",
        )
        import torch
        kwargs = {**PPO_DEFAULTS}
        ppo_model = PPO(
            env=h_env, seed=cfg.train_seed, device=cfg.device,
            tensorboard_log="./results/tb_logs/ppo_apf",
            **kwargs,
        )
        metrics_cb = AUVMetricsCallback(log_freq=cfg.log_freq,
                                         verbose=cfg.verbose)
        early_stop = EarlyStopCallback(
            target_success=0.93, patience=5,
            check_freq=4_000, metrics_cb=metrics_cb, verbose=cfg.verbose,
        )
        saver = BestModelCallback(
            save_path=cfg.save_dir, model_name="ppo_apf",
            metrics_cb=metrics_cb, check_freq=8_000, verbose=cfg.verbose,
        )
        ppo_model.learn(
            total_timesteps=cfg.ppo_timesteps,
            callback=CallbackList([metrics_cb, early_stop, saver]),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        ppo_model.save(model_path)
        model = ppo_model
        converged_ep = getattr(early_stop, "converged_at_ep", None)
        h_env.close()

    t0  = time.perf_counter()
    res = evaluate_hybrid(
        model, alpha_mode="adaptive",
        n_episodes=cfg.n_eval_episodes,
        seed=cfg.eval_seed,
        method="PPO-APF (adaptive α)",
        is_discrete=False,
    )
    res["converged_episode"] = converged_ep
    res["elapsed_s"]         = time.perf_counter() - t0
    _print_result(res)
    return res


# ─── Orchestrator ─────────────────────────────────────────────────────────────
def run_all_experiments(cfg: ExperimentConfig) -> list[dict]:
    """
    Run every method in cfg.methods and collect results.
    Already-trained models are loaded from disk automatically.
    """
    _RUNNERS = {
        "apf"      : run_apf_baseline,
        "astar"    : run_astar_baseline,
        "rrtstar"  : run_rrtstar_baseline,
        "pure_dqn" : run_pure_dqn,
        "pure_ppo" : run_pure_ppo,
        "dqn_apf"  : run_dqn_apf,
        "ppo_apf"  : run_ppo_apf,
    }

    print("\n" + "═" * 60)
    print("  AUV PATH PLANNING — FULL EXPERIMENT SUITE")
    mode_tag = "FAST MODE" if cfg.fast_mode else "FULL MODE"
    print(f"  {mode_tag}  |  eval_eps={cfg.n_eval_episodes}  "
          f"train_steps={cfg.ppo_timesteps:,}")
    print("═" * 60)

    results = []
    t_total = time.perf_counter()

    for method_key in cfg.methods:
        if method_key not in _RUNNERS:
            print(f"  ⚠  Unknown method '{method_key}' — skipping")
            continue
        try:
            res = _RUNNERS[method_key](cfg)
            results.append(res)
        except Exception as e:
            print(f"  ✗  {method_key} failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n  Total wall time: {(time.perf_counter()-t_total)/60:.1f} min")
    return results


# ─── Results display & persistence ───────────────────────────────────────────
_PAPER_TARGETS = {
    "APF"                    : {"success_rate": 0.82, "mean_energy_kj": 1.45, "collision_rate": 0.08},
    "Pure PPO"               : {"success_rate": 0.82, "mean_energy_kj": 1.82, "collision_rate": 0.12},
    "PPO-APF (adaptive α)"   : {"success_rate": 0.96, "mean_energy_kj": 1.28, "collision_rate": 0.01},
}


def print_results_table(results: list[dict]) -> None:
    """
    Print Table 1 from the paper with our reproduced results alongside.
    """
    print("\n" + "═" * 90)
    print("  TABLE 1 — AUV Path Planning Benchmark Results")
    print("═" * 90)
    hdr = (f"  {'Method':<28} {'Success':>8} {'Energy(kJ)':>11} "
           f"{'Collisions':>11} {'Conv.Ep':>9}  {'Status'}")
    print(hdr)
    print("  " + "─" * 86)

    for r in results:
        method  = r.get("method", "?")
        sr      = r.get("success_rate",   0.0)
        ej      = r.get("mean_energy_kj", 0.0)
        cr      = r.get("collision_rate", 0.0)
        conv    = r.get("converged_episode")
        conv_s  = f"{conv:,}" if conv else "—"

        target  = _PAPER_TARGETS.get(method, {})
        sr_t    = target.get("success_rate",   None)
        ej_t    = target.get("mean_energy_kj", None)

        # Comparison badges
        sr_badge = ""
        if sr_t:
            diff = sr - sr_t
            sr_badge = f"({'↑' if diff>=0 else '↓'}{abs(diff)*100:.1f}pp)"
        ej_badge = ""
        if ej_t:
            diff = ej - ej_t
            ej_badge = f"({'↑' if diff>0 else '↓'}{abs(diff)*100:.0f}J)"

        star = " ★" if "PPO-APF" in method and "adaptive" in method else ""
        print(
            f"  {method+star:<28} {sr*100:7.1f}%  {ej:9.4f}kJ  "
            f"{cr*100:9.1f}%   {conv_s:>8}  "
            f"{sr_badge} {ej_badge}"
        )

    print("  " + "─" * 86)
    print("  ★ = paper's key contribution")

    # Paper target reminder
    print("\n  PAPER TARGETS:")
    print(f"  {'APF':<28} 82.0%   1.4500kJ     8.0%")
    print(f"  {'Pure PPO':<28} 82.0%   1.8200kJ    12.0%")
    print(f"  {'PPO-APF (adaptive α)':<28} 96.0%   1.2800kJ     1.0%")
    print("═" * 90)


def save_results(results: list[dict], cfg: ExperimentConfig) -> None:
    """Persist results to JSON and CSV for analysis."""
    import csv

    ts = int(time.time())

    # JSON (full fidelity)
    json_path = os.path.join(cfg.results_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "config" : asdict(cfg),
            "results": results,
            "timestamp": ts,
        }, f, indent=2, default=str)
    print(f"\n  Results saved → {json_path}")

    # CSV (for Excel / pandas)
    csv_path = os.path.join(cfg.results_dir, f"results_{ts}.csv")
    if results:
        keys = ["method", "success_rate", "mean_energy_kj",
                "collision_rate", "mean_steps", "converged_episode"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"  CSV saved      → {csv_path}")

    # Also save a "latest" symlink-style copy
    latest_path = os.path.join(cfg.results_dir, "results_latest.json")
    with open(latest_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f,
                  indent=2, default=str)


def load_latest_results(results_dir: str = "./results") -> list[dict]:
    """Load most recent results for re-plotting without re-running."""
    path = os.path.join(results_dir, "results_latest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results found at {path}")
    with open(path) as f:
        data = json.load(f)
    return data["results"]


# ─── Internal helpers ─────────────────────────────────────────────────────────
def _aggregate(
    method    : str,
    successes : list,
    energies  : list,
    collisions: list,
    steps     : list,
    elapsed   : float,
    converged_ep: Optional[int],
) -> dict:
    return {
        "method"          : method,
        "success_rate"    : float(np.mean(successes)),
        "mean_energy_kj"  : float(np.mean(energies)),
        "collision_rate"  : float(np.mean(collisions)),
        "mean_steps"      : float(np.mean(steps)),
        "std_energy_kj"   : float(np.std(energies)),
        "std_success"     : float(np.std(successes)),
        "n_episodes"      : len(successes),
        "converged_episode": converged_ep,
        "elapsed_s"       : elapsed,
    }


def _print_result(r: dict) -> None:
    print(f"\n  ┌─ {r['method']} ──────────────────────────────────────")
    print(f"  │  success_rate  : {r['success_rate']*100:6.1f}%")
    print(f"  │  mean_energy   : {r['mean_energy_kj']:.4f} kJ")
    print(f"  │  collision_rate: {r['collision_rate']*100:6.1f}%")
    print(f"  │  mean_steps    : {r.get('mean_steps',0):.1f}")
    if r.get("converged_episode"):
        print(f"  │  converged_ep  : {r['converged_episode']:,}")
    print(f"  └──────────────────────────────────────────────────────")


# ─── CLI entry point ──────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="AUV Path Planning Experiment Runner"
    )
    p.add_argument("--fast",      action="store_true",
                   help="Fast mode: 30k training steps, 20 eval episodes")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load saved models and evaluate only")
    p.add_argument("--method",    type=str, default=None,
                   choices=["apf","astar","rrtstar","pure_dqn",
                            "pure_ppo","dqn_apf","ppo_apf"],
                   help="Run a single method only")
    p.add_argument("--device",    type=str, default="auto",
                   help="PyTorch device: 'auto', 'cpu', 'cuda'")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override training timesteps")
    p.add_argument("--eval-eps",  type=int, default=None,
                   help="Override number of evaluation episodes")
    p.add_argument("--seed",      type=int, default=0,
                   help="Training seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = ExperimentConfig(
        fast_mode  = args.fast,
        device     = args.device,
        train_seed = args.seed,
        methods    = [args.method] if args.method else [
            "apf", "astar", "rrtstar",
            "pure_dqn", "pure_ppo", "dqn_apf", "ppo_apf",
        ],
    )

    if args.timesteps:
        cfg.ppo_timesteps = args.timesteps
        cfg.dqn_timesteps = args.timesteps
    if args.eval_eps:
        cfg.n_eval_episodes = args.eval_eps

    results = run_all_experiments(cfg)
    print_results_table(results)
    save_results(results, cfg)