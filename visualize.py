"""
visualize.py  —  Step 6
========================
Publication-quality figures for:
  "Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning"

Figures generated
-----------------
  fig1_3d_trajectories.pdf   — 3-D path comparison: APF vs PPO vs PPO-APF
  fig2_training_curves.pdf   — success rate + energy over training episodes
  fig3_alpha_schedule.pdf    — α(d) adaptive schedule + smooth variant
  fig4_results_comparison.pdf— bar charts for all 7 methods (Table 1 visual)
  fig5_energy_distribution.pdf — violin/box plot of per-episode energy

Usage
-----
  python visualize.py                    # generate all figures
  python visualize.py --figure 1        # single figure
  python visualize.py --show            # display interactively
"""

from __future__ import annotations
import argparse, os, sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines   import Line2D
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.auv_environment       import AUVEnvironment, AUVEnvironmentDiscrete
from planning.apf_module        import APFPlanner, run_apf_episode
from planning.hybrid_fusion     import HybridFusionEnv, DiscreteHybridFusionEnv, evaluate_hybrid
from trajectory_data             import METHODS, collect_trajectories

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 12,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "legend.fontsize"   : 10,
    "legend.framealpha" : 0.85,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# Method colour palette (consistent across all figures)
COLORS = {
    "APF"                  : "#E67E22",   # orange
    "A*"                   : "#27AE60",   # green
    "RRT*"                 : "#8E44AD",   # purple
    "Pure DQN"             : "#2980B9",   # blue
    "Pure PPO"             : "#E74C3C",   # red
    "DQN-APF (fixed α)"   : "#17A589",   # teal
    "PPO-APF (adaptive α)" : "#2C3E50",   # near-black  ← KEY METHOD
}
LINESTYLES = {
    "APF"                  : "--",
    "A*"                   : "-.",
    "RRT*"                 : ":",
    "Pure DQN"             : "-",
    "Pure PPO"             : "-",
    "DQN-APF (fixed α)"   : "-",
    "PPO-APF (adaptive α)" : "-",
}
OUTPUT_DIR = os.path.join(_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Trajectory collection helpers ───────────────────────────────────────────
def _collect_trajectory(env, action_fn, max_steps=500) -> np.ndarray:
    """Run one episode, return (N, 3) trajectory array."""
    obs, _ = env.reset()
    traj   = [env.auv_pos.copy()]
    done   = False
    steps  = 0
    while not done and steps < max_steps:
        action  = action_fn(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        traj.append(env.auv_pos.copy())
        done  = terminated or truncated
        steps += 1
    return np.array(traj)


def _apf_action_fn(obs, env):
    planner = getattr(_apf_action_fn, "_planner", None)
    if planner is None:
        _apf_action_fn._planner = APFPlanner(seed=0)
        _apf_action_fn._planner.reset()
    return _apf_action_fn._planner.compute_action(env)


def _sphere_mesh(cx, cy, cz, r, n=14):
    """Return (X, Y, Z) surface arrays for a sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0,     np.pi, n)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = cz + r * np.outer(np.ones(n), np.cos(v))
    return X, Y, Z


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  3-D Trajectory Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_3d_trajectories(
    seed     : int  = 3,
    save     : bool = True,
    show     : bool = False,
    ppo_path : Optional[str] = None,
    apf_path : Optional[str] = None,
) -> plt.Figure:
    """Render all five real agent/planner trajectories in one shared world."""
    trajectories, env_info = collect_trajectories(seed, METHODS)
    start_pos = env_info["start"]
    goal_pos = env_info["goal"]
    obs_snaps = env_info["obstacles"]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection="3d")

    for method in METHODS:
        traj = trajectories[method]["positions"]
        color = COLORS[method]
        ls = "--" if method == "APF" else ":" if method == "DQN-APF (fixed α)" else "-"
        lw = 2.5 if "PPO-APF" in method else 2.0
        label = method
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, linewidth=lw, linestyle=ls,
                label=label, alpha=0.9, zorder=5)
        # Direction arrows every 20 steps
        for k in range(20, len(traj)-1, 25):
            d = traj[k+1] - traj[k]
            ax.quiver(traj[k,0], traj[k,1], traj[k,2],
                      d[0], d[1], d[2],
                      color=color, alpha=0.6, length=3, normalize=True)

    # Obstacles
    for pos, rad in obs_snaps:
        X, Y, Z = _sphere_mesh(pos[0], pos[1], pos[2], rad, n=12)
        ax.plot_surface(X, Y, Z, alpha=0.18, color="#BDC3C7", zorder=1)
        ax.plot_wireframe(X, Y, Z, alpha=0.25, color="#7F8C8D",
                          linewidth=0.4, zorder=2)

    # Start / Goal markers
    ax.scatter(*start_pos, s=180, c="#27AE60", marker="^", zorder=10,
               label="Start", edgecolors="white", linewidths=1.5)
    ax.scatter(*goal_pos,  s=180, c="#E74C3C", marker="*", zorder=10,
               label="Goal",  edgecolors="white", linewidths=1.5)

    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 50)
    ax.set_title("Figure 1 — Real 3-D AUV Trajectory Comparison\n"
                 "(same environment, seed={})".format(seed), pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig1_3d_trajectories.pdf")
    if save: fig.savefig(path, dpi=300);  print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Training Curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_training_curves(
    histories: dict | None = None,
    save     : bool = True,
    show     : bool = False,
) -> plt.Figure:
    """
    Success rate and mean energy over training episodes for RL methods.

    `histories` must come from AUVMetricsCallback or another persisted
    training-log export. The function deliberately refuses synthetic curves.
    """
    if histories is None:
        raise ValueError(
            "Real training histories are required for Figure 2; "
            "synthetic curves are disabled."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_sr, ax_en = axes

    for method, hist in histories.items():
        color = COLORS.get(method, "#555555")
        ls    = LINESTYLES.get(method, "-")
        eps   = hist["episode"]
        if not eps:
            continue

        # Success rate (smoothed with rolling mean)
        sr  = np.array(hist["success_rate"])
        sr_smooth = _rolling_mean(sr, w=5)
        ax_sr.plot(eps, sr_smooth * 100, color=color, linestyle=ls,
                   linewidth=2.0, label=method, alpha=0.9)

        # Energy
        en  = np.array(hist["mean_energy_kj"])
        en_smooth = _rolling_mean(en, w=5)
        ax_en.plot(eps, en_smooth, color=color, linestyle=ls,
                   linewidth=2.0, label=method, alpha=0.9)

    # Paper target reference lines
    ax_sr.axhline(96, color=COLORS["PPO-APF (adaptive α)"],
                  linestyle=":", alpha=0.5, linewidth=1.2, label="Target (PPO-APF)")
    ax_sr.axhline(82, color=COLORS["Pure PPO"],
                  linestyle=":", alpha=0.5, linewidth=1.2, label="Target (PPO)")
    ax_en.axhline(1.28, color=COLORS["PPO-APF (adaptive α)"],
                  linestyle=":", alpha=0.5, linewidth=1.2, label="Target (PPO-APF)")
    ax_en.axhline(1.82, color=COLORS["Pure PPO"],
                  linestyle=":", alpha=0.5, linewidth=1.2, label="Target (PPO)")

    ax_sr.set_xlabel("Training Episodes");  ax_sr.set_ylabel("Success Rate (%)")
    ax_en.set_xlabel("Training Episodes");  ax_en.set_ylabel("Mean Energy (kJ)")
    ax_sr.set_title("(a) Convergence — Success Rate")
    ax_en.set_title("(b) Convergence — Energy Consumption")
    ax_sr.set_ylim(0, 105)
    ax_en.set_ylim(0.8, 3.0)
    ax_sr.legend(fontsize=9, ncol=2)
    ax_en.legend(fontsize=9, ncol=2)

    fig.suptitle("Figure 2 — Training Convergence Curves", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig2_training_curves.pdf")
    if save: fig.savefig(path, dpi=300);  print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Adaptive α Schedule
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_alpha_schedule(
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Visualise α(d) — the paper's core novelty in one diagram.
    Shows both the step function (paper spec) and smooth sigmoid alternative.
    """
    from planning.hybrid_fusion import compute_alpha, D_NEAR, D_FAR
    from planning.hybrid_fusion import ALPHA_NEAR, ALPHA_MID, ALPHA_FAR

    d_vals  = np.linspace(0, 60, 600)
    a_step  = [compute_alpha(d, "adaptive") for d in d_vals]
    a_smooth= [compute_alpha(d, "smooth")   for d in d_vals]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Step function
    ax.step(d_vals, a_step, where="post",
            color=COLORS["PPO-APF (adaptive α)"], linewidth=2.5,
            label=r"Adaptive α (paper — step function)", zorder=5)

    # Smooth alternative
    ax.plot(d_vals, a_smooth,
            color=COLORS["PPO-APF (adaptive α)"], linewidth=1.5,
            linestyle="--", alpha=0.65, label=r"Smooth sigmoid alternative")

    # Fixed α baseline
    ax.axhline(0.5, color=COLORS["DQN-APF (fixed α)"], linewidth=1.5,
               linestyle=":", label="Fixed α = 0.5  (DQN-APF baseline)")

    # Zone shading
    ax.axvspan( 0,     D_NEAR, alpha=0.07, color="#E74C3C", label=f"Near zone (d < {D_NEAR}m)")
    ax.axvspan(D_NEAR, D_FAR,  alpha=0.07, color="#F39C12", label=f"Transition ({D_NEAR}–{D_FAR}m)")
    ax.axvspan(D_FAR,  60,     alpha=0.07, color="#27AE60", label=f"Open water (d ≥ {D_FAR}m)")

    # Zone boundary markers
    for d, label in [(D_NEAR, f"d = {D_NEAR}m"), (D_FAR, f"d = {D_FAR}m")]:
        ax.axvline(d, color="#7F8C8D", linewidth=1.0, linestyle="-.", alpha=0.7)
        ax.text(d + 0.5, 0.32, label, fontsize=9, color="#7F8C8D", rotation=90, va="bottom")

    # α value annotations
    for d_pt, a_pt, lbl in [
        (D_NEAR / 2,          ALPHA_NEAR, f"α = {ALPHA_NEAR}\n(APF dominates)"),
        ((D_NEAR + D_FAR) / 2, ALPHA_MID,  f"α = {ALPHA_MID}\n(shared)"),
        (D_FAR + 10,           ALPHA_FAR,  f"α = {ALPHA_FAR}\n(RL dominates)"),
    ]:
        ax.annotate(lbl, xy=(d_pt, a_pt), xytext=(d_pt, a_pt + 0.07),
                    ha="center", fontsize=9, color="#2C3E50",
                    arrowprops=dict(arrowstyle="-", color="#BDC3C7"))

    ax.set_xlabel("Distance to Nearest Obstacle Surface, d  (m)")
    ax.set_ylabel("Fusion Coefficient  α  (RL weight)")
    ax.set_title("Figure 3 — Adaptive α Schedule:  a_exec = α·a_RL + (1−α)·a_APF")
    ax.set_xlim(0, 60);  ax.set_ylim(0.2, 1.02)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig3_alpha_schedule.pdf")
    if save: fig.savefig(path, dpi=300);  print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Results Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_results_comparison(
    results  : list[dict] | None = None,
    save     : bool = True,
    show     : bool = False,
) -> plt.Figure:
    """
    Bar chart comparing all 7 methods on success rate, energy, and collisions.
    Requires measured evaluation results; paper target values are not plotted.
    """
    if results is None or len(results) == 0:
        raise ValueError("Real evaluation results are required for Figure 4.")

    methods   = [r["method"]           for r in results]
    sr        = [r["success_rate"]*100 for r in results]
    energy    = [r["mean_energy_kj"]   for r in results]
    collision = [r["collision_rate"]*100 for r in results]
    colors_list = [
        COLORS.get(m, COLORS.get(m.split(" (")[0], "#888888"))
        for m in methods
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.arange(len(methods))
    bar_w = 0.62

    def _add_value_labels(ax, bars, fmt="{:.1f}"):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    fmt.format(h), ha="center", va="bottom", fontsize=8.5)

    # ── Success Rate ──────────────────────────────────────────────────────
    bars = axes[0].bar(x, sr, width=bar_w, color=colors_list,
                        edgecolor="white", linewidth=1.2, zorder=3)
    axes[0].axhline(96, color="#2C3E50", linestyle="--",
                     linewidth=1.2, alpha=0.5, label="PPO-APF target")
    _add_value_labels(axes[0], bars, "{:.1f}%")
    axes[0].set_ylabel("Success Rate (%)"); axes[0].set_ylim(0, 110)
    axes[0].set_title("(a) Success Rate")

    # ── Energy ────────────────────────────────────────────────────────────
    bars = axes[1].bar(x, energy, width=bar_w, color=colors_list,
                        edgecolor="white", linewidth=1.2, zorder=3)
    axes[1].axhline(1.28, color="#2C3E50", linestyle="--",
                     linewidth=1.2, alpha=0.5)
    _add_value_labels(axes[1], bars, "{:.3f}")
    axes[1].set_ylabel("Mean Energy (kJ)"); axes[1].set_ylim(0, 2.2)
    axes[1].set_title("(b) Energy Consumption")

    # ── Collision Rate ────────────────────────────────────────────────────
    bars = axes[2].bar(x, collision, width=bar_w, color=colors_list,
                        edgecolor="white", linewidth=1.2, zorder=3)
    _add_value_labels(axes[2], bars, "{:.1f}%")
    axes[2].set_ylabel("Collision Rate (%)"); axes[2].set_ylim(0, 18)
    axes[2].set_title("(c) Collision Rate")

    short_labels = [
        m.replace(" (adaptive α)", "\n(adaptive α)")
         .replace(" (fixed α)", "\n(fixed α)")
        for m in methods
    ]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    # Star marker on PPO-APF bars
    for ax, vals in zip(axes, [sr, energy, collision]):
        for i, m in enumerate(methods):
            if "PPO-APF" in m and "adaptive" in m:
                ax.text(i, vals[i] + (0.8 if ax is axes[0] else 0.02),
                        "★", ha="center", fontsize=13, color="#2C3E50")

    # Shared legend (method colours)
    handles = [mpatches.Patch(color=c, label=m)
               for m, c in zip(methods, colors_list)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10), fontsize=9, framealpha=0.9)

    fig.suptitle("Figure 4 — Performance Comparison: All 7 Methods  (★ = key contribution)",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig4_results_comparison.pdf")
    if save: fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Energy Distribution (Violin Plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_energy_distribution(
    n_episodes: int  = 30,
    seed      : int  = 2000,
    save      : bool = True,
    show      : bool = False,
) -> plt.Figure:
    """
    Violin plot showing per-episode energy spread for key methods.
    Shows measured per-episode energy for the evaluated checkpoints.
    """
    from planning.hybrid_fusion import HybridFusionEnv
    from planning.apf_module    import APFPlanner

    fig, ax = plt.subplots(figsize=(10, 6))

    energies_by_method = {}

    # APF
    apf_energies = []
    for ep in range(n_episodes):
        env = AUVEnvironment(mode="continuous", seed=seed + ep)
        pl  = APFPlanner(seed=seed + ep)
        r   = run_apf_episode(env, pl)
        apf_energies.append(r["energy"])
    energies_by_method["APF"] = apf_energies

    # PPO-APF (load trained model)
    ppo_apf_path = os.path.join(_ROOT, "results", "models", "ppo_apf_best.zip")
    try:
        from stable_baselines3 import PPO
        model = PPO.load(ppo_apf_path)
        papf_energies = []
        for ep in range(n_episodes):
            base = AUVEnvironment(mode="continuous", seed=seed + ep)
            apf  = APFPlanner(seed=seed + ep)
            env  = HybridFusionEnv(base, apf, alpha_mode="adaptive")
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            papf_energies.append(base.episode_energy / 1000.0)
            env.close()
        energies_by_method["PPO-APF (adaptive α)"] = papf_energies
    except Exception as exc:
        raise RuntimeError("PPO-APF checkpoint evaluation failed") from exc

    # Pure PPO checkpoint
    ppo_path = os.path.join(_ROOT, "results", "models", "ppo_auv_final.zip")
    try:
        from stable_baselines3 import PPO
        model_p = PPO.load(ppo_path)
        ppo_energies = []
        for ep in range(n_episodes):
            env = AUVEnvironment(mode="continuous", seed=seed + ep)
            obs, _ = env.reset()
            done   = False
            while not done:
                action, _ = model_p.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            ppo_energies.append(env.episode_energy / 1000.0)
            env.close()
        energies_by_method["Pure PPO"] = ppo_energies
    except Exception as exc:
        raise RuntimeError("Pure PPO checkpoint evaluation failed") from exc

    # ── Plot ──────────────────────────────────────────────────────────────
    method_list = list(energies_by_method.keys())
    data_list   = [energies_by_method[m] for m in method_list]
    colors_vln  = [COLORS.get(m, "#888888") for m in method_list]

    parts = ax.violinplot(data_list, positions=range(len(method_list)),
                          showmeans=True, showmedians=True,
                          showextrema=True, widths=0.6)

    for pc, col in zip(parts["bodies"], colors_vln):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    for part in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[part].set_color("#2C3E50"); parts[part].set_linewidth(1.5)

    # Scatter individual points
    for i, (data, col) in enumerate(zip(data_list, colors_vln)):
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   color=col, alpha=0.65, s=22, zorder=5, edgecolors="white",
                   linewidths=0.5)

    # Paper target
    ax.axhline(1.28, color=COLORS["PPO-APF (adaptive α)"],
               linestyle="--", linewidth=1.5, alpha=0.6,
               label="Paper target: PPO-APF 1.28 kJ")
    ax.axhline(1.82, color=COLORS["Pure PPO"],
               linestyle="--", linewidth=1.5, alpha=0.6,
               label="Paper target: Pure PPO 1.82 kJ")

    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels(method_list, fontsize=10)
    ax.set_ylabel("Episode Energy (kJ)")
    ax.set_ylim(0.5, 2.8)
    ax.set_title(f"Figure 5 — Energy Distribution per Episode  ({n_episodes} eval episodes)\n"
                 "Lower and narrower = more efficient and consistent")
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig5_energy_distribution.pdf")
    if save: fig.savefig(path, dpi=300);  print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6  —  α and Alignment Over an Episode
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_episode_fusion_dynamics(
    seed: int  = 5,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """
    Shows how α and cos(Δyaw_RL_APF) evolve over a single episode.
    Reveals the fusion mechanism in action: α dips near obstacles,
    alignment rises as RL learns to cooperate with APF.
    """
    from planning.hybrid_fusion import HybridFusionEnv
    from planning.apf_module    import APFPlanner
    from planning.hybrid_fusion import ALPHA_NEAR, ALPHA_FAR

    ppo_apf_path = os.path.join(_ROOT, "results", "models", "ppo_apf_best.zip")
    from stable_baselines3 import PPO
    model = PPO.load(ppo_apf_path)

    base = AUVEnvironment(mode="continuous", seed=seed)
    apf  = APFPlanner(seed=seed)
    env  = HybridFusionEnv(base, apf, alpha_mode="adaptive")
    obs, _ = env.reset()

    alphas, alignments, d_nearest_vals, d_goal_vals = [], [], [], []
    done = False
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            raise RuntimeError("PPO-APF checkpoint is required for Figure 6.")
        obs, _, terminated, truncated, info = env.step(action)
        alphas.append(info["alpha"])
        alignments.append(info["alignment"])
        d_nearest_vals.append(info["d_nearest"])
        d_goal_vals.append(info["d_goal"])
        done = terminated or truncated
    env.close()

    t = np.arange(len(alphas))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: α over time
    axes[0].plot(t, alphas, color=COLORS["PPO-APF (adaptive α)"],
                 linewidth=1.8, label="α (fusion coefficient)")
    axes[0].axhline(ALPHA_NEAR, color="#E74C3C", linestyle=":",
                    alpha=0.6, linewidth=1.2, label=f"α_near = {ALPHA_NEAR}")
    axes[0].axhline(ALPHA_FAR, color="#27AE60", linestyle=":",
                    alpha=0.6, linewidth=1.2, label=f"α_far = {ALPHA_FAR}")
    axes[0].set_ylabel("α (RL weight)")
    axes[0].set_ylim(0.20, 1.0)
    axes[0].set_title("Figure 6 — Fusion Dynamics Over One Episode")
    axes[0].legend(fontsize=9, loc="upper right")

    # Panel 2: alignment bonus
    axes[1].plot(t, alignments, color="#E67E22", linewidth=1.8,
                 label="cos(Δyaw_RL_APF)  [alignment]")
    axes[1].axhline(0, color="#7F8C8D", linestyle="-", linewidth=0.8)
    axes[1].fill_between(t, 0, alignments,
                          where=np.array(alignments) > 0,
                          alpha=0.18, color="#27AE60", label="Positive alignment")
    axes[1].fill_between(t, 0, alignments,
                          where=np.array(alignments) < 0,
                          alpha=0.18, color="#E74C3C", label="Negative alignment")
    axes[1].set_ylabel("cos(Δyaw)")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend(fontsize=9, loc="lower right")

    # Panel 3: d_nearest and d_goal
    axes[2].plot(t, d_nearest_vals, color="#8E44AD", linewidth=1.5,
                 label="d_nearest_obs (m)")
    axes[2].plot(t, d_goal_vals,    color="#2980B9", linewidth=1.5,
                 linestyle="--", label="d_goal (m)")
    from planning.hybrid_fusion import D_NEAR, D_FAR
    axes[2].axhline(D_NEAR, color="#E74C3C", linestyle=":",
                    alpha=0.5, linewidth=1.0, label=f"D_NEAR = {D_NEAR}m")
    axes[2].axhline(D_FAR, color="#27AE60", linestyle=":",
                    alpha=0.5, linewidth=1.0, label=f"D_FAR = {D_FAR}m")
    axes[2].set_ylabel("Distance (m)")
    axes[2].set_xlabel("Timestep within Episode")
    axes[2].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_episode_fusion_dynamics.pdf")
    if save: fig.savefig(path, dpi=300);  print(f"  Saved → {path}")
    if show: plt.show()
    return fig


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _rolling_mean(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Causal rolling mean (no look-ahead)."""
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        out[i] = arr[max(0, i-w+1): i+1].mean()
    return out


# ─── Main entry point ─────────────────────────────────────────────────────────
def generate_all_figures(
    results  : list[dict] | None = None,
    histories: dict | None = None,
    show     : bool = False,
) -> None:
    """Generate and save all 6 paper figures."""
    print("\n" + "═" * 60)
    print("  GENERATING PAPER FIGURES")
    print(f"  Output → {OUTPUT_DIR}")
    print("═" * 60)

    # Load measured results if not provided. Paper targets are never substituted.
    if results is None:
        results = _load_results()
        print(f"  Loaded results: {len(results)} measured methods")

    print("\n  [1/6] 3-D trajectory comparison ...")
    fig1_3d_trajectories(show=show)

    if histories is not None:
        print("  [2/6] Training convergence curves ...")
        fig2_training_curves(histories=histories, show=show)
    else:
        print("  [2/6] Training convergence curves skipped: no real callback history supplied")

    print("  [3/6] Adaptive α schedule ...")
    fig3_alpha_schedule(show=show)

    print("  [4/6] Results comparison bar chart ...")
    fig4_results_comparison(results=results, show=show)

    print("  [5/6] Energy distribution violin plot ...")
    fig5_energy_distribution(n_episodes=20, show=show)

    print("  [6/6] Episode fusion dynamics ...")
    fig6_episode_fusion_dynamics(show=show)

    print("\n" + "═" * 60)
    print(f"  ✅  All 6 figures saved to {OUTPUT_DIR}/")
    print("═" * 60)


def _load_results() -> list[dict]:
    import glob
    import json
    paths = sorted(glob.glob(os.path.join(_ROOT, "results", "results_*.json")))
    if not paths:
        raise FileNotFoundError("No measured results JSON found in results/")
    with open(paths[-1], encoding="utf-8") as f:
        results = json.load(f)["results"]
    wanted = {"APF", "Pure PPO", "Pure DQN", "PPO-APF (adaptive α)", "DQN-APF (fixed α)"}
    measured = [row for row in results if row["method"] in wanted]
    if not any(row["method"] == "DQN-APF (fixed α)" for row in measured):
        from stable_baselines3 import DQN
        checkpoint = os.path.join(_ROOT, "results", "models", "dqn_apf_best.zip")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Missing DQN-APF checkpoint: {checkpoint}")
        model = DQN.load(checkpoint, device="cpu")
        measured.append(evaluate_hybrid(
            model,
            alpha_mode="fixed",
            fixed_alpha=0.5,
            n_episodes=100,
            seed=1000,
            method="DQN-APF (fixed α)",
            is_discrete=True,
        ))
    return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, default=0,
                        help="0=all, 1-6=specific figure")
    parser.add_argument("--show",  action="store_true")
    args = parser.parse_args()

    fns = {
        1: fig1_3d_trajectories,
        2: fig2_training_curves,
        3: fig3_alpha_schedule,
        4: fig4_results_comparison,
        5: fig5_energy_distribution,
        6: fig6_episode_fusion_dynamics,
    }

    if args.figure == 0:
        generate_all_figures(show=args.show)
    elif args.figure in fns:
        fns[args.figure](show=args.show)
    else:
        print(f"Unknown figure {args.figure}. Use 1-6 or 0 for all.")