"""
dashboard_3d.py — 3D Interactive Open Day Dashboard
=====================================================
Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning
UVCE · Undergraduate Research · Underwater Robotics Conference 2026

ALL DATA IS REAL — pulled from:
  agents/             trained DQN & PPO model checkpoints
  planning/           APF planner + hybrid fusion layer
  envs/               3D ocean simulation (100×100×50 m)
  results/            saved evaluation metrics (100-episode runs)

Run:
  streamlit run dashboard_3d.py
"""

import sys, os, json, math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────── Path setup ───────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUV Path Planning — Open Day 3D",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (ocean / dark theme enhancements) ─────────────────────────────
st.markdown("""
<style>
/* Subtle dark-blue tint on the main app background */
.stApp { background-color: #0B1120 !important; }
[data-testid="stSidebar"] { background-color: #0E1726 !important; }
h1, h2, h3, h4 { color: #E0E6ED !important; }
.stMetric [data-testid="stMetricValue"] {
    font-size: 20px !important; font-family: 'Courier New', monospace !important;
}
.stTabs [data-baseweb="tab-list"] button {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Codebase imports ──────────────────────────────────────────────────────────
from envs.auv_environment import (
    AUVEnvironment, X_MAX, Y_MAX, Z_MAX,
    ENERGY_BUDGET, OBS_RADIUS, GOAL_RADIUS,
)
from planning.apf_module import APFPlanner, run_apf_episode
from planning.hybrid_fusion import (
    HybridFusionEnv, compute_alpha,
    ALPHA_NEAR, ALPHA_MID, ALPHA_FAR, D_NEAR, D_FAR,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  &  COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
METHOD_COLORS = {
    "APF (classical)"      : "#E67E22",   # warm orange
    "Pure PPO"             : "#E74C3C",   # red
    "PPO-APF (adaptive α)" : "#2ECC71",   # vibrant green  ← key method
    "Pure DQN"             : "#3498DB",   # blue
    "A*"                   : "#9B59B6",   # purple
    "RRT*"                 : "#1ABC9C",   # teal
    "DQN-APF (fixed α)"   : "#F1C40F",   # yellow
}

# Plotly dark-ocean colours
OCEAN_BG       = "rgb(8, 18, 35)"
GRID_COLOR     = "rgb(30, 50, 80)"
PANEL_BG       = "rgba(15, 25, 45, 0.85)"
LEGEND_BG      = "rgba(10, 20, 40, 0.85)"
FONT_COLOR     = "rgb(200, 210, 220)"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA  LOADING   (all cached — instant after first call)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔧 Loading trained models from agents/ …")
def load_models() -> dict:
    """
    Load SB3 trained model checkpoints from results/models/.

    Uses the real agent configs from agents/dqn_agent.py and agents/ppo_agent.py.
    Falls back gracefully if stable-baselines3 is not installed.
    """
    models = {}
    try:
        from stable_baselines3 import PPO, DQN

        model_map = {
            "PPO-APF (adaptive α)": (PPO, [
                os.path.join(_ROOT, "results", "models", "ppo_apf_best.zip"),
                os.path.join(_ROOT, "results", "models", "ppo_apf_final.zip"),
            ]),
            "Pure PPO": (PPO, [
                os.path.join(_ROOT, "results", "models", "ppo_auv_best.zip"),
                os.path.join(_ROOT, "results", "models", "ppo_auv_final.zip"),
            ]),
            "Pure DQN": (DQN, [
                os.path.join(_ROOT, "results", "models", "dqn_auv_final.zip"),
            ]),
        }
        for name, (Cls, paths) in model_map.items():
            for p in paths:
                if os.path.exists(p):
                    try:
                        models[name] = Cls.load(p, device="cpu")
                    except Exception:
                        continue
                    break
    except ImportError:
        pass
    return models


@st.cache_data(show_spinner=False)
def load_real_results() -> tuple:
    """Load evaluation results from results/results_*.json (real experiment data)."""
    # Find the most recent results file
    results_dir = os.path.join(_ROOT, "results")
    json_files = sorted(
        [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")],
        reverse=True,
    ) if os.path.isdir(results_dir) else []

    for jf in json_files:
        try:
            with open(os.path.join(results_dir, jf), encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", []), data.get("config", {})
        except Exception:
            continue
    return [], {}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY  COLLECTION   (real physics + real models)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="🏊 Running AUV simulation …")
def collect_trajectories(seed: int, methods: tuple) -> tuple:
    """
    Run all requested methods on the SAME obstacle layout, collecting
    positions, α values, alignment, and distances from the real simulation.

    Data sources:
      • APF planner     → planning/apf_module.APFPlanner
      • PPO / DQN       → results/models/*.zip  via agents/ loader
      • PPO-APF hybrid  → planning/hybrid_fusion.HybridFusionEnv
    """
    # 1. Reference environment → shared obstacle positions & current
    ref_env = AUVEnvironment(mode="continuous", seed=seed)
    ref_env.reset()
    obs_snapshot  = [(o.pos.copy(), o.radius) for o in ref_env.obstacles]
    current_vec   = ref_env.current_vec.copy()
    goal_pos      = ref_env.goal_pos.copy()
    start_pos     = ref_env.auv_pos.copy()

    models  = load_models()
    results = {}

    for method in methods:
        td = _run_one(method, seed, obs_snapshot, current_vec, models)
        if td is not None:
            results[method] = td

    env_info = dict(
        obstacles   = obs_snapshot,
        current_vec = current_vec,
        goal_pos    = goal_pos,
        start_pos   = start_pos,
    )
    return results, env_info


def _apply_snapshot(env, obs_snapshot, current_vec):
    """Overwrite obstacle positions & current so all methods share the same world."""
    for i, (pos, _) in enumerate(obs_snapshot):
        env.obstacles[i].pos = pos.copy()
    env.current_vec = current_vec.copy()


def _run_one(method, seed, obs_snapshot, current_vec, models) -> dict | None:
    """Run a single method episode on the shared environment layout."""
    env = AUVEnvironment(mode="continuous", seed=seed)
    env.reset()
    _apply_snapshot(env, obs_snapshot, current_vec)

    positions  = [env.auv_pos.copy()]
    alphas     = []
    alignments = []
    d_nearests = []
    energies   = [0.0]
    info       = {}

    # ── APF (classical) ──────────────────────────────────────────────────
    if method == "APF (classical)":
        apf = APFPlanner(seed=seed)
        apf.reset()
        done = False
        while not done:
            action = apf.compute_action(env)
            _, _, term, trunc, info = env.step(action)
            positions.append(env.auv_pos.copy())
            energies.append(env.episode_energy / 1000.0)
            done = term or trunc

    # ── Pure PPO ──────────────────────────────────────────────────────────
    elif method == "Pure PPO" and "Pure PPO" in models:
        model = models["Pure PPO"]
        obs, _ = env.reset()
        _apply_snapshot(env, obs_snapshot, current_vec)
        positions = [env.auv_pos.copy()]
        energies  = [0.0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            positions.append(env.auv_pos.copy())
            energies.append(env.episode_energy / 1000.0)
            done = term or trunc

    # ── PPO-APF (adaptive α) — THE KEY METHOD ────────────────────────────
    elif method == "PPO-APF (adaptive α)" and "PPO-APF (adaptive α)" in models:
        model = models["PPO-APF (adaptive α)"]
        apf   = APFPlanner(seed=seed)
        wrapped = HybridFusionEnv(env, apf, alpha_mode="adaptive")
        obs, _  = wrapped.reset()
        _apply_snapshot(wrapped.env, obs_snapshot, current_vec)
        positions = [wrapped.env.auv_pos.copy()]
        energies  = [0.0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = wrapped.step(action)
            positions.append(wrapped.env.auv_pos.copy())
            energies.append(wrapped.env.episode_energy / 1000.0)
            alphas.append(info.get("alpha", ALPHA_FAR))
            alignments.append(info.get("alignment", 0.0))
            d_nearests.append(info.get("d_nearest", 50.0))
            done = term or trunc

    else:
        return None   # model not available

    return dict(
        positions  = np.array(positions),
        alphas     = alphas,
        alignments = alignments,
        d_nearests = d_nearests,
        energies   = energies,
        success    = bool(info.get("reached_goal", False)),
        energy_kj  = env.episode_energy / 1000.0,
        collided   = bool(env.episode_collided),
        steps      = len(positions) - 1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3-D  VISUALISATION  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sphere_surface(cx, cy, cz, r, color, opacity=0.35, name="", n=16):
    """Return a Plotly Surface trace that draws a 3-D sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones(n), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="name",
        showlegend=False,
    )


def _danger_ring(cx, cy, cz, radius, color, n=50):
    """Dashed circle at obstacle height showing influence zone."""
    theta = np.linspace(0, 2 * np.pi, n)
    return go.Scatter3d(
        x=cx + radius * np.cos(theta),
        y=cy + radius * np.sin(theta),
        z=np.full(n, cz),
        mode="lines",
        line=dict(color=color, width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip",
    )


def _seabed_surface():
    """Create a subtle textured seabed to make depth legible in the 3-D view."""
    x = np.linspace(0, X_MAX, 24)
    y = np.linspace(0, Y_MAX, 24)
    xx, yy = np.meshgrid(x, y)
    zz = Z_MAX - 1.5 + 0.9 * np.sin(xx / 10.0) * np.cos(yy / 13.0)
    return go.Surface(
        x=xx,
        y=yy,
        z=zz,
        surfacecolor=zz,
        colorscale=[
            [0.0, "rgb(5, 23, 38)"],
            [0.45, "rgb(9, 43, 61)"],
            [1.0, "rgb(18, 70, 83)"],
        ],
        cmin=Z_MAX - 3,
        cmax=Z_MAX,
        showscale=False,
        opacity=0.72,
        name="Seabed",
        hovertemplate="Seabed<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
        showlegend=False,
    )


def _trajectory_glow(fig, traj, color):
    """Layer translucent lines behind a path to create a readable glow."""
    for width, opacity in [(15, 0.05), (9, 0.10), (5, 0.18)]:
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        ))


def _ocean_scene():
    """Standard 3-D axis layout with underwater colouring."""
    axis_common = dict(
        backgroundcolor=OCEAN_BG,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showbackground=True,
        zerolinecolor=GRID_COLOR,
        color=FONT_COLOR,
        dtick=20,
    )
    return dict(
        bgcolor=OCEAN_BG,
        xaxis=dict(title="X (m)", range=[0, X_MAX], **axis_common),
        yaxis=dict(title="Y (m)", range=[0, Y_MAX], **axis_common),
        zaxis=dict(title="Z — Depth (m)", range=[0, Z_MAX], **axis_common),
        camera=dict(
            eye=dict(x=1.65, y=1.65, z=1.12),
            up=dict(x=0, y=0, z=1),
        ),
        aspectmode="manual",
        aspectratio=dict(x=2, y=2, z=1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE  BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_3d_world(env_info, traj_results=None):
    """
    The centrepiece 3-D figure: ocean volume, obstacles, current field,
    start/goal markers, and AUV trajectory paths.
    """
    fig = go.Figure()
    obstacles  = env_info["obstacles"]
    goal       = env_info["goal_pos"]
    start      = env_info["start_pos"]
    current    = env_info["current_vec"]

    # ── Bounding-box wireframe ────────────────────────────────────────────
    edges = [
        ([0, X_MAX], [0, 0],     [0, 0]),
        ([0, X_MAX], [Y_MAX, Y_MAX], [0, 0]),
        ([0, 0],     [0, Y_MAX], [0, 0]),
        ([X_MAX, X_MAX], [0, Y_MAX], [0, 0]),
        ([0, X_MAX], [0, 0],     [Z_MAX, Z_MAX]),
        ([0, X_MAX], [Y_MAX, Y_MAX], [Z_MAX, Z_MAX]),
        ([0, 0],     [0, Y_MAX], [Z_MAX, Z_MAX]),
        ([X_MAX, X_MAX], [0, Y_MAX], [Z_MAX, Z_MAX]),
        ([0, 0],         [0, 0],         [0, Z_MAX]),
        ([X_MAX, X_MAX], [0, 0],         [0, Z_MAX]),
        ([0, 0],         [Y_MAX, Y_MAX], [0, Z_MAX]),
        ([X_MAX, X_MAX], [Y_MAX, Y_MAX], [0, Z_MAX]),
    ]
    for ex, ey, ez in edges:
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="rgba(80,120,170,0.25)", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Seabed plane ─────────────────────────────────────────────────────
    fig.add_trace(_seabed_surface())

    # ── Obstacles (spheres + danger-zone rings) ───────────────────────────
    for i, (pos, rad) in enumerate(obstacles):
        fig.add_trace(_sphere_surface(
            pos[0], pos[1], pos[2], rad,
            color="rgba(192,57,43,0.55)", opacity=0.4,
            name=f"Obstacle {i+1} (r={rad:.0f}m)",
        ))
        fig.add_trace(_danger_ring(
            pos[0], pos[1], pos[2], rad + D_NEAR,
            color="rgba(231,76,60,0.12)",
        ))

    # ── Ocean-current field (cone arrows) ─────────────────────────────────
    cx_pts, cy_pts, cz_pts = [], [], []
    cu, cv, cw = [], [], []
    for x in np.linspace(10, 90, 5):
        for y in np.linspace(10, 90, 5):
            for z in [10, 25, 40]:
                ux, uy = float(current[0]), float(current[1])
                for pos, _ in obstacles:
                    r = math.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
                    if r > 0.5:
                        th = math.atan2(y - pos[1], x - pos[0])
                        s  = 0.15 * math.exp(-r / 15)
                        ux -= s * math.sin(th)
                        uy += s * math.cos(th)
                cx_pts.append(x); cy_pts.append(y); cz_pts.append(z)
                cu.append(ux); cv.append(uy); cw.append(float(current[2]))

    fig.add_trace(go.Cone(
        x=cx_pts, y=cy_pts, z=cz_pts, u=cu, v=cv, w=cw,
        sizemode="absolute", sizeref=3,
        colorscale=[[0, "rgba(52,152,219,0.25)"], [1, "rgba(41,128,185,0.55)"]],
        showscale=False, opacity=0.35, name="Ocean Current", hoverinfo="skip",
        showlegend=False,
    ))

    # ── Start & Goal markers ──────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2]],
        mode="markers+text",
        marker=dict(size=10, color="#2ECC71", symbol="diamond",
                    line=dict(width=2, color="white")),
        text=["START"], textposition="top center",
        textfont=dict(size=12, color="#2ECC71", family="Arial Black"),
        name="Start  (5, 5, 5)", showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[goal[0]], y=[goal[1]], z=[goal[2]],
        mode="markers+text",
        marker=dict(size=12, color="#E74C3C", symbol="cross",
                    line=dict(width=2, color="white")),
        text=["GOAL"], textposition="top center",
        textfont=dict(size=12, color="#E74C3C", family="Arial Black"),
        name="Goal  (90, 90, 40)", showlegend=True,
    ))

    # A faint direct route makes the planning problem immediately obvious.
    fig.add_trace(go.Scatter3d(
        x=[start[0], goal[0]], y=[start[1], goal[1]], z=[start[2], goal[2]],
        mode="lines",
        line=dict(color="rgba(210,230,245,0.18)", width=2, dash="dot"),
        name="Direct start-goal vector",
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Trajectories from real simulation ─────────────────────────────────
    if traj_results:
        for method, data in traj_results.items():
            traj    = data["positions"]
            color   = METHOD_COLORS.get(method, "#FFFFFF")
            success = data.get("success", False)
            energy  = data.get("energy_kj", 0)
            steps   = data.get("steps", 0)
            tag     = "✓ Goal" if success else "✗ No Goal"

            # Build the visual hierarchy before the interactive foreground path.
            _trajectory_glow(fig, traj, color)

            if method == "PPO-APF (adaptive α)" and data.get("alphas"):
                # ── Colour path by α (the paper's innovation made visible) ──
                a_arr = data["alphas"]
                # Pad α list to match positions length (positions has 1 extra)
                a_plot = a_arr + [a_arr[-1]] * max(0, len(traj) - len(a_arr))
                a_plot = a_plot[:len(traj)]
                step_values = np.arange(len(traj))
                energy_values = np.asarray(data.get("energies", [0.0] * len(traj)))
                energy_values = np.pad(
                    energy_values,
                    (0, max(0, len(traj) - len(energy_values))),
                    mode="edge",
                )[:len(traj)]

                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                    mode="lines+markers",
                    line=dict(color="rgba(46,204,113,0.72)", width=7),
                    marker=dict(
                        size=4.5,
                        color=a_plot,
                        colorscale="RdYlGn",
                        cmin=0.25, cmax=0.95,
                        colorbar=dict(
                            title=dict(text="α (RL wt)", font=dict(size=10)),
                            thickness=15, len=0.4, x=1.02,
                            tickvals=[0.3, 0.5, 0.7, 0.9],
                            ticktext=["0.3 APF", "0.5", "0.7", "0.9 RL"],
                        ),
                    ),
                    name=f"★ {method}  ({tag}, {energy:.3f} kJ, {steps} steps)",
                    hovertemplate=(
                        "<b>PPO-APF (adaptive α)</b><br>"
                        "Step %{customdata[0]}<br>"
                        "(%{x:.1f}, %{y:.1f}, %{z:.1f}) m<br>"
                        "α = %{marker.color:.2f}<br>"
                        "Energy = %{customdata[1]:.3f} kJ<extra></extra>"
                    ),
                    customdata=np.column_stack((step_values, energy_values)),
                ))
            else:
                # ── Solid-colour path ────────────────────────────────────
                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=4 if "APF" in method else 3.5,
                        dash="dash" if method == "APF (classical)" else "solid",
                    ),
                    name=f"{method}  ({tag}, {energy:.3f} kJ, {steps} steps)",
                    hovertemplate=(
                        f"<b>{method}</b><br>"
                        "Step %{customdata}<br>"
                        "(%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>"
                    ),
                    customdata=np.arange(len(traj)),
                ))

            # Waypoints and endpoint status make the path understandable to a
            # non-specialist without hiding the underlying trajectory detail.
            waypoint_idx = np.linspace(0, len(traj) - 1, min(8, len(traj))).astype(int)
            fig.add_trace(go.Scatter3d(
                x=traj[waypoint_idx, 0],
                y=traj[waypoint_idx, 1],
                z=traj[waypoint_idx, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color="#F7F9FC",
                    symbol="circle-open",
                    line=dict(color=color, width=2),
                ),
                name=f"{method} waypoints",
                legendgroup=method,
                showlegend=False,
                hovertemplate=f"<b>{method}</b><br>Waypoint %{{pointNumber}}<br>"
                              "(%{x:.1f}, %{y:.1f}, %{z:.1f}) m<extra></extra>",
            ))

            # Direction arrows (cones) every few steps
            step_iv = max(1, len(traj) // 8)
            for k in range(step_iv, len(traj) - 1, step_iv):
                d = traj[k] - traj[k - 1]
                if np.linalg.norm(d) > 0.1:
                    fig.add_trace(go.Cone(
                        x=[traj[k, 0]], y=[traj[k, 1]], z=[traj[k, 2]],
                        u=[d[0]], v=[d[1]], w=[d[2]],
                        sizemode="absolute", sizeref=2.5,
                        colorscale=[[0, color], [1, color]],
                        showscale=False, opacity=0.65,
                        showlegend=False, hoverinfo="skip",
                    ))

            # A compact vehicle marker at the current/final position.
            final = traj[-1]
            direction = traj[-1] - traj[-2] if len(traj) > 1 else np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            fig.add_trace(go.Cone(
                x=[final[0]], y=[final[1]], z=[final[2]],
                u=[direction[0]], v=[direction[1]], w=[direction[2]],
                sizemode="absolute", sizeref=4.5,
                colorscale=[[0, color], [1, color]],
                showscale=False, opacity=0.95,
                name=f"{method} vehicle",
                showlegend=False,
                hovertemplate=f"<b>{method}</b><br>{tag}<br>"
                              f"Energy: {energy:.3f} kJ<extra></extra>",
            ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        scene=_ocean_scene(),
        paper_bgcolor=OCEAN_BG,
        plot_bgcolor=OCEAN_BG,
        font=dict(color=FONT_COLOR, family="Arial"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor=LEGEND_BG,
            bordercolor="rgba(100,140,180,0.4)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=720,
        hoverlabel=dict(
            bgcolor="#10283b",
            bordercolor="#58c8ef",
            font=dict(color="#f1fbff", size=12),
        ),
    )
    return fig


# ─── α schedule (2-D) ────────────────────────────────────────────────────────

def build_alpha_schedule():
    """The α(d) function — the paper's core innovation in one diagram."""
    d = np.linspace(0, 60, 400)
    a_step   = [compute_alpha(di, "adaptive") for di in d]
    a_smooth = [compute_alpha(di, "smooth")   for di in d]

    fig = go.Figure()

    # Zone shading
    fig.add_vrect(x0=0,      x1=D_NEAR, fillcolor="rgba(231,76,60,0.07)",
                  line_width=0)
    fig.add_vrect(x0=D_NEAR, x1=D_FAR,  fillcolor="rgba(243,156,18,0.06)",
                  line_width=0)
    fig.add_vrect(x0=D_FAR,  x1=60,     fillcolor="rgba(46,204,113,0.06)",
                  line_width=0)

    fig.add_trace(go.Scatter(x=d, y=a_step,   mode="lines",
                             line=dict(color="#2ECC71", width=3, shape="hv"),
                             name="Adaptive α (paper — step function)"))
    fig.add_trace(go.Scatter(x=d, y=a_smooth, mode="lines",
                             line=dict(color="#2ECC71", width=2, dash="dash"),
                             name="Smooth sigmoid alternative", opacity=0.55))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#F1C40F", opacity=0.6,
                  annotation_text="Fixed α = 0.5  (DQN-APF baseline)",
                  annotation_position="bottom right")

    fig.add_vline(x=D_NEAR, line_dash="dashdot",
                  line_color="rgba(150,150,150,0.5)",
                  annotation_text=f"D_NEAR = {D_NEAR}m")
    fig.add_vline(x=D_FAR,  line_dash="dashdot",
                  line_color="rgba(150,150,150,0.5)",
                  annotation_text=f"D_FAR = {D_FAR}m")

    for d_pt, a_pt, lbl, clr in [
        (D_NEAR / 2,          ALPHA_NEAR, "α=0.3  APF dominates\n(safety first)",  "#E74C3C"),
        ((D_NEAR + D_FAR) / 2, ALPHA_MID,  "α=0.7  Shared control",                "#F39C12"),
        (D_FAR + 12,           ALPHA_FAR,  "α=0.9  RL dominates\n(energy efficiency)", "#2ECC71"),
    ]:
        fig.add_annotation(x=d_pt, y=a_pt + 0.05, text=lbl, showarrow=True,
                           arrowhead=2, arrowcolor=clr,
                           font=dict(size=10, color=clr))

    fig.update_layout(
        xaxis_title="Distance to Nearest Obstacle Surface, d  (m)",
        yaxis_title="Fusion Coefficient  α  (RL weight)",
        yaxis=dict(range=[0.2, 1.02], dtick=0.1),
        xaxis=dict(range=[0, 60]),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL_BG,
        font=dict(color=FONT_COLOR),
        legend=dict(bgcolor=LEGEND_BG, font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor="rgba(50,70,100,0.3)")
    fig.update_yaxes(gridcolor="rgba(50,70,100,0.3)")
    return fig


# ─── α / alignment dynamics over an episode ──────────────────────────────────

def build_fusion_dynamics(traj_data):
    """3-panel chart: α, cos(Δψ) alignment, and d_nearest over timesteps."""
    alphas     = traj_data.get("alphas", [])
    alignments = traj_data.get("alignments", [])
    d_nears    = traj_data.get("d_nearests", [])

    if not alphas:
        fig = go.Figure()
        fig.add_annotation(text="Run a PPO-APF episode to see fusion dynamics",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#7F8C8D"))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor=PANEL_BG, font=dict(color=FONT_COLOR))
        return fig

    t = np.arange(len(alphas))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.35, 0.30], vertical_spacing=0.06,
        subplot_titles=(
            "α — Fusion Coefficient  (how much RL vs APF controls the AUV)",
            "Alignment — cos(Δψ)  (are RL and APF heading in the same direction?)",
            "Distance to Nearest Obstacle  (determines which zone the AUV is in)",
        ),
    )

    # Panel 1: α
    fig.add_trace(go.Scatter(
        x=t, y=alphas, mode="lines", fill="tozeroy",
        fillcolor="rgba(46,204,113,0.08)",
        line=dict(color="#2ECC71", width=2.5),
        name="α (fusion coeff)",
    ), row=1, col=1)
    for val, clr, lbl in [
        (ALPHA_NEAR, "#E74C3C", "α=0.3 APF"),
        (ALPHA_MID,  "#F39C12", "α=0.7 shared"),
        (ALPHA_FAR,  "#2ECC71", "α=0.9 RL"),
    ]:
        fig.add_hline(y=val, line_dash="dot", line_color=clr, opacity=0.5,
                      row=1, col=1, annotation_text=lbl,
                      annotation_position="right")

    # Panel 2: alignment
    if alignments:
        al = np.array(alignments)
        fig.add_trace(go.Scatter(
            x=t[:len(al)], y=np.clip(al, 0, None), mode="lines",
            fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="Cooperative (>0)", showlegend=True,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=t[:len(al)], y=np.clip(al, None, 0), mode="lines",
            fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="Opposing (<0)", showlegend=True,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=t[:len(al)], y=al, mode="lines",
            line=dict(color="#9B59B6", width=2),
            name="cos(Δψ_RL_APF)",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="rgba(150,150,150,0.4)", row=2, col=1)

    # Panel 3: d_nearest
    if d_nears:
        fig.add_trace(go.Scatter(
            x=t[:len(d_nears)], y=d_nears, mode="lines",
            line=dict(color="#3498DB", width=2),
            name="d_nearest (m)",
        ), row=3, col=1)
        fig.add_hline(y=D_NEAR, line_dash="dot", line_color="#E74C3C", opacity=0.5,
                      row=3, col=1,
                      annotation_text=f"D_NEAR = {D_NEAR}m")
        fig.add_hline(y=D_FAR, line_dash="dot", line_color="#2ECC71", opacity=0.5,
                      row=3, col=1,
                      annotation_text=f"D_FAR = {D_FAR}m")

    fig.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL_BG,
        font=dict(color=FONT_COLOR),
        legend=dict(bgcolor=LEGEND_BG, font=dict(size=9)),
    )
    fig.update_xaxes(gridcolor="rgba(50,70,100,0.3)",
                     title_text="Timestep", row=3, col=1)
    fig.update_yaxes(gridcolor="rgba(50,70,100,0.3)")
    return fig


# ─── Benchmark comparison bar charts ─────────────────────────────────────────

def build_benchmarks(real_results):
    """Bar chart trio: success rate, energy, collision rate — all from real data."""
    if not real_results:
        return go.Figure()

    methods   = [r["method"] for r in real_results]
    sr        = [r["success_rate"] * 100 for r in real_results]
    energy    = [r["mean_energy_kj"] for r in real_results]
    collision = [r.get("collision_rate", 0) * 100 for r in real_results]
    colors    = [METHOD_COLORS.get(m, "#888888") for m in methods]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Success Rate  (%)", "Energy  (kJ)", "Collision Rate  (%)"),
        horizontal_spacing=0.08,
    )

    for col_idx, (vals, fmt) in enumerate([
        (sr,        "{:.1f}%"),
        (energy,    "{:.4f}"),
        (collision, "{:.1f}%"),
    ], start=1):
        fig.add_trace(go.Bar(
            x=methods, y=vals, marker_color=colors,
            text=[fmt.format(v) for v in vals], textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL_BG,
        font=dict(color=FONT_COLOR),
        margin=dict(t=50, b=120),
    )
    fig.update_xaxes(tickangle=35, gridcolor="rgba(50,70,100,0.3)",
                     tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="rgba(50,70,100,0.3)")
    return fig


# ─── Radar / spider chart ────────────────────────────────────────────────────

def build_radar(real_results):
    """Spider plot overlaying method performance profiles."""
    if not real_results or len(real_results) < 2:
        return go.Figure()

    def hex_to_rgba(h, a):
        h = h.lstrip('#')
        if len(h) == 6:
            return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"
        return h

    cats = ["Success Rate", "Energy Efficiency", "Safety", "Path Efficiency"]
    fig = go.Figure()

    for r in real_results:
        m   = r["method"]
        clr = METHOD_COLORS.get(m, "#888888")
        sr  = r["success_rate"]
        ee  = max(0, 1.0 - r["mean_energy_kj"] / 0.8)  # normalise
        sa  = 1.0 - r.get("collision_rate", 0)
        pe  = max(0, 1.0 - r["mean_steps"] / 500)
        vals = [sr, ee, sa, pe, sr]  # close polygon

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor=hex_to_rgba(clr, 0.15),
            line=dict(color=clr, width=2),
            name=m,
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(50,70,100,0.3)"),
            angularaxis=dict(gridcolor="rgba(50,70,100,0.3)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_COLOR),
        height=460,
        legend=dict(bgcolor=LEGEND_BG),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN  UI
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = load_models()
REAL_RESULTS, EXP_CONFIG = load_real_results()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controls")

    seed = st.number_input(
        "Episode Seed", 0, 9999, 42,
        help="Same seed → same obstacle positions → fair comparison",
    )

    available = ["APF (classical)"]
    if "Pure PPO" in MODELS:
        available.append("Pure PPO")
    if "PPO-APF (adaptive α)" in MODELS:
        available.append("PPO-APF (adaptive α)")

    methods_sel = st.multiselect(
        "Methods to Compare", options=available, default=available,
    )

    run_btn = st.button("▶  Run Simulation", type="primary",
                        use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 Loaded Models")
    for name in MODELS:
        st.success(f"✅ {name}", icon="🤖")
    if not MODELS:
        st.warning("No SB3 models found — only APF is available.")

    st.markdown("---")
    st.caption(
        "**Data sources**\n"
        "- `agents/`  — RL model configs\n"
        "- `planning/` — APF & fusion\n"
        "- `envs/`    — 3-D ocean sim\n"
        "- `results/` — eval metrics"
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "traj_results" not in st.session_state:
    st.session_state.traj_results = None
if "env_info" not in st.session_state:
    st.session_state.env_info = None

if run_btn and methods_sel:
    tr, ei = collect_trajectories(int(seed), tuple(methods_sel))
    st.session_state.traj_results = tr
    st.session_state.env_info     = ei
elif st.session_state.traj_results is None:
    # Auto-run on first page load with defaults
    tr, ei = collect_trajectories(42, tuple(available))
    st.session_state.traj_results = tr
    st.session_state.env_info     = ei

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:0.8rem 0 0.4rem;">
  <h1 style="color:#2ECC71; margin-bottom:0.2rem; font-size:2rem;">
    🌊  Adaptive RL-APF Framework for AUV Path Planning
  </h1>
  <p style="font-size:1.1rem; color:#BDC3C7;">
    3-D Interactive Dashboard  ·  UVCE · Underwater Robotics Conference 2026
  </p>
  <p style="font-size:0.9rem; color:#7F8C8D;">
    All visualisations are powered by <b>real trained models</b> and
    <b>live physics simulation</b> — not dummy data
  </p>
</div>
""", unsafe_allow_html=True)

# ── Core equation ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; background:rgba(46,204,113,0.07);
            border:1px solid rgba(46,204,113,0.25); border-radius:10px;
            padding:0.9rem; margin:0 0 1rem;">
  <span style="font-size:1.35rem; font-family:'Courier New',monospace;
               color:#2ECC71; letter-spacing:1px;">
    a<sub>exec</sub> = α · a<sub>RL</sub>  +  (1 − α) · a<sub>APF</sub>
  </span><br/>
  <span style="font-size:0.85rem; color:#95A5A6;">
    α adapts with obstacle proximity:
    near obstacles → APF leads (safety)  ·  open water → RL leads (efficiency)
  </span>
</div>
""", unsafe_allow_html=True)

# ── Live metrics row ──────────────────────────────────────────────────────────
if st.session_state.traj_results:
    mcols = st.columns(len(st.session_state.traj_results))
    for col, (method, data) in zip(mcols, st.session_state.traj_results.items()):
        status = "✅ Goal Reached" if data["success"] else "❌ Did Not Reach"
        star   = " ★" if "PPO-APF" in method else ""
        col.metric(
            label=f"{method}{star}",
            value=status,
            delta=f"{data['energy_kj']:.3f} kJ  ·  {data['steps']} steps",
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  3-D Ocean World",
    "⚙️  How α Works",
    "📊  Benchmarks",
    "📈  Fusion Dynamics",
    "🧠  Architecture",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1  —  3-D Ocean World
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Interactive 3-D AUV Trajectory Comparison")
    st.markdown(
        "🖱️ **Rotate** (drag)  ·  **Zoom** (scroll)  ·  "
        "**Pan** (right-drag)  ·  **Hover** for details"
    )

    if st.session_state.env_info:
        fig_world = build_3d_world(
            st.session_state.env_info,
            st.session_state.traj_results,
        )
        st.plotly_chart(fig_world, use_container_width=True, key="main_3d")

    with st.expander("📖  What am I looking at?", expanded=False):
        st.markdown("""
**The 3-D Ocean Volume** (100 × 100 × 50 metres):

| Element | Description |
|---------|-------------|
| 🔴 Red spheres | Dynamic obstacles — 8 underwater objects drifting in the current |
| 🟢 Green diamond | AUV start position (5, 5, 5) |
| ❌ Red cross | Goal position (90, 90, 40) |
| 🔵 Blue cones | Ocean current vectors (0.1–0.5 m/s) |
| Coloured paths | AUV trajectory taken by each planning method |

**For the PPO-APF path** (our key contribution ★):
- Path colour shows the **α value** at each point
- 🔴 **Red** markers (α ≈ 0.3) → APF is in charge — obstacle avoidance mode
- 🟢 **Green** markers (α ≈ 0.9) → RL is in charge — energy-efficient routing
        """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2  —  How α Works
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### The Adaptive Fusion Mechanism")
    st.markdown(
        "The **key innovation**: instead of a fixed blend ratio, α adapts "
        "dynamically based on how close the AUV is to the nearest obstacle."
    )

    c_left, c_right = st.columns([1.3, 1])

    with c_left:
        st.plotly_chart(build_alpha_schedule(), use_container_width=True)

    with c_right:
        st.markdown("""
#### α Schedule  (Paper Spec)

| Zone | Distance | α | Controller |
|------|----------|---|------------|
| 🔴 **Near** | d < 15 m | **0.3** | APF dominates  (safety!) |
| 🟡 **Transition** | 15 ≤ d < 30 m | **0.7** | Shared control |
| 🟢 **Open** | d ≥ 30 m | **0.9** | RL dominates  (efficiency) |

#### Why it works

- **APF** is reactive and safe — but gets stuck in local minima
- **RL** is flexible and energy-efficient — but slow to learn safety
- **Hybrid** combines both: APF handles obstacles, RL optimises routes

#### Alignment Bonus

```
R_align = 5 · cos(Δψ_RL_APF)
```

Rewards the RL agent for choosing headings that **agree** with
APF's safe gradient direction.
        """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3  —  Benchmarks
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Method Comparison — Real Evaluation Data")
    st.markdown(
        f"**{EXP_CONFIG.get('n_eval_episodes', '?')} evaluation episodes**  ·  "
        f"seed {EXP_CONFIG.get('eval_seed', '?')}  ·  "
        f"device: {EXP_CONFIG.get('device', '?')}"
    )

    if REAL_RESULTS:
        st.plotly_chart(build_benchmarks(REAL_RESULTS), use_container_width=True)

        st.markdown("#### Performance Profiles")
        st.plotly_chart(build_radar(REAL_RESULTS), use_container_width=True)

        st.markdown("#### Detailed Results Table")
        df = pd.DataFrame(REAL_RESULTS)
        show_cols = [c for c in [
            "method", "success_rate", "mean_energy_kj",
            "collision_rate", "mean_steps", "n_episodes",
        ] if c in df.columns]
        df_show = df[show_cols].copy()
        df_show["success_rate"]  = df_show["success_rate"].apply(
            lambda x: f"{x*100:.1f}%")
        df_show["mean_energy_kj"] = df_show["mean_energy_kj"].apply(
            lambda x: f"{x:.4f}")
        if "collision_rate" in df_show.columns:
            df_show["collision_rate"] = df_show["collision_rate"].apply(
                lambda x: f"{x*100:.1f}%")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.info(
            "💡 These are **real** evaluation results from your experiment run "
            "(`results/results_*.json`), not paper target values."
        )
    else:
        st.warning("No evaluation results found in `results/` directory.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4  —  Fusion Dynamics
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Fusion Dynamics — Last Simulated Episode")
    st.markdown(
        "How α, alignment, and obstacle distance evolve over a single episode"
    )

    papf_key = "PPO-APF (adaptive α)"
    if (st.session_state.traj_results
            and papf_key in st.session_state.traj_results
            and st.session_state.traj_results[papf_key].get("alphas")):

        papf = st.session_state.traj_results[papf_key]
        st.plotly_chart(build_fusion_dynamics(papf), use_container_width=True)

        a_arr = np.array(papf["alphas"])
        al_arr = np.array(papf.get("alignments", []))

        mc = st.columns(4)
        mc[0].metric("Mean α",          f"{a_arr.mean():.3f}")
        mc[1].metric("Time in Safety Zone",
                     f"{(a_arr <= ALPHA_NEAR + 0.01).mean()*100:.1f}%")
        mc[2].metric("Time in RL Zone",
                     f"{(a_arr >= ALPHA_FAR  - 0.01).mean()*100:.1f}%")
        if len(al_arr):
            mc[3].metric("Mean Alignment", f"{al_arr.mean():.3f}")
    else:
        st.info(
            "👈 Run a simulation with **PPO-APF (adaptive α)** selected "
            "to see live fusion dynamics."
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5  —  Architecture
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Network Architecture  &  Hyperparameters")

    c_dqn, c_ppo = st.columns(2)

    with c_dqn:
        st.markdown("""
#### 🧠 DQN  (Deep Q-Network)

```
Input(16) → Linear(256) → ReLU
           → Linear(256) → ReLU
           → Linear(15)       ← Q-values
```

| Parameter | Value |
|-----------|-------|
| Learning rate | 1 × 10⁻⁴ |
| Replay buffer | 100 000 |
| Batch size | 128 |
| γ (discount) | 0.99 |
| ε decay | 1.0 → 0.05 (30%) |
| Target update | Every 1 000 steps |
| Actions | **Discrete(15)** |
        """)

    with c_ppo:
        st.markdown("""
#### 🧠 PPO  (Proximal Policy Optimisation)

```
Shared: Input(16) → Linear(256) → Tanh
                   → Linear(256) → Tanh
Actor:  → Linear(3) + log_std  ← Gaussian
Critic: → Linear(1)            ← V(s)
```

| Parameter | Value |
|-----------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Rollout | 2 048 × 4 envs |
| Batch size | 256 |
| γ (discount) | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Actions | **Box(3)** continuous |
        """)

    st.divider()

    st.markdown("""
#### 🌊 Environment Specification

| Property | Value |
|----------|-------|
| Ocean volume | 100 × 100 × 50 m |
| Obstacles | 8 dynamic spheres (r = 4 m, drift ≤ 0.2 m/s) |
| Ocean current | 0.1 – 0.5 m/s, slowly time-varying |
| AUV speed | 0.5 – 2.0 m/s |
| Energy model | Quadratic drag:  P = 2.5 · v² · Δt |
| Energy budget | 500 J per episode |
| Goal radius | 3.0 m |
| State dim | 16  (pos, vel, energy, 8×obs_dist, goal_dist) |
| Max steps | 500 |

#### 📐 Reward Function

```
R = 100·[reached_goal] − 50·[collided] − 0.1·d_goal
    − 0.01·ΔE  +  5.0·cos(Δψ_RL_APF)
```
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7F8C8D; font-size:0.85rem; padding:0.8rem 0;">
  <b>Adaptive RL-APF Framework</b>  ·  UVCE Undergraduate Research  ·
  Underwater Robotics Conference 2026<br/>
  All data is <em>real</em> — generated from trained models and physics
  simulation, not synthetic
</div>
""", unsafe_allow_html=True)
