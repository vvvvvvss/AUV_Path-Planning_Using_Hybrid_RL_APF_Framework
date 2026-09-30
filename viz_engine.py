"""
viz_engine.py
=============
Scientific figure engine matching the reference paper figure style:
  - Ocean current vector field (quiver arrows, colored by speed)
  - Obstacle danger/repulsive field (red contour heatmap)
  - Path trajectories (colored line with waypoint markers)
  - Hybrid path colored by α with colorbar
  - Multi-subplot comparison layout
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Style defaults (scientific paper look) ────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 10,
    "axes.titlesize"  : 11,
    "axes.labelsize"  : 10,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "axes.linewidth"  : 0.8,
    "figure.dpi"      : 130,
    "savefig.dpi"     : 200,
    "savefig.bbox"    : "tight",
})

METHOD_COLORS = {
    "PPO-APF (adaptive α)": "#1A1A2E",
    "Pure PPO"             : "#C0392B",
    "APF (classical)"      : "#1565C0",
}
METHOD_LABELS = {
    "PPO-APF (adaptive α)": "PPO-APF Path",
    "Pure PPO"             : "Pure PPO Path",
    "APF (classical)"      : "APF Path",
}
PANEL_LETTERS = ['(a)', '(b)', '(c)', '(d)']

X_MAX, Y_MAX, OBS_RADIUS = 100., 100., 4.0


def _build_fields(obstacles, current_vec, nx=22, ny=22):
    """
    Compute current vector field and obstacle danger field on a grid.
    Matches Image 2 & 3 style: arrows coloured by speed, red heatmap for danger.
    """
    gx = np.linspace(0, X_MAX, nx)
    gy = np.linspace(0, Y_MAX, ny)
    GX, GY = np.meshgrid(gx, gy)

    # Base current (uniform, from simulation)
    U = np.full_like(GX, current_vec[0])
    V = np.full_like(GY, current_vec[1])

    # Vortex disturbance around each obstacle (like Image 2)
    for obs_pos, obs_r in obstacles:
        r = np.sqrt((GX - obs_pos[0])**2 + (GY - obs_pos[1])**2) + 1e-6
        theta = np.arctan2(GY - obs_pos[1], GX - obs_pos[0])
        strength = 0.35 * np.exp(-r / 14)
        U += strength * (-np.sin(theta))
        V += strength *   np.cos(theta)

    speed = np.sqrt(U**2 + V**2)

    # Danger field: Gaussian around each obstacle
    DANGER = np.zeros_like(GX)
    for obs_pos, obs_r in obstacles:
        r = np.sqrt((GX - obs_pos[0])**2 + (GY - obs_pos[1])**2)
        sigma = obs_r * 2.8
        DANGER += np.exp(-r**2 / (2 * sigma**2))
    DANGER = np.clip(DANGER, 0, 1)

    return GX, GY, U, V, speed, DANGER


def _draw_ax(ax, method, traj, alphas, obstacles, current_vec, letter):
    """Render one subplot panel."""
    GX, GY, U, V, speed, DANGER = _build_fields(obstacles, current_vec)

    # ── Danger heatmap (red, like Image 2) ───────────────────────────────
    cf = ax.contourf(GX, GY, DANGER, levels=30,
                     cmap='Reds', alpha=0.38, vmin=0, vmax=1.0)

    # ── Current field (arrows coloured by speed, like Images 2 & 3) ──────
    skip = 2
    spd_s = speed[::skip, ::skip]
    qv = ax.quiver(
        GX[::skip, ::skip], GY[::skip, ::skip],
        U[::skip, ::skip],  V[::skip, ::skip],
        spd_s,
        cmap='cool', alpha=0.55,
        scale=9.0, width=0.003,
        pivot='mid',
    )

    # ── Obstacles ─────────────────────────────────────────────────────────
    for obs_pos, obs_r in obstacles:
        circ = plt.Circle(obs_pos[:2], obs_r,
                           color='#555555', alpha=0.75, zorder=4)
        ax.add_patch(circ)
        ring = plt.Circle(obs_pos[:2], obs_r + 5,
                           fill=False, edgecolor='#CC2222',
                           alpha=0.22, linestyle='--', linewidth=0.7)
        ax.add_patch(ring)

    # ── Path ──────────────────────────────────────────────────────────────
    if traj is not None and len(traj) > 1:
        traj_xy = np.array(traj)[:, :2] if traj.ndim == 2 and traj.shape[1] >= 2 \
                  else np.array(traj)

        if method == "PPO-APF (adaptive α)" and alphas and len(alphas) >= 2:
            # Colour path by α (the paper's innovation made visible)
            pts    = traj_xy.reshape(-1, 1, 2)
            segs   = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n_seg  = len(segs)
            a_vals = np.array(
                alphas[:n_seg] + [alphas[-1]] * max(0, n_seg - len(alphas))
            )
            lc = LineCollection(
                segs, cmap='RdYlGn',
                norm=Normalize(0.28, 0.92),
                linewidth=2.8, zorder=6,
            )
            lc.set_array(a_vals[:n_seg])
            ax.add_collection(lc)
            # Colorbar (α weight)
            div  = make_axes_locatable(ax)
            cax  = div.append_axes("right", size="4%", pad=0.06)
            cb   = plt.colorbar(lc, cax=cax)
            cb.set_label('α  (RL weight)', fontsize=8)
            cb.set_ticks([0.3, 0.5, 0.7, 0.9])
            cb.ax.tick_params(labelsize=7)
            cb.ax.yaxis.set_label_position('right')
        else:
            col = METHOD_COLORS.get(method, 'black')
            ax.plot(traj_xy[:, 0], traj_xy[:, 1],
                    color=col, linewidth=2.2,
                    label=METHOD_LABELS.get(method, method),
                    zorder=6)

        # Waypoint markers (yellow circles, like Image 2)
        step = max(1, len(traj_xy) // 7)
        ax.scatter(traj_xy[::step, 0], traj_xy[::step, 1],
                   s=35, facecolors='yellow', edgecolors='black',
                   linewidths=0.7, zorder=7)

        # Direction arrows along path
        col = METHOD_COLORS.get(method, 'black')
        for k in range(0, len(traj_xy)-3, max(1, len(traj_xy)//5)):
            dx = traj_xy[k+1, 0] - traj_xy[k, 0]
            dy = traj_xy[k+1, 1] - traj_xy[k, 1]
            norm = np.sqrt(dx**2+dy**2)+1e-9
            ax.annotate('',
                xy=(traj_xy[k,0]+dx/norm*4, traj_xy[k,1]+dy/norm*4),
                xytext=(traj_xy[k,0], traj_xy[k,1]),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=1.4, mutation_scale=10),
                zorder=8,
            )

    # ── Start / Goal ──────────────────────────────────────────────────────
    ax.scatter([5], [5], s=90, c='#F39C12', marker='o',
               edgecolors='black', linewidths=0.8, zorder=9, label='Start')
    ax.scatter([90], [90], s=130, c='#C0392B', marker='*',
               edgecolors='black', linewidths=0.8, zorder=9, label='Goal')
    ax.text(90, 93, 'Goal', ha='center', va='bottom', fontsize=8, color='#C0392B')
    ax.text(5,   2, 'Start', ha='center', va='bottom', fontsize=8, color='#7F6000')

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xlim(0, X_MAX); ax.set_ylim(0, Y_MAX)
    ax.set_xlabel('X-Coordinate (m)', fontsize=9)
    ax.set_ylabel('Y-Coordinate (m)', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(direction='in', length=3)

    # Panel letter + title (like Image 3)
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')
    ax.set_title(method, fontsize=10, fontweight='bold', pad=6)

    # Current speed colorbar (shared concept from Image 3)
    div2 = make_axes_locatable(ax)
    if method != "PPO-APF (adaptive α)" or not alphas:
        cax2 = div2.append_axes("right", size="4%", pad=0.06)
        cb2  = plt.colorbar(qv, cax=cax2)
        cb2.set_label('Current speed (m/s)', fontsize=8)
        cb2.ax.tick_params(labelsize=7)


def make_comparison_figure(
    trajectories: dict,
    alphas: list,
    alignments: list,
    obstacles: list,    # list of (pos_array, radius)
    current_vec: np.ndarray,
    episode_results: dict = None,
) -> plt.Figure:
    """
    Build the full comparison figure (Image 3 style).
    trajectories: dict method → np.ndarray (N,2) or (N,3)
    """
    n = len(trajectories)
    if n == 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, 'Run an episode to see the comparison',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='gray')
        return fig

    fig_w = max(6 * n, 10)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 6.5),
                              facecolor='white')
    if n == 1:
        axes = [axes]

    for i, (method, traj) in enumerate(trajectories.items()):
        a_vals = alphas if method == "PPO-APF (adaptive α)" else []
        _draw_ax(axes[i], method, traj, a_vals, obstacles, current_vec,
                 PANEL_LETTERS[i])

        # Results annotation box
        if episode_results and method in episode_results:
            r = episode_results[method]
            status = "✓ SUCCESS" if r.get("success") else "✗ DID NOT REACH"
            col    = '#1A7A3A' if r.get("success") else '#8B0000'
            axes[i].text(
                0.98, 0.02,
                f"{status}\n{r.get('energy_kj',0):.3f} kJ",
                transform=axes[i].transAxes,
                ha='right', va='bottom', fontsize=8.5,
                color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=col, alpha=0.85),
            )

    fig.suptitle(
        'AUV Path Planning Comparison  ·  Same environment, seed-matched obstacles',
        fontsize=11, y=1.01, fontweight='normal', color='#333333',
    )
    plt.tight_layout()
    return fig


def make_alpha_dynamics_figure(
    alphas: list,
    alignments: list,
    d_nearest_vals: list = None,
) -> plt.Figure:
    """
    Figure 6-style: α and alignment over episode.
    Matches the paper's fusion dynamics visualisation.
    """
    if not alphas:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'No hybrid episode data yet',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return fig

    steps = np.arange(len(alphas))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor='white')

    # ── Panel 1: α over time ─────────────────────────────────────────────
    ax = axes[0]
    a_arr = np.array(alphas)
    # Colour-fill by zone
    ax.fill_between(steps, 0.28, 0.55, alpha=0.08, color='#C0392B', label='Safety zone')
    ax.fill_between(steps, 0.55, 0.75, alpha=0.06, color='#F39C12', label='Transition')
    ax.fill_between(steps, 0.75, 0.95, alpha=0.06, color='#27AE60', label='AI zone')
    ax.plot(steps, a_arr, color='#1A1A2E', linewidth=1.8, zorder=5, label='α (this episode)')
    ax.axhline(0.3, color='#C0392B', linestyle=':', linewidth=0.9, alpha=0.7)
    ax.axhline(0.7, color='#F39C12', linestyle=':', linewidth=0.9, alpha=0.7)
    ax.axhline(0.9, color='#27AE60', linestyle=':', linewidth=0.9, alpha=0.7)
    ax.text(steps[-1]*0.98, 0.32, 'α=0.3  APF leads',  ha='right', va='bottom', fontsize=7.5, color='#C0392B')
    ax.text(steps[-1]*0.98, 0.72, 'α=0.7  shared',     ha='right', va='bottom', fontsize=7.5, color='#F39C12')
    ax.text(steps[-1]*0.98, 0.92, 'α=0.9  RL leads',   ha='right', va='bottom', fontsize=7.5, color='#27AE60')
    ax.set_ylim(0.25, 0.97); ax.set_xlim(0, max(steps[-1], 1))
    ax.set_xlabel('Timestep', fontsize=9); ax.set_ylabel('α  (RL weight)', fontsize=9)
    ax.set_title('(a)  Adaptive α over the episode', fontsize=10, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.15, linestyle='--'); ax.tick_params(direction='in', length=3)

    # ── Panel 2: alignment (cos Δyaw) ────────────────────────────────────
    ax2 = axes[1]
    if alignments:
        al_arr = np.array(alignments)
        ax2.fill_between(steps[:len(al_arr)], 0, al_arr,
                          where=al_arr >= 0, alpha=0.18, color='#27AE60', label='Cooperative (>0)')
        ax2.fill_between(steps[:len(al_arr)], 0, al_arr,
                          where=al_arr <  0, alpha=0.18, color='#C0392B', label='Opposing  (<0)')
        ax2.plot(steps[:len(al_arr)], al_arr, color='#8E44AD',
                  linewidth=1.6, zorder=5, label='cos(Δyaw_RL_APF)')
        ax2.axhline(0, color='gray', linewidth=0.7)
    ax2.set_ylim(-1.05, 1.05); ax2.set_xlim(0, max(steps[-1], 1))
    ax2.set_xlabel('Timestep', fontsize=9)
    ax2.set_ylabel('cos(Δyaw_RL_APF)', fontsize=9)
    ax2.set_title('(b)  RL–APF alignment bonus  (shapes cooperative behaviour)',
                   fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=7.5)
    ax2.grid(True, alpha=0.15, linestyle='--'); ax2.tick_params(direction='in', length=3)

    plt.tight_layout()
    return fig