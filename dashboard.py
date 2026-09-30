"""
dashboard.py — Open Day Streamlit Dashboard
============================================
Runs your actual paper code. Uses:
  envs/auv_environment.py     real 3D ocean simulation
  planning/apf_module.py      real APF planner
  planning/hybrid_fusion.py   real adaptive α fusion
  results/models/*.zip        your trained SB3 models
  viz_engine.py               scientific figure engine

Run:
  cd /path/to/auv
  streamlit run dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AUV Path Planning — Open Day",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #FAFBFC !important; }
.stMetric [data-testid="stMetricValue"] {
    font-size: 22px !important; font-family: 'Courier New', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports from your actual codebase ────────────────────────────────────────
from envs.auv_environment import AUVEnvironment, X_MAX, Y_MAX, ENERGY_BUDGET
from planning.apf_module import APFPlanner, run_apf_episode
from planning.hybrid_fusion import HybridFusionEnv, compute_alpha, ALPHA_FAR
from viz_engine import make_comparison_figure, make_alpha_dynamics_figure


@st.cache_resource
def load_models():
    from stable_baselines3 import PPO
    models = {}
    paths = {
        "PPO-APF (adaptive α)": "results/models/ppo_apf_demo.zip",
        "Pure PPO"             : "results/models/ppo_pure_demo.zip",
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = PPO.load(path, device="cpu")
    return models

MODELS = load_models()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "history": [], "ep_count": 0,
    "last_trajs": {}, "last_alphas": [], "last_aligns": [],
    "last_obs": [], "last_current": np.zeros(3),
    "last_ep_results": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")

    method = st.selectbox(
        "Method",
        ["Compare all 3", "PPO-APF (adaptive α)", "Pure PPO", "APF (classical)"],
    )
    seed = st.number_input("Episode seed", 0, 9999, 42,
                            help="Same seed = same obstacles for fair comparison")
    speed = st.slider("Steps per render", 1, 25, 10)

    st.markdown("---")
    run_btn   = st.button("▶  Run episode",   use_container_width=True, type="primary")
    reset_btn = st.button("↺  Reset stats",   use_container_width=True)

    st.markdown("---")
    st.markdown("**Cumulative session results**")
    h = st.session_state.history
    if h:
        by_m = {}
        for r in h:
            m = r["method"]
            by_m.setdefault(m, {"w": 0, "n": 0, "e": []})
            by_m[m]["n"] += 1
            if r["success"]: by_m[m]["w"] += 1
            by_m[m]["e"].append(r["energy_kj"])
        for m, d in by_m.items():
            sr = 100 * d["w"] / d["n"]
            eg = np.mean(d["e"])
            st.metric(m, f"{sr:.0f}% success", f"{eg:.3f} kJ avg energy")
    else:
        st.caption("No episodes yet — press ▶ Run episode")

    st.markdown("---")
    st.markdown(
        "**Running:**\n"
        "- `envs/auv_environment.py`\n"
        "- `planning/apf_module.py`\n"
        "- `planning/hybrid_fusion.py`\n"
        "- `stable_baselines3` PPO\n"
    )

if reset_btn:
    for k in ["history","ep_count","last_trajs","last_alphas","last_aligns",
              "last_obs","last_current","last_ep_results"]:
        st.session_state[k] = [] if isinstance(st.session_state[k], list) else \
                               {} if isinstance(st.session_state[k], dict) else \
                               np.zeros(3) if isinstance(st.session_state[k], np.ndarray) else 0
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "## Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning"
)
st.markdown(
    "**UVCE · Undergraduate Research · Underwater Robotics Conference 2026**  "
    "— all plots generated live from your actual simulation code"
)
st.divider()

# ── Episode runner ────────────────────────────────────────────────────────────
def _run_single(method_name, seed_val, obs_snapshot, current_vec, steps_per_frame,
                plot_placeholder):
    """Run one method episode, updating the plot every steps_per_frame steps."""
    env = AUVEnvironment(mode="continuous", seed=seed_val)
    env.reset()
    # Fix obstacle positions to match the snapshot
    for i, (pos, _) in enumerate(obs_snapshot):
        env.obstacles[i].pos = pos.copy()
    env.current_vec = current_vec.copy()

    apf = APFPlanner(seed=seed_val)

    if method_name == "PPO-APF (adaptive α)" and method_name in MODELS:
        wrapped = HybridFusionEnv(env, apf, alpha_mode="adaptive")
        wrapped.reset()
        for i, (pos, _) in enumerate(obs_snapshot):
            wrapped.env.obstacles[i].pos = pos.copy()
        wrapped.env.current_vec = current_vec.copy()
        agent_env = wrapped
        def get_obs(): return wrapped.env._get_obs()
        def step(a): return wrapped.step(a)
        base_env = wrapped.env
        model = MODELS[method_name]
    elif method_name == "Pure PPO" and method_name in MODELS:
        agent_env = env
        def get_obs(): return env._get_obs()
        def step(a): return env.step(a)
        base_env = env
        model = MODELS[method_name]
        apf = None
        wrapped = None
    else:  # APF
        apf.reset()
        agent_env = env
        base_env = env
        model = None
        wrapped = None

    traj    = [base_env.auv_pos[:2].copy()]
    alphas  = []
    aligns  = []
    done    = False
    step_n  = 0

    while not done:
        for _ in range(steps_per_frame):
            if done: break
            if model:
                obs_a = get_obs()
                action, _ = model.predict(obs_a, deterministic=True)
                _, _, te, tr, info = step(action)
                if method_name == "PPO-APF (adaptive α)":
                    alphas.append(info.get("alpha", ALPHA_FAR))
                    aligns.append(info.get("alignment", 0.0))
            else:
                action = apf.compute_action(base_env)
                _, _, te, tr, info = env.step(action)
            traj.append(base_env.auv_pos[:2].copy())
            done = te or tr
            step_n += 1

        # Live plot update
        cur_trajs = {method_name: np.array(traj)}
        cur_obs   = [(o.pos.copy(), o.radius) for o in base_env.obstacles]
        fig = make_comparison_figure(
            cur_trajs, alphas, aligns, cur_obs,
            base_env.current_vec.copy(), episode_results=None
        )
        plot_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)

    return (np.array(traj), alphas, aligns, info,
            base_env.episode_energy / 1000., base_env.episode_collided,
            [(o.pos.copy(), o.radius) for o in base_env.obstacles],
            base_env.current_vec.copy())


def run_episode(method_choice, seed_val, steps_per_frame):
    """Orchestrate single or multi-method episode."""
    st.session_state.ep_count += 1

    # Determine methods to run
    if method_choice == "Compare all 3":
        methods = ["APF (classical)", "Pure PPO", "PPO-APF (adaptive α)"]
    else:
        methods = [method_choice]

    # First, run APF to establish the shared obstacle layout
    ref_env = AUVEnvironment(seed=seed_val)
    ref_env.reset()
    obs_snapshot = [(o.pos.copy(), o.radius) for o in ref_env.obstacles]
    current_vec  = ref_env.current_vec.copy()

    all_trajs, all_alphas, all_aligns, ep_results = {}, [], [], {}
    plot_ph = st.empty()

    for m in methods:
        with st.spinner(f"Running {m}…"):
            traj, alphas, aligns, info, e_kj, collided, final_obs, final_cur = \
                _run_single(m, seed_val, obs_snapshot, current_vec, steps_per_frame, plot_ph)

        all_trajs[m] = traj
        ep_results[m] = {
            "success"  : bool(info.get("reached_goal", False)),
            "energy_kj": e_kj,
            "collided" : collided,
            "steps"    : info.get("step", len(traj)),
        }
        st.session_state.history.append({
            "method": m, "episode": st.session_state.ep_count, "seed": seed_val,
            **ep_results[m],
        })
        if m == "PPO-APF (adaptive α)":
            all_alphas = alphas
            all_aligns = aligns

    # Final combined figure
    final_obs_snap = [(o.pos.copy(), o.radius) for o in ref_env.obstacles]
    fig_final = make_comparison_figure(
        all_trajs, all_alphas, all_aligns,
        obs_snapshot, current_vec, ep_results,
    )
    plot_ph.pyplot(fig_final, use_container_width=True)
    plt.close(fig_final)

    # Store for re-display
    st.session_state.last_trajs      = all_trajs
    st.session_state.last_alphas     = all_alphas
    st.session_state.last_aligns     = all_aligns
    st.session_state.last_obs        = obs_snapshot
    st.session_state.last_current    = current_vec
    st.session_state.last_ep_results = ep_results
    return fig_final


# ── Main display area ─────────────────────────────────────────────────────────
main_plot = st.empty()

# Show last figure while idle
if not run_btn and st.session_state.last_trajs:
    fig_idle = make_comparison_figure(
        st.session_state.last_trajs,
        st.session_state.last_alphas,
        st.session_state.last_aligns,
        st.session_state.last_obs,
        st.session_state.last_current,
        st.session_state.last_ep_results,
    )
    main_plot.pyplot(fig_idle, use_container_width=True)
    plt.close(fig_idle)
elif not run_btn:
    main_plot.info("← Select a method and press **▶ Run episode** to start the simulation.")

if run_btn:
    with main_plot.container():
        run_episode(method, int(seed), int(speed))

# ── α dynamics figure ─────────────────────────────────────────────────────────
if st.session_state.last_alphas:
    st.divider()
    st.markdown("### Fusion dynamics — last episode")
    fig_dyn = make_alpha_dynamics_figure(
        st.session_state.last_alphas,
        st.session_state.last_aligns,
    )
    st.pyplot(fig_dyn, use_container_width=True)
    plt.close(fig_dyn)

# ── Episode results table ─────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.markdown("### Episode history — from your actual simulation")
    import pandas as pd
    df = pd.DataFrame(st.session_state.history)
    df["success"]  = df["success"].map({True: "✓ YES", False: "✗ No"})
    df["collided"] = df["collided"].map({True: "yes", False: "no"})
    df["energy_kj"] = df["energy_kj"].round(4)
    st.dataframe(
        df[["episode","method","success","energy_kj","collided","steps","seed"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Metrics row ───────────────────────────────────────────────────────────────
if st.session_state.last_ep_results:
    st.divider()
    st.markdown("### Last episode — head-to-head")
    cols = st.columns(len(st.session_state.last_ep_results))
    for col, (m, r) in zip(cols, st.session_state.last_ep_results.items()):
        with col:
            st.markdown(f"**{m}**")
            st.metric("Result",     "✓ Success" if r["success"] else "✗ Failed")
            st.metric("Energy",     f"{r['energy_kj']:.3f} kJ")
            st.metric("Steps",      str(r["steps"]))
            st.metric("Collisions", "Yes" if r["collided"] else "No")