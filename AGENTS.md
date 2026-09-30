# AGENTS.md

## Project context

This repository implements an adaptive RL-APF framework for energy-efficient AUV path planning. The code is organized around four main areas:

- `agents/` — training, evaluation, and callback logic for DQN and PPO
- `envs/` — the 3D Gymnasium environment and obstacle/current model
- `planning/` — classical APF and hybrid fusion logic, including adaptive alpha scheduling
- `results/` — saved models, TensorBoard logs, and evaluation JSON/CSV artifacts

This repo is research code: prioritize reproducibility, real data, and consistency with the existing paper-spec defaults.

## Working conventions for AI coding agents

### 1. Prefer the real project pipeline

When changing agent behavior, follow the project’s established flow instead of creating synthetic shortcuts:

- Train/evaluate through the existing agent modules in `agents/dqn_agent.py` and `agents/ppo_agent.py`
- Use the shared callback metrics from `agents/training_callbacks.py`
- Persist outputs under `results/models/` and `results/`
- Keep dashboard and visualization code grounded in real model outputs and JSON results, not hardcoded placeholders

### 2. Respect the model architecture contracts

The repository uses Stable-Baselines3 models with specific conventions:

- `DQN_DEFAULTS` in `agents/dqn_agent.py` defines the discrete-control pipeline
- `PPO_DEFAULTS` in `agents/ppo_agent.py` defines the continuous-control pipeline
- evaluation should use `model.predict(obs, deterministic=True)` to match paper reporting and avoid exploration noise
- metric keys should remain compatible with the callback pipeline and the results JSON schema

### 3. Keep data and naming consistent

Project-specific names matter here:

- `method` names like `Pure PPO`, `Pure DQN`, `PPO-APF (adaptive α)`, and `APF (classical)` are used across dashboards and plots
- result fields include metrics such as `success_rate`, `mean_energy_kj`, `collision_rate`, and `mean_steps`
- saved model files should typically use the existing naming pattern, such as `ppo_auv_final.zip`, `ppo_apf_best.zip`, and `dqn_auv_final.zip`

### 4. Avoid broad rewrites without a narrow root cause

This project is not a generic ML template. Changes should be minimal and respect the current runtime assumptions:

- keep the environment API compatible with `AUVEnvironment` and `AUVEnvironmentDiscrete`
- preserve the episodic info payload used by `training_callbacks.py`
- do not add dummy metrics or invented benchmark values when the real results already exist in `results/`

## Key commands

Use the project’s built-in entry points when possible:

```bash
# Quick paper-style smoke test
python experiments/main_experiment.py --fast

# Single-method training/evaluation
python experiments/main_experiment.py --method ppo_apf --device cuda

# Run the 3D Open Day dashboard
streamlit run dashboard_3d.py

# Generate the publication figures
python visualize.py
```

## Useful files to read first

- [README.md](README.md)
- [agents/dqn_agent.py](agents/dqn_agent.py)
- [agents/ppo_agent.py](agents/ppo_agent.py)
- [agents/training_callbacks.py](agents/training_callbacks.py)
- [envs/auv_environment.py](envs/auv_environment.py)
- [planning/hybrid_fusion.py](planning/hybrid_fusion.py)
- [dashboard_3d.py](dashboard_3d.py)

## Agent-specific guidance

When working in the agent stack:

- prefer editing the shared configuration defaults before creating ad hoc overrides
- keep training callbacks and result summaries aligned with the paper’s metrics
- preserve compatibility with TensorBoard logging in `results/tb_logs/`
- when evaluating or comparing models, use the same seed logic described in the repo’s reproducibility section

## Notes for dashboard and visualization work

The implementation plan for the Open Day dashboard is intentionally grounded in real research artifacts. For any agent-level or evaluation task tied to the dashboard:

- use real model checkpoints from `results/models/`
- use real JSON results from `results/results_*.json`
- prefer derived values from the environment and training callbacks over hardcoded numbers
- treat the implementation plan as a design artifact, not as a license to fabricate data

## When in doubt

Follow the repo’s explicit patterns rather than generic ML conventions:

- model defaults are defined centrally
- training metrics are logged in a callback layer
- evaluation is deterministic and seed-based
- visualization is expected to reflect real experiment outputs
