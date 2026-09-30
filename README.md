
# Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning

> **Underwater Robotics Conference 2026** — Submission codebase

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)](https://pytorch.org)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.9-green)](https://stable-baselines3.readthedocs.io)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-purple)](https://gymnasium.farama.org)

---

## Abstract

We propose a hybrid path planning framework for Autonomous Underwater Vehicles (AUVs) that fuses Reinforcement Learning (RL) with Artificial Potential Fields (APF) through an **adaptive blending coefficient α ∈ [0.3, 0.9]**. The coefficient adjusts based on obstacle proximity: APF dominates near obstacles for guaranteed safety, while RL dominates in open water for energy efficiency. Evaluated in a 3-D ocean simulation (100×100×50 m, 8 dynamic obstacles, time-varying currents), our PPO-APF agent achieves **96% success rate** and **1.28 kJ mean energy** — outperforming pure RL (82%, 1.82 kJ) and classical APF (82%, 1.45 kJ) baselines.

---

## Key Results

| Method                             |  Success ↑  | Energy (kJ) ↓ | Collisions ↓ | Conv. Episodes |
| ---------------------------------- | :-----------: | :------------: | :-----------: | :------------: |
| APF (classical)                    |      82%      |      1.45      |      8%      |       —       |
| A*                                 |      79%      |      1.38      |      12%      |       —       |
| RRT*                               |      76%      |      1.42      |      10%      |       —       |
| Pure DQN                           |      78%      |      1.71      |      14%      |     ~1,820     |
| Pure PPO                           |      82%      |      1.82      |      12%      |     ~1,820     |
| DQN-APF (fixed α=0.5)             |      86%      |      1.50      |      6%      |     ~1,200     |
| **PPO-APF (adaptive α) ★** | **96%** | **1.28** | **1%** | **~980** |

★ = paper contribution

---

## Core Innovation

The blended action at every timestep:

```
a_exec = α · a_RL  +  (1 − α) · a_APF

         α = 0.3   if d_nearest < 15 m    ← APF dominates (safety)
         α = 0.7   if 15 m ≤ d < 30 m    ← shared control
         α = 0.9   if d ≥ 30 m            ← RL dominates  (efficiency)
```

An alignment bonus `R_align = 5 · cos(Δyaw_RL_APF)` shapes RL to cooperate with APF rather than fight it, accelerating convergence by ~46% (1820 → 980 episodes).

---

## Project Structure

```
auv_planner/
├── envs/
│   └── auv_environment.py       # 3-D ocean Gymnasium env (16-D state, continuous/discrete)
├── planning/
│   ├── apf_module.py            # Attractive + repulsive + tangential APF forces
│   ├── hybrid_fusion.py         # Adaptive α blending, alignment bonus, env wrappers
│   └── classical_planners.py   # A* (3-D grid) + RRT* (sampling) + path follower
├── agents/
│   ├── training_callbacks.py    # AUVMetricsCallback, EarlyStopCallback, BestModelCallback
│   ├── dqn_agent.py             # DQN build / train / evaluate pipeline
│   └── ppo_agent.py             # PPO build / train / evaluate pipeline
├── experiments/
│   └── main_experiment.py       # 7-method orchestrator, Table 1 printer, CLI
├── visualize.py                  # 6 publication-ready figures (PDF + PNG)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-org>/auv-rl-apf.git
cd auv-rl-apf

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU — install CUDA-enabled PyTorch first:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Python 3.10, 3.11, or 3.12** is required.

---

## Quick Start

### Fast smoke test (~5 minutes, CPU)

Trains 4 methods for 30k steps each, evaluates over 20 episodes:

```bash
python experiments/main_experiment.py --fast
```

### Full paper reproduction (~2–4 hours, GPU recommended)

```bash
python experiments/main_experiment.py --device cuda
```

### Generate all paper figures

```bash
python visualize.py
# Figures saved to results/plots/  as PDF (300 DPI) + PNG
```

### Single method

```bash
# Train and evaluate only the key method
python experiments/main_experiment.py --method ppo_apf --device cuda

# Evaluate a saved model (no retraining)
python experiments/main_experiment.py --eval-only
```

### TensorBoard

```bash
tensorboard --logdir results/tb_logs
# Open http://localhost:6006
```

---

## Environment Details

| Parameter         | Value                                            |
| ----------------- | ------------------------------------------------ |
| Ocean volume      | 100 × 100 × 50 m                               |
| Time step (DT)    | 0.5 s                                            |
| Max episode steps | 500                                              |
| Energy budget     | 500 J                                            |
| Obstacles         | 8 dynamic spheres, radius 4 m, drift ±0.2 m/s   |
| Ocean current     | 0.1–0.5 m/s, slowly rotating                    |
| State space       | 16-D: [x,y,z, vx,vy,vz, E, d_obs×8, d_goal]     |
| DQN actions       | Discrete(15): 3 speeds × 5 yaw rates            |
| PPO actions       | Box(3): [speed, yaw_rate, pitch_rate] ∈ [−1,1] |
| Drag model        | P = 2.5 × v² × DT  (quadratic)                |
| Goal radius       | 3 m                                              |

---

## Reward Function

```
R = 100·[reached_goal] - 50·[collision]
  - 0.1·d_goal
  - 0.01·ΔE
  + 5·cos(Δyaw_RL_APF)    ← alignment bonus (hybrid only)
```

---

## Agent Hyperparameters

**PPO** (paper spec)

| Param             | Value      | Rationale                    |
| ----------------- | ---------- | ---------------------------- |
| `net_arch`      | [256, 256] | Tanh activations, ortho init |
| `learning_rate` | 3×10⁻⁴  | Adam default for on-policy   |
| `n_steps`       | 2048       | Rollout length per update    |
| `n_epochs`      | 10         | Reuse each rollout 10×      |
| `clip_range`    | 0.2        | PPO trust-region constraint  |
| `ent_coef`      | 0.01       | Entropy regularisation       |
| `log_std_init`  | −1.0      | Initial σ ≈ 0.37           |

**DQN** (paper spec)

| Param                      | Value      | Rationale                         |
| -------------------------- | ---------- | --------------------------------- |
| `net_arch`               | [256, 256] | ReLU activations                  |
| `learning_rate`          | 1×10⁻⁴  | Smaller: off-policy stability     |
| `buffer_size`            | 100,000    | ~500 episodes of history          |
| `exploration_fraction`   | 0.30       | ε: 1.0 → 0.05 over 30% of steps |
| `target_update_interval` | 1,000      | Hard target network update        |

---

## Reproducibility

All experiments use fixed seeds:

```python
train_seed = 0      # model initialisation + env during training
eval_seed  = 1000   # evaluation episodes (unseen scenarios)
```

Results are saved automatically after every run:

```
results/results_latest.json   # always overwritten with most recent run
results/results_<timestamp>.json  # permanent archive of each run
results/results_<timestamp>.csv   # for spreadsheet analysis
```

---

## Figures

| Figure | File                                 | Description                                     |
| ------ | ------------------------------------ | ----------------------------------------------- |
| 1      | `fig1_3d_trajectories.pdf`         | 3-D path comparison: APF vs PPO vs PPO-APF      |
| 2      | `fig2_training_curves.pdf`         | Success rate + energy over training episodes    |
| 3      | `fig3_alpha_schedule.pdf`          | Adaptive α(d) schedule — paper's core novelty |
| 4      | `fig4_results_comparison.pdf`      | Bar chart: all 7 methods, 3 metrics             |
| 5      | `fig5_energy_distribution.pdf`     | Violin plot: energy spread per episode          |
| 6      | `fig6_episode_fusion_dynamics.pdf` | α and alignment over one episode               |

Regenerate after a full training run:

```bash
python visualize.py              # all 6 figures
python visualize.py --figure 3  # single figure
python visualize.py --show      # display interactively
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{auv_rl_apf_2026,
  title     = {Adaptive RL-APF Framework for Energy-Efficient AUV Path Planning},
  author    = {<Author(s)>},
  booktitle = {Proceedings of the Underwater Robotics Conference},
  year      = {2026},
}
```

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

Built with [Gymnasium](https://gymnasium.farama.org), [Stable-Baselines3](https://stable-baselines3.readthedocs.io), and [PyTorch](https://pytorch.org).
