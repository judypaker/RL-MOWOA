# RL-MOWOA Core (MATLAB)

This is the core code for Reinforcement-Learning enhanced Multi-Objective Whale Optimization Algorithm (RL-MOWOA). Baseline algorithms, visualization tools, and historical outputs have been removed for clean open-sourcing.

## Requirements
- MATLAB R2021a or newer
- No third-party toolboxes required

## Quick Start
- Set this folder as the MATLAB working directory
- Run the entry script:
  - `Main_RL_MOWOA` (in Command Window)
- Default behavior:
  - Runs the RSM problem for 30 independent trials
  - Saves each trial's Pareto front to CSV at results/RSM/pareto_run_XX.csv
- To change problem size or runtime:
  - Edit function `run_rsm_bt1_save_pareto_runs()` in `Main_RL_MOWOA.m`
  - Parameters to adjust: `D, M, LB, UB, SearchAgents_no, Max_iteration, num_runs, Max_evals`

## What's Included
- `Main_RL_MOWOA.m`: Entry script with batch run and CSV export
- `RL_MOWOA.m`: Core optimizer integrating RL with MOWOA
- `QLearningAgent.m`: ε-greedy Q-learning over discretized multi-parameter action space
- `RL_Utils.m`: State construction, metrics, and reward aggregation
- `initialize_variables.m`, `non_domination_sort_mod.m`, `replace_chromosome.m`, `replace_chromosome_uniform.m`: Initialization, ranking, and selection
- `evaluate_objective.m`, `bound_with_step.m`: RSM objectives and fixed-step boundary handling
- `hv2d.m`, `hv2d_norm.m`, `hv_contrib_2d.m`, `metrics_utils.m`: HV/IGD/Spread metrics

## Key Features
- Mixed policy+parameter control via RL
- Action vector `[SF, b, p, mutation_rate]` with ε-greedy selection
- Progress-adjusted `p_eff` switches between encircling and spiral updates

## Configuration
- Seeds per run: `rng(33 + run - 1)`
- Termination by evaluations: `Max_evals`
- RL settings: `learning_rate=0.12`, `discount_factor=0.95`, `epsilon=0.35→0.05` with `epsilon_decay=0.9975`
- Action space: `SF=[1.15,1.25,1.35,1.45]`, `b=[1.2,1.4,1.6,1.8]`, `p=[0.60,0.65,0.70,0.75]`, `mutation=[0.08,0.10,0.12,0.14]`

## Outputs
- CSV per run: `results/RSM/pareto_run_XX.csv`
- Each file contains objective values of the first Pareto front from the final population

## Reproducibility Tips
- Keep seeds fixed for fair comparison across runs
- Tune `LB/UB` and `Max_evals` to control search range and budget
- Adjust `SearchAgents_no` and `num_runs` based on desired statistical confidence

## Algorithm Outline
- Initialize population with stratified sampling, rank by non-domination and crowding
- Per iteration:
  - Build 4-dim state `[convergence, diversity, progress, quality]` and select action by ε-greedy Q-learning
  - Pareto-aware leader sampling for direction
  - Encircling or spiral update (chosen by progress-adjusted `p_eff`), then fixed-step bounding
  - Adaptive polynomial mutation and archive update
  - Compute metrics and reward; update Q-table
  - Structured uniformization and elite injection; record convergence curve
- Stop when evaluation budget is exhausted; extract first front and save

## Citation
If you use this code in academic work, please cite the corresponding paper.
