# RL-MOWOA 核心代码（MATLAB）

这是强化学习增强的多目标鲸鱼优化算法（RL-MOWOA）的核心代码。为了简洁开源，已移除对比算法、可视化工具和历史输出数据。

## 运行环境
- MATLAB R2021a 或更新版本
- 无需第三方工具箱

## 快速开始
- 将此文件夹设为 MATLAB 工作目录
- 运行入口脚本：
  - 在命令窗口输入 `Main_RL_MOWOA`
- 默认行为：
  - 运行 RSM 问题 30 次独立试验
  - 将每次的 Pareto 前沿保存为 CSV 文件到 results/RSM/pareto_run_XX.csv
- 修改问题规模或运行参数：
  - 编辑 `Main_RL_MOWOA.m` 中的 `run_rsm_bt1_save_pareto_runs()` 函数
  - 可调参数：`D, M, LB, UB, SearchAgents_no, Max_iteration, num_runs, Max_evals`

## 文件说明
- `Main_RL_MOWOA.m`：入口脚本，批量运行并导出 CSV
- `RL_MOWOA.m`：RL 增强的 MOWOA 核心优化器
- `QLearningAgent.m`：离散化多参数动作空间的 ε-greedy Q-learning
- `RL_Utils.m`：状态构建、指标计算和奖励聚合
- `initialize_variables.m`, `non_domination_sort_mod.m`, `replace_chromosome.m`, `replace_chromosome_uniform.m`：初始化、排序和选择
- `evaluate_objective.m`, `bound_with_step.m`：RSM 目标函数和固定步长边界处理
- `hv2d.m`, `hv2d_norm.m`, `hv_contrib_2d.m`, `metrics_utils.m`：HV/IGD/Spread 指标

## 主要特点
- 通过 RL 混合策略+参数控制
- 动作向量 `[SF, b, p, mutation_rate]`，ε-greedy 选择
- 进度调整的 `p_eff` 在包围和螺旋更新间切换

## 配置说明
- 每次运行的种子：`rng(33 + run - 1)`
- 终止条件：`Max_evals`（评估次数）
- RL 参数：`learning_rate=0.12`, `discount_factor=0.95`, `epsilon=0.35→0.05`, `epsilon_decay=0.9975`
- 动作空间：`SF=[1.15,1.25,1.35,1.45]`, `b=[1.2,1.4,1.6,1.8]`, `p=[0.60,0.65,0.70,0.75]`, `mutation=[0.08,0.10,0.12,0.14]`

## 输出
- 每次运行的 CSV：`results/RSM/pareto_run_XX.csv`
- 每个文件包含最终种群中第一 Pareto 前沿的目标值

## 复现建议
- 保持种子固定以进行公平对比
- 调整 `LB/UB` 和 `Max_evals` 控制搜索范围和预算
- 根据需要调整 `SearchAgents_no` 和 `num_runs`

## 算法流程
- 初始化种群（分层抽样），按非支配和拥挤度排序
- 每次迭代：
  - 构建 4 维状态 `[收敛性, 多样性, 进度, 质量]`，通过 ε-greedy Q-learning 选择动作
  - Pareto 感知的领导者采样
  - 包围或螺旋更新（由进度调整的 `p_eff` 选择），然后固定步长边界
  - 自适应多项式变异和存档更新
  - 计算指标和奖励；更新 Q 表
  - 结构化均匀化和精英注入；记录收敛曲线
- 评估预算耗尽时停止；提取第一前沿并保存

## 引用
如果在学术工作中使用此代码，请引用相应论文。
