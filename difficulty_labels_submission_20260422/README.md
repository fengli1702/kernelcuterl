# Difficulty Labels (Ops / AST / Output / Correctness / Speedup)

本目录用于提交题目难度标签结果，核心目标是给每条题目提供：
- 单指标难度标签（ops、ast depth、output length、correctness、speedup）
- 加权汇总后的最终难度标签（4档）

## 1. 文件说明

- `per_task_multi_metric_difficulty.csv`
  - 逐题难度标签主表（`15565` 条）
  - 包含每个指标的原始值、每个指标的等级、最终加权等级
- `per_task_multi_metric_difficulty_summary.json`
  - 阈值、权重、分布统计（可用于审计/复现）
- `PER_TASK_MULTI_METRIC_DIFFICULTY_REPORT.md`
  - 人类可读摘要报告
- `build_multi_metric_difficulty.py`
  - 标签生成脚本

## 2. 指标定义

### 2.1 ops 指标
- 字段：`ops_metric`
- 来源：
  - 优先 `model_ops_total_calls_v3`（图追踪得到）
  - 缺失时回退 `ops_count`
- 含义：模型内部算子调用总量（不含输入构造等测试代码）
- 方向：值越大，难度越高

### 2.2 ast/图深度指标
- 字段：`ast_depth_metric`
- 来源优先级：
  1. `forward_graph_depth_v3`
  2. `ast_forward_graph_depth`
  3. `model_ast_depth`
- 含义：`forward` 计算图（或回退AST）的链路深度
- 方向：值越大，难度越高

说明：该字段名叫 `ast_depth_metric`，但实际优先是图深度（`forward_graph_depth_v3`）。

### 2.3 输出长度指标
- 字段：`output_len_metric`
- 来源：`output_tokens`
- 方向：值越大，难度越高

### 2.4 正确性指标
- 字段：`correctness_level`
- 来源：`compiled` + `correct`
- 映射规则：
  - `compiled=False` -> `level4_expert`
  - `compiled=True and correct=False` -> `level3_hard`
  - `correct=True` -> `level1_easy`

### 2.5 加速比指标
- 字段：`speedup_metric`
- 来源：`speedup`
- 方向：值越小，难度越高（反向分级）

## 3. 单指标分级规则（4档）

连续指标统一按全量分位点 `Q25/Q50/Q75` 划分 4 档。

- 对于 `ops/ast_depth/output_len`（正向指标）：
  - `<=Q25 -> L1`
  - `(Q25,Q50] -> L2`
  - `(Q50,Q75] -> L3`
  - `>Q75 -> L4`

- 对于 `speedup`（反向指标）：
  - `<=Q25 -> L4`
  - `(Q25,Q50] -> L3`
  - `(Q50,Q75] -> L2`
  - `>Q75 -> L1`

本次提交实际阈值（来自 `summary.json`）：
- `ops_q25/q50/q75 = (2.0, 4.0, 8.0)`
- `ast_depth_q25/q50/q75 = (2.0, 3.0, 6.0)`
- `output_len_q25/q50/q75 = (1598.0, 3541.0, 8556.0)`
- `speedup_q25/q50/q75 = (1.1789, 1.7551, 2.8102)`

## 4. 最终加权难度

### 4.1 权重
- `ops`: `0.25`
- `ast_depth`: `0.25`
- `output_len`: `0.15`
- `correctness`: `0.20`
- `speedup`: `0.15`

### 4.2 加权分数

`weighted_difficulty_score =`
- `0.25 * ops_level`
- `+ 0.25 * ast_depth_level`
- `+ 0.15 * output_len_level`
- `+ 0.20 * correctness_level`
- `+ 0.15 * speedup_level`

### 4.3 最终等级

对 `weighted_difficulty_score` 再按全量 `Q25/Q50/Q75` 划 4 档：
- `final_q25/q50/q75 = (1.45, 2.05, 2.75)`
- `<=1.45 -> level1_easy`
- `(1.45,2.05] -> level2_medium`
- `(2.05,2.75] -> level3_hard`
- `>2.75 -> level4_expert`

## 5. 主表关键字段

- 基础评测：`task_id, compiled, correct, speedup, output_tokens`
- 原始指标：`ops_metric, ast_depth_metric, output_len_metric, speedup_metric`
- 单指标标签：
  - `ops_level(_label)`
  - `ast_depth_level(_label)`
  - `output_len_level(_label)`
  - `correctness_level(_label)`
  - `speedup_level(_label)`
- 汇总标签：`weighted_difficulty_score, final_level, final_label`

## 6. 复现命令

在同环境下可直接重跑：

```bash
/home/i_lidaifeng/venvs/fmoe-h100/bin/python /home/i_lidaifeng/kernelcuterl/difficulty_labels_submission_20260422/build_multi_metric_difficulty.py
```

