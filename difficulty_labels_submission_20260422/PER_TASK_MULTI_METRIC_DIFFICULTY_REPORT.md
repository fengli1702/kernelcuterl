
## 指标与方法

- 1 `ops_metric`: 值越大越难
- 2 `ast_depth_metric`: `ast_forward_graph_depth`，值越大越难
- 3 `output_len_metric`: `output_tokens`，值越大越难
- 4 `correctness_level`: 一次运行口径（correct=True→L1；compiled=True且correct=False→L3；compiled=False→L4）
- 5 `speedup_level`: `speedup`，值越小越难

## 权重

- `ops`: 0.2500
- `ast_depth`: 0.2500
- `output_len`: 0.1500
- `correctness`: 0.2000
- `speedup`: 0.1500


- ops q25/q50/q75: (2.0, 4.0, 8.0)
- ast_depth q25/q50/q75: (2.0, 3.0, 6.0)
- output_len q25/q50/q75: (1598.0, 3541.0, 8556.0)
- speedup q25/q50/q75: (1.1789, 1.7551, 2.8102)
- final weighted score q25/q50/q75: (1.4500000000000002, 2.05, 2.75)

## 各指标分级分布

| 指标 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| ops_level | 5604 (36.00%) | 3523 (22.63%) | 2784 (17.89%) | 3654 (23.48%) |
| ast_depth_level | 5981 (38.43%) | 2527 (16.24%) | 3381 (21.72%) | 3676 (23.62%) |
| output_len_level | 3892 (25.00%) | 3891 (25.00%) | 3891 (25.00%) | 3891 (25.00%) |
| correctness_level | 12931 (83.08%) | 0 (0.00%) | 2396 (15.39%) | 238 (1.53%) |
| speedup_level | 3890 (24.99%) | 3892 (25.00%) | 3890 (24.99%) | 3893 (25.01%) |

## 加权汇总后最终分级

| 最终等级 | 数量 | 占比 |
|---|---:|---:|
| level1_easy | 4024 | 25.85% |
| level2_medium | 3840 | 24.67% |
| level3_hard | 3853 | 24.75% |
| level4_expert | 3848 | 24.72% |

