# Final Dataset Release Report (ZH)

统计时间：2026-04-13  
最终发布对象：`canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar_semantic_ir_embedding_dedup_decontam.jsonl`

## 1) 介绍

本报告对应当前可训练的最终 release 数据集，目标是：

- 保留多源 GPU kernel 任务覆盖（CUDA + Triton）
- 避免与评测集（尤其 `source_split=eval`）发生泄漏
- 在结构、语义和文本层面完成多阶段去重

最终数据条数：`21227`

最终主文件：

- `data/processed/canonical_jsonl/canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar_semantic_ir_embedding_dedup_decontam.jsonl`

---

## 2) 清洗与去重流水线（阶段数量）

| 阶段 | 文件 | 条数 | 相邻阶段移除 |
|---|---|---:|---:|
| pre | `canonical_all_plus_xpuoj.jsonl` | 29018 | - |
| AST+exact 后 | `canonical_all_plus_xpuoj_ast_near_dedup_exact.jsonl` | 25088 | 3930 |
| KernelBench 跨源筛查后 | `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl` | 25064 | 24 |
| semantic IR 去重后 | `..._semantic_ir_dedup.jsonl` | 21956 | 3108 |
| embedding 去重后 | `..._semantic_ir_embedding_dedup.jsonl` | 21227 | 729 |
| eval 泄漏清洗后（最终） | `..._decontam.jsonl` | 21227 | 0 |

总体变化：`29018 -> 21227`，净移除 `7791`（26.85%）。

---

## 3) 分布

### 3.1 按 Source

| source | 条数 | 占比 |
|---|---:|---:|
| kernelbook | 9092 | 42.83% |
| drkernel | 6677 | 31.46% |
| cuda_agent_ops_6k | 4350 | 20.49% |
| computeeval | 544 | 2.56% |
| kernelbench | 244 | 1.15% |
| tritonbench | 178 | 0.84% |
| cudabench | 97 | 0.46% |
| xpuoj | 23 | 0.11% |
| multikernelbench | 22 | 0.10% |

### 3.2 按目标后端

| target_backend | 条数 | 占比 |
|---|---:|---:|
| triton | 15960 | 75.19% |
| cuda | 5267 | 24.81% |

### 3.3 按任务族

| task_family | 条数 | 占比 |
|---|---:|---:|
| ref_code_to_kernel | 9358 | 44.09% |
| trajectory_opt | 6677 | 31.46% |
| rl_query_pool | 4350 | 20.49% |
| text_to_kernel | 842 | 3.97% |

### 3.4 按数据角色（split）

| source_split | 条数 | 占比 |
|---|---:|---:|
| train | 20394 | 96.08% |
| eval | 833 | 3.92% |

---

## 4) 组成

### 4.1 参考载荷组成（是否有代码）

| 组成类型 | 条数 | 占比 |
|---|---:|---:|
| code_only（仅 reference_code） | 20385 | 96.03% |
| code+text（reference_code + reference_text） | 842 | 3.97% |
| text_only | 0 | 0.00% |
| empty_ref | 0 | 0.00% |

说明：

- 最终 release 中 `reference_code` 是全量非空（100%）。
- 因此不存在“只有文字、没有代码参考”的样本。
- `reference_modality` 字段在部分源仍标注 `text`，但该类样本在本数据集中也已持有 `reference_code`。

### 4.2 子数据集主要组成（任务形态）

| source | 主要组成 |
|---|---|
| kernelbook | PyTorch->Triton 配对翻译题，覆盖基础算子与模块实现，`task_family=ref_code_to_kernel`。 |
| drkernel | 多轮 Triton 优化轨迹，核心是同一任务的迭代优化过程，`task_family=trajectory_opt`。 |
| cuda_agent_ops_6k | 组合算子 query pool（偏 RL 训练），`task_family=rl_query_pool`。 |
| kernelbench | 评测向 reference 题，含 `difficulty1~4`，保留为 eval。 |
| computeeval / cudabench / tritonbench / xpuoj | 文本指令驱动的 kernel 实现任务，统一归入 `text_to_kernel`，并已保证有 `reference_code`。 |
| multikernelbench | 小规模 reference 题集合（attention/broadcast/index 等分类）。 |

---

## 5) 这次额外做了什么清洗（你要求的第 5 项）

本轮新增“train 侧对 held-out eval 侧的泄漏清洗”：

- 脚本：`scripts/decontam_eval_ir_embedding.py`
- 报告：`data/metadata/stats/eval_contamination_ir_embedding_report.json`
- 输入：`..._semantic_ir_embedding_dedup.jsonl`（21227）
- held-out：`source_split == eval`（833）
- 候选：`source_split != eval`（20394）
- 清洗链路：
  1. `exact payload hash`（`reference_code + reference_text`）
  2. `semantic_ir_signature` 命中
  3. `embedding cosine`（TF-IDF word+char）

结果：

- `removed_as_contaminated = 0`
- 分阶段移除：`exact=0`, `semantic_ir=0`, `embedding=0`
- 近邻比较对数：`550`

---

## 6) 发布与追溯

发布分片（Git 仓库 `kernelcuterl`）：

- `dataset_shards/release_20260412/`
- 含 `manifest.txt`、`*.sha256`、`canonical_20260412_decontam.tar.zst.part-*`

主要可追溯日志：

- `semantic_ir_dedup_log.md`
- `semantic_embedding_dedup_log.md`
- `semantic_task_dedup_log.md`

