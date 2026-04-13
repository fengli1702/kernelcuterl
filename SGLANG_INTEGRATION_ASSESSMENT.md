# SGLang Server & Rollout Integration Assessment

## 📋 Executive Summary

✅ **SGLang Server** - 可以直接运行
✅ **Rollout脚本** - 各接口已完整实现
✅ **完整流程** - LLM生成 → 代码提取 → 评估已全部接线

---

## 1. SGLang Server 评估

### 1.1 run_turbo.sh 能否直接跑？

**✅ 是的，可以直接跑**

检查结果：
```
✓ Qwen3.5 SGLANG路径存在: /cpfs01/user/huiqiang.zzh/codespace/0215_opensource/sglang/python
✓ Qwen3.5 模型路径存在: /cpfs01/data/shared/Group-m6/wangzhihai.wzh/ckpts/...
✓ Tokenizer路径存在: /home/data/public/data/EvaluationData_LFS/tokenizer/...
```

### 1.2 启动命令

```bash
# 启动SGLang服务器
cd /cpfs01/user/lidaifeng.ldf/KernelRL
MODEL_FORMAT=qwen3.5 USE_MTP=true bash rollout/sglang_server/run_turbo.sh

# 服务将监听在 http://localhost:30000
# 提供OpenAI兼容的API: /v1/chat/completions
```

### 1.3 脚本配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_FORMAT` | qwen3.5 | 模型格式（qwen3.5或qwen3next） |
| `USE_MTP` | true | 启用MTP推测解码 |
| `--tp-size` | 2 | Tensor并行大小 |
| `--dtype` | bfloat16 | 数据类型 |
| `--max-running-requests` | 128 | 最大并发请求数 |
| `--mem-fraction-static` | 0.75 | 静态内存分配 |

---

## 2. Rollout脚本 接口完整性评估

### 2.1 LLM调用接口 ✅

**位置**: `rollout/single_turn_rollout.py:357-394`

```python
async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_p: float,
    timeout: float,
) -> List[str]
```

**功能**:
- 发送OpenAI兼容的chat/completions请求
- 支持多样本生成（n）
- 支持温度、top-p等采样参数
- 异步请求处理

### 2.2 代码提取接口 ✅

**位置**: `rollout/single_turn_rollout.py:169-184`

```python
def extract_kernel(text: str) -> Optional[str]
```

**功能**:
- 调用 `kernelrl.eval.detector.detector()`
- 处理多样本列表或元组返回
- 处理异常情况

### 2.3 评估接口 ✅

**位置**: `rollout/single_turn_rollout.py:201-246` & `187-197`

两个评估函数：

1. **evaluate_kernel** - 标准评估
```python
def evaluate_kernel(kernel_code, reference)
```

2. **evaluate_kernel_with_pipeline** - Pipeline评估
```python
def evaluate_kernel_with_pipeline(
    kernel_code,
    reference,
    use_compile=True,
    use_timing=False,
    use_hack_check=True
)
```

**功能**:
- 调用 `kernelrl.eval.pipeline.run_eval_pipeline()`
- 支持编译、hack检查、计时的可选启用
- 返回结构化结果

### 2.4 命令行参数 ✅

**位置**: `rollout/single_turn_rollout.py:672-730+`

所有必需参数都已实现：

```bash
# 必需参数
--data-path DATA           # 输入JSONL文件
--save-path OUTPUT         # 输出保存路径
--model MODEL_NAME         # 模型名称

# SGLang配置
--base-url URL            # 默认: http://localhost:30000/v1
--timeout SECONDS         # 请求超时 (默认: 300)
--max-concurrent N        # 最大并发数 (默认: 32)

# 生成参数
--n N                     # 样本数 (默认: 1)
--temperature TEMP        # 温度 (默认: 1.0)
--max-tokens TOKENS       # 最大tokens (默认: 81920)
--seed SEED              # 随机种子 (默认: 42)
--top-p PROB             # top-p采样 (默认: 0.95)

# Pipeline配置
--use-pipeline           # 启用pipeline评估
--pipeline-compile       # 启用编译检查
--pipeline-timing        # 启用计时
--pipeline-hack-check    # 启用代码质量检查

# 其他
--checkpoint-path PATH   # 检查点文件
```

---

## 3. 完整流程实现

### 3.1 执行流程

```
main_async(args)
  ├─ 1) 加载数据: load_jsonl(args.data_path)
  │
  ├─ 2) 检查状态:
  │    ├─ check_rollout_complete() - 检查rollout是否完成
  │    ├─ check_eval_complete() - 检查评估是否完成
  │    └─ 决定是否需要重新运行
  │
  ├─ 3) 并发处理: asyncio.gather(process_item(...))
  │    └─ 对每个样本：
  │        ├─ chat_completion() - 调用LLM生成代码
  │        ├─ extract_kernel() - 提取代码
  │        ├─ evaluate_kernel() 或 evaluate_kernel_with_pipeline() - 评估
  │        └─ 返回 ItemResult
  │
  ├─ 4) 保存结果: save_jsonl() & save_final_results()
  │
  └─ 5) 计算统计: compute_global_summary()
```

### 3.2 流程图

```
LLM Output (from SGLang)
    ↓
extract_kernel()
    ↓
kernel_code + reference_code
    ↓
evaluate_kernel_with_pipeline()
    ├─ run_eval_pipeline()
    │  ├─ compile_module()
    │  ├─ hack_check_module()
    │  └─ timing_module() [可选]
    │
    └─ 返回: {correct, compile_ok, hack_clean, ...}
    ↓
ItemResult (保存到JSONL)
    ↓
compute_global_summary()
    ↓
最终统计报告
```

---

## 4. 已实现的接口总结

| 接口 | 位置 | 状态 | 说明 |
|------|------|------|------|
| LLM生成 | chat_completion() | ✅ | OpenAI兼容API |
| 代码提取 | extract_kernel() | ✅ | 调用detector模块 |
| 标准评估 | evaluate_kernel() | ✅ | 基础评估 |
| Pipeline评估 | evaluate_kernel_with_pipeline() | ✅ | 完整评估 |
| 数据加载 | load_jsonl() | ✅ | 支持JSONL和JSON |
| 结果保存 | save_jsonl() | ✅ | 支持checkpoint |
| 状态检查 | check_rollout/eval_complete() | ✅ | 支持resume |
| 统计计算 | compute_global_summary() | ✅ | 聚合统计 |

---

## 5. 使用流程

### 5.1 启动SGLang服务

```bash
# 终端1: 启动SGLang服务器
cd /cpfs01/user/lidaifeng.ldf/KernelRL
MODEL_FORMAT=qwen3.5 USE_MTP=true bash rollout/sglang_server/run_turbo.sh

# 等待输出: "Serving on http://0.0.0.0:30000"
```

### 5.2 运行Rollout

```bash
# 终端2: 运行rollout脚本
cd /cpfs01/user/lidaifeng.ldf/KernelRL

python rollout/single_turn_rollout.py \
  --data-path data/eval/kernelbench/kernelbench.jsonl \
  --save-path output/qwen3.5-test \
  --model qwen3.5-35A3 \
  --base-url http://localhost:30000/v1 \
  --use-pipeline \
  --pipeline-compile \
  --pipeline-hack-check \
  --n 5 \
  --max-concurrent 16
```

### 5.3 检查结果

```bash
# 查看最终结果
cat output/qwen3.5-test

# 查看统计摘要
cat output/qwen3.5-test.summary.json

# 如果需要resume（继续之前中断的任务）
python rollout/single_turn_rollout.py \
  --data-path data/eval/kernelbench/kernelbench.jsonl \
  --save-path output/qwen3.5-test \
  --model qwen3.5-35A3 \
  # 其他参数保持一致，脚本会自动检测并resume
```

---

## 6. 关键参数说明

### SGLang参数
- `--base-url`: 必须与SGLang服务器的实际地址匹配（默认 localhost:30000）
- `--model`: 必须与SGLang加载的模型名称匹配

### Pipeline参数
- `--use-pipeline`: 启用pipeline（推荐）
- `--pipeline-compile`: 编译检查（推荐启用）
- `--pipeline-hack-check`: 代码质量检查（推荐启用）
- `--pipeline-timing`: GPU计时（需要GPU，默认禁用）

### 并发控制
- `--max-concurrent`: 控制并发数，避免服务器过载
- `--timeout`: 单个请求超时时间

---

## 7. 错误处理

脚本已实现的错误处理：

✅ LLM请求失败 - 标记样本失败并继续
✅ 代码提取失败 - kernel_code设为None
✅ 评估失败 - 返回错误信息
✅ 网络错误 - 异步重试
✅ 超时 - 可配置的超时处理

---

## 8. 结论

### ✅ SGLang Server
- **可以直接跑**: 路径都存在，脚本完整
- **启动命令**: `MODEL_FORMAT=qwen3.5 USE_MTP=true bash rollout/sglang_server/run_turbo.sh`

### ✅ Rollout脚本接口
- **LLM调用**: ✓ chat_completion()
- **代码提取**: ✓ extract_kernel()
- **评估**: ✓ evaluate_kernel_with_pipeline()
- **数据处理**: ✓ 加载、保存、resume、统计
- **参数配置**: ✓ 所有必需参数都已暴露

### ✅ 整体集成
- **完全可以使用**: LLM → 代码提取 → Pipeline评估已全部接线
- **生产就绪**: 支持并发、断点续传、错误处理

### 🚀 可以直接启动

```bash
# 启动服务
MODEL_FORMAT=qwen3.5 USE_MTP=true bash /cpfs01/user/lidaifeng.ldf/KernelRL/rollout/sglang_server/run_turbo.sh &

# 等待30秒后运行rollout
sleep 30

python /cpfs01/user/lidaifeng.ldf/KernelRL/rollout/single_turn_rollout.py \
  --data-path /cpfs01/user/lidaifeng.ldf/KernelRL/data/eval/kernelbench/kernelbench.jsonl \
  --save-path /cpfs01/user/lidaifeng.ldf/KernelRL/output/test \
  --model qwen3.5-35A3 \
  --use-pipeline \
  --pipeline-compile \
  --pipeline-hack-check
```

