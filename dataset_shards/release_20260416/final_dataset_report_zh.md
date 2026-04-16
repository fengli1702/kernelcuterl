
最终数据条数：`21227`
最终发布文件：`data/processed/canonical_jsonl/canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar_semantic_ir_embedding_dedup_decontam_queryfull_en_v2.jsonl`

`query` 完整性修复（2026-04-14）：

- 补齐条数：`819`（`computeeval=544`, `tritonbench=178`, `cudabench=97`）
- 修复规则：将 `reference_artifacts.reference_code` 追加到 `query`（`Reference implementation` 代码块）
- 修复后校验：`21227/21227` 条样本满足“`query` 含完整参考代码”

`query` 英文化清洗（2026-04-15）：

- 清洗输入：`..._decontam_queryfull.jsonl`
- 清洗输出（最终）：`..._decontam_queryfull_en_v2.jsonl`
- 清洗策略：优先翻译 `query` 非代码段；代码段中仅翻译注释/文档字符串；对极少数残留样本做非 ASCII 标识符与中文错误信息替换
- 清洗结果：`query` 含中文从 `122` 条降到 `0` 条，同时保持 `reference_code` 全量非空且 `ref in query = 21227/21227`

## 2) 清洗与去重


### 2.1 去重

| Point | 去重手段 | 技术手段 | 主要去掉的重复 | 真实样例（removed -> keeper） | 示例说明 |
|---|---|---|---|---|---|
| 1 | 结构一致去重（AST-near） | 代码结构 token（Python AST / C-like）+ MinHash/LSH + Jaccard 聚类 | 代码骨架一致、变量名/局部常量改动、模板型重复 | `kernelbook__92ee5e3713c90d98 -> kernelbook__123ba292aa23f1de` | 同类 matmul/GEMM 题，结构近似，Point1 直接合并。 |
| 2 | IR 去重（Semantic IR） | 自定义语义 IR（归一化 AST），忽略 `get_inputs/get_init_inputs/main` workload 脚手架，按 `semantic_ir_signature` 聚类 | 核心算子图一致但 workload 配置不同；包装层改写但语义不变 | `kernelbook__adc3f5e0122f1429 -> kernelbook__1fe051a3c54799bb` | 同类 matmul 核心语义一致，即使 workload/输入构造不同，也在 Point2 被去掉。 |
| 3 | 语义相近去重（Embedding） | `query+reference_text+code_calls+backend/task` 生成语义文本，TF-IDF(word+char) 向量，cosine 相似度聚类 | 文本表达不同但任务意图相同的近重复 | `kernelbook__7fd92cede5791c60 -> kernelbook__532b8ea7d30fcbf2` | 代码未必同构，但任务描述高度同义，Point3 去掉语义近重复。 |

补充结论（关于你问的 workload）：

- “同是矩阵乘法、只是 workload 不同”主要在 **Point2（Semantic IR）** 被去重。
- 如果结构也很接近，Point1 可能更早命中。
- 如果结构差异大但题意接近，则 Point3 命中。

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


### 3.3 数据集组成

| source | 主要组成 |
|---|---|
| kernelbook | PyTorch->Triton 配对翻译题，覆盖基础算子与模块实现，`task_family=ref_code_to_kernel`。 |
| drkernel | 多轮 Triton 优化轨迹，核心是同一任务的迭代优化过程，`task_family=trajectory_opt`。 |
| cuda_agent_ops_6k | 组合算子 query pool（偏 RL 训练），`task_family=rl_query_pool`。 |
| kernelbench | 评测向 reference 题，含 `difficulty1~4`，保留为 eval。 |
| computeeval / cudabench / tritonbench / xpuoj | 文本指令驱动的 kernel 实现任务，统一归入 `text_to_kernel`，并已保证有 `reference_code`。 |
| multikernelbench | 小规模 reference 题集合（attention/broadcast/index 等分类）。 |

## 4) 各 Source case

说明：以下每个 source 抽取最终 release 中 1 条真实样本；展示该样本的 `query` 与 `reference_code` 全量内容。


### 4.1 kernelbook

- `sample_id`: `kernelbook__976bb64b0712842c`
- `source_split`: `train`
- `task_family`: `ref_code_to_kernel`
- `target_backend`: `triton`
- `upstream_id`: `1151`
- `file_path`: ``
- `reference_code_length`: `1778`

#### Query（完整）

````text
You are an expert GPU kernel engineer.

Task family: ref_code_to_kernel
Target backend: triton
Correctness requirement: preserve semantics of the reference implementation.
Performance goal: optimize runtime without changing outputs.

Reference implementation:
```
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):

    def __init__(self, output_dim: 'int', eps: 'float'=0.0):
        """Layer that computes hard whitening for W-MSE using the Cholesky decomposition.

        Args:
            output_dim (int): number of dimension of projected features.
            eps (float, optional): eps for numerical stability in Cholesky decomposition. Defaults
                to 0.0.
        """
        super(Whitening2d, self).__init__()
        self.output_dim = output_dim
        self.eps = eps

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Performs whitening using the Cholesky decomposition.

        Args:
            x (torch.Tensor): a batch or slice of projected features.

        Returns:
            torch.Tensor: a batch or slice of whitened features.
        """
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.output_dim, -1).mean(-1).view(1, -1, 1, 1)
        xn = x - m
        T = xn.permute(1, 0, 2, 3).contiguous().view(self.output_dim, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)
        eye = torch.eye(self.output_dim).type(f_cov.type())
        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.cholesky(
            f_cov_shrinked), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.
            output_dim, 1, 1)
        decorrelated = conv2d(xn, inv_sqrt)
        return decorrelated.squeeze(2).squeeze(2)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4}]
```

Input / initialization contract:
Not provided.

Requirements:
1. Return only executable code.
2. The main optimized entry point must be `ModelNew`.
````

#### Reference Code（完整）

````text
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):

    def __init__(self, output_dim: 'int', eps: 'float'=0.0):
        """Layer that computes hard whitening for W-MSE using the Cholesky decomposition.

        Args:
            output_dim (int): number of dimension of projected features.
            eps (float, optional): eps for numerical stability in Cholesky decomposition. Defaults
                to 0.0.
        """
        super(Whitening2d, self).__init__()
        self.output_dim = output_dim
        self.eps = eps

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Performs whitening using the Cholesky decomposition.

        Args:
            x (torch.Tensor): a batch or slice of projected features.

        Returns:
            torch.Tensor: a batch or slice of whitened features.
        """
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.output_dim, -1).mean(-1).view(1, -1, 1, 1)
        xn = x - m
        T = xn.permute(1, 0, 2, 3).contiguous().view(self.output_dim, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)
        eye = torch.eye(self.output_dim).type(f_cov.type())
        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.cholesky(
            f_cov_shrinked), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.
            output_dim, 1, 1)
        decorrelated = conv2d(xn, inv_sqrt)
        return decorrelated.squeeze(2).squeeze(2)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4}]
````


### 4.2 drkernel

- `sample_id`: `drkernel__c01e52ddb7875b4b`
- `source_split`: `train`
- `task_family`: `trajectory_opt`
- `target_backend`: `triton`
- `upstream_id`: `1`
- `file_path`: ``
- `reference_code_length`: `277`

#### Query（完整）

````text
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

        ```

        import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


        ```

        The example new arch with custom Triton kernels looks like this:

        ```
        import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
        ```

    You are given the following architecture:
    ```
    import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = x - 1.0
        return x


def get_inputs():
    return [torch.randn(64, 128)]


def get_init_inputs():
    return []
    ```

Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
````

#### Reference Code（完整）

````text
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = x - 1.0
        return x


def get_inputs():
    return [torch.randn(64, 128)]


def get_init_inputs():
    return []
````


### 4.3 cuda_agent_ops_6k

- `sample_id`: `cuda_agent_ops_6k__6729d73a27fb57e9`
- `source_split`: `train`
- `task_family`: `rl_query_pool`
- `target_backend`: `cuda`
- `upstream_id`: `0`
- `file_path`: ``
- `reference_code_length`: `607`

#### Query（完整）

````text
You are an expert GPU kernel engineer.

Task family: rl_query_pool
Target backend: cuda
Correctness requirement: preserve semantics of the reference implementation.
Performance goal: optimize runtime without changing outputs.

Reference implementation:
```
import torch
from torch import digamma, max
from torch.nn import BatchNorm3d, Parameter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm3d(10)
        self.parameter = Parameter(torch.randn(10))

    def forward(self, x):
        x = self.batch_norm(x)
        x = digamma(x)
        x = max(x)
        x = x + self.parameter
        return x


batch_size = 512
channels = 10
depth = 15
height = 15
width = 15


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width)]


def get_init_inputs():
    return []
```

Input / initialization contract:
Not provided.

Requirements:
1. Return only executable code.
2. The main optimized entry point must be `ModelNew`.
````

#### Reference Code（完整）

````text
import torch
from torch import digamma, max
from torch.nn import BatchNorm3d, Parameter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm3d(10)
        self.parameter = Parameter(torch.randn(10))

    def forward(self, x):
        x = self.batch_norm(x)
        x = digamma(x)
        x = max(x)
        x = x + self.parameter
        return x


batch_size = 512
channels = 10
depth = 15
height = 15
width = 15


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width)]


def get_init_inputs():
    return []
````


### 4.4 computeeval

- `sample_id`: `computeeval__5a8d2d2f0fdc82cf`
- `source_split`: `eval`
- `task_family`: `text_to_kernel`
- `target_backend`: `cuda`
- `upstream_id`: `2025-1::CUDA/31`
- `file_path`: ``
- `reference_code_length`: `442`

#### Query（完整）

````text
System message:
You are a senior CUDA/C/C++ engineer. Produce complete, compilable solutions from a structured problem specification. Follow these rules:

General
- You will be given: a problem description, context files (editable), and build environment details (e.g., build command).
- Hidden tests exist but are not shown. Do not mention tests, do not write test code, and do not add I/O used only for testing.
- Use only the APIs and contracts specified in the problem and context files. Preserve all provided function signatures exactly.
- Prefer using only headers already present in the provided codebase. Avoid adding new headers unless strictly necessary and supported by the build command. Do not introduce third-party dependencies.

Context files policy
- You may modify provided context files when necessary. If you include any file in your solution output (new or modified), emit its full and final contents; your output will overwrite the provided version.
- Only emit files you add or modify. Do not output files that are unchanged, and do not include placeholder blocks saying "no changes" or similar.

Build command
- You should pay careful attention to the build command or any context files about the build process.
- The build command and/or context build files may include important hints about required files or expected project structure.  This likely includes the name of the expected solution file, important macros, standards, or linked libraries.
- Pay special attention to -I or -isystem flags -- they indicate important include paths.  Remember, if a -I or -isystem flag is present you do not need to include the relative path in your #include statements.

Output format
- Output only source files needed for the solution. No explanations or commentary.
- Each file must be in its own fenced code block, with the first line indicating its path as a comment.
  Example:
  ```
  // file: geodistance.cu
  #include "geodistance.h"
  ...
  ```

Code quality and constraints

The solution must compile cleanly with the provided build command and target architectures.
Avoid unnecessary heap allocations, environment access, and global mutable state. Keep deterministic behavior.
Honor all contracts, constants, and macros defined in provided headers.

For CUDA:
Implement kernels with correct global signatures and parameter types.
Bounds-check all memory accesses; consider grid-stride loops when appropriate for scalability.
Favor coalesced memory access and avoid undefined behavior.
Apply appropriate numerical stability practices when needed (e.g., clamp arguments before acos/asin).

Reasoning discipline

Think through edge cases and performance internally, but output only the final code files, no analysis or explanations.

User message:
Produce the complete solution as one or more source files that compile with the provided build command. Do not output anything except the code files.

Problem
Description:
Write a function `void add(int num_items, const int *in, int *out)` that adds `num_items` elements on GPU using CUB library.

The resulting sum should be stored in `out`.

Your solution should include the contract header `add.h` which declares the function signature.

The following headers are available and should not be re-included in your solution:
```cuda
#include <cub/device/device_reduce.cuh>
```

Build command:
nvcc -I include -o test.out solution.cu test/*.cu -arch=native

Context files:
--- file: include/add.h
```h
#ifndef ADD_H
#define ADD_H

void add(int num_items, const int* in, int *out);

#endif // ADD_H
```


Output requirements

Emit only the source files necessary to satisfy the problem (new or modified).
Only emit files you add or modify. Do not output files that are unchanged, and do not include placeholder blocks saying "no changes" or similar.
Do not include any test code or references to tests.
If an interface header is provided (e.g., declares functions to implement), place implementations in a corresponding .cu/.cc source file and include that header.
Begin your response with the first code block.
````

#### Reference Code（完整）

````text
// file: solution.cu
#include "add.h"
#include <cub/device/device_reduce.cuh>
#include <cstddef>

void add(int num_items, const int* in, int *out) {
  std::size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, in, out, num_items);

  void* temp_storage;
  cudaMalloc(&temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, in, out, num_items);

  cudaFree(temp_storage);
}
````


### 4.5 kernelbench

- `sample_id`: `kernelbench__68cd5b24345490fc`
- `source_split`: `eval`
- `task_family`: `ref_code_to_kernel`
- `target_backend`: `cuda`
- `upstream_id`: `KernelBench/level1/100_HingeLoss.py`
- `file_path`: `KernelBench/level1/100_HingeLoss.py`
- `reference_code_length`: `566`

#### Query（完整）

````text
You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA operators and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!

Here's an example to show you the syntax of inline embedding custom CUDA operators in PyTorch:

Example:

Input architecture:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

Optimized with CUDA operators:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []

Note: The kernels should be optimized for FP32 (32-bit floating point) precision.
````

#### Reference Code（完整）

````text
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []
````


### 4.6 tritonbench

- `sample_id`: `tritonbench__41474e4de6cf1a77`
- `source_split`: `train`
- `task_family`: `text_to_kernel`
- `target_backend`: `triton`
- `upstream_id`: `TritonBench_G_simp_alpac_v1.json::0`
- `file_path`: ``
- `reference_code_length`: `12719`

#### Query（完整）

````text
You are a expert in writing Triton operators for efficient GPU programming. Use triton language write a kernel and wrapper according following instruction.
            The Triton code implements a custom attention mechanism with forward and backward kernels for computing gradients. There are two main functions, `LightningAttention2NoDecay.forward` for the forward pass and `LightningAttention2NoDecay.backward` for the backward pass. In the forward pass, it computes the output using queries (Q), keys (K), and values (V) tensors. The backward pass computes gradients with respect to Q, K, and V.

            The `_fwd_kernel` function computes the attention output using block processing. It loads blocks of Q, K, and V, computes the QK product, and then computes the output using matrix multiplication.

            The `_bwd_intra_kernel` function calculates gradients within each block by backpropagating the error from the output to the inputs. It updates gradients of Q, K, and V.

            The `_bwd_inter_kernel` handles inter-block computations for gradients. It iterates over blocks to compute and accumulate gradients for K and V.

            The block size for forward and backward computations is 64, while the compute block size (CBLOCK) in the backward intra part is 32.
````

#### Reference Code（完整）

````text
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    ##### init diag decay(Lambda); q, k decay; kv
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    ##### get block ptr
    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    ##### init diag decay(Lambda)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]

    ##### load block
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    ##### compute
    dqk = tl.dot(do, v_trans)
    dqk = tl.where(index >= 0, dqk, 0)
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    # save
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    ##### get block ptr
    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    # mask
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)

    ##### init lambda; kv
    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    ##### compute dq inter
    for i in range(NUM_BLOCK):
        # compute in subblock
        for j in range(NUM_CBLOCK):
            if i > 0:  # if not add this, may have bug
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans)
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        # update kv in subblock
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            kv_trans_current += tl.dot(v_trans, k)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans += kv_trans_current

    ##### get block ptr
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    ##### init dkv
    dkv = tl.zeros([d, e], dtype=tl.float32)

    ##### compute dk, dv inter
    for i in range(NUM_BLOCK - 1, -1, -1):
        # compute in subblock
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:  # if not add this, may have bug
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                dk_inter_trans = tl.dot(dkv, v_trans)
                dv_inter = tl.dot(k, dkv)

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        # update dkv in subblock
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            dkv_current += tl.dot(q_trans, do)

        dkv += dkv_current


class LightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
        # parallel over channel
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))

        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            BLOCK_MODEL=BLOCK_MODEL,
        )

        ctx.save_for_backward(q, k, v)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        b, h, n, d = q.shape
        e = v.shape[-1]

        # block size
        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        # compute block size
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK

        # for intra part, compute in parallel
        grid = (b * h, NUM_BLOCK)
        _bwd_intra_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # for inter part, compute in sequencial
        grid = (b * h,)
        _bwd_inter_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        return dq, dk, dv


lightning_attn2_no_decay = LightningAttention2NoDecay.apply
````


### 4.7 cudabench

- `sample_id`: `cudabench__f279f1494ee6f798`
- `source_split`: `train`
- `task_family`: `text_to_kernel`
- `target_backend`: `cuda`
- `upstream_id`: `0::level1_prompt`
- `file_path`: ``
- `reference_code_length`: `8840`

#### Query（完整）

````text
Implement the Post_Process_GL kernel that processes an input image. The input is a 128x128 RGBA image stored as uint8 values. The kernel should apply a disc-shaped convolution filter with radius r=4 to each pixel. For pixels within the disc neighborhood, if their luminance (average of RGB channels) exceeds threshold=0.8, multiply their RGB values by highlight=2.0 before averaging. The output should be a flattened 16384-element array of uint32 values, each representing an ABGR-packed pixel (alpha=255) with processed RGB channels. Boundary handling should use clamp addressing, and shared memory should be used for efficient neighborhood access.
````

#### Reference Code（完整）

````text
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// Constants and Macros
// ----------------------------------------------------------------------------
// SMEM macro to index the shared memory array "sdata"
// tilew = blockDim.x + 2*r. However, since the shared memory is 1D array of uchar4
// and we treat it as 2D: sdata[ y * tilew + x ]
// In the kernel, 'tilew' is passed as argument.
#define SMEM(X, Y) sdata[(Y)*tilew + (X)]

// ----------------------------------------------------------------------------
// Texture Object
// ----------------------------------------------------------------------------
// We need to use a texture object for reading input as per kernel signature.
// However, since we are writing a standalone app without OpenGL interop,
// we will create a CUDA texture object from a CUDA array.

// ----------------------------------------------------------------------------
// Helper Device Functions (from typical NVIDIA samples context)
// ----------------------------------------------------------------------------
__device__ inline uchar4 getPixel(int x, int y, cudaTextureObject_t tex) {
    // tex2D<uchar4>(tex, x, y) returns the pixel
    // Since we use cudaReadModeElementType, we get uchar4 directly
    return tex2D<uchar4>(tex, (float)x, (float)y);
}

__device__ inline unsigned int rgbToInt(float r, float g, float b) {
    // Pack into ABGR or ARGB integer.
    // The kernel comment says "// ABGR".
    // Typically: (a << 24) | (b << 16) | (g << 8) | r
    // Assuming alpha is 255
    unsigned int ir = (unsigned int)min(255.0f, max(0.0f, r));
    unsigned int ig = (unsigned int)min(255.0f, max(0.0f, g));
    unsigned int ib = (unsigned int)min(255.0f, max(0.0f, b));
    unsigned int ia = 255;
    return (ia << 24) | (ib << 16) | (ig << 8) | ir;
}

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
__global__ void cudaProcess(unsigned int       *g_odata,
                            int                 imgw,
                            int                 imgh,
                            int                 tilew,
                            int                 r,
                            float               threshold,
                            float               highlight,
                            cudaTextureObject_t inTex)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x  = blockIdx.x * bw + tx;
    int y  = blockIdx.y * bh + ty;

    // Check bounds for global read/write
    // Although the tiling logic loads halo, we must be careful not to read
    // outside image bounds if texture is not set to Clamp/Border.
    // Texture hardware handles OOB if configured correctly.

    // copy tile to shared memory
    // center region
    SMEM(r + tx, r + ty) = getPixel(x, y, inTex);

    // borders
    if (threadIdx.x < r) {
        // left
        SMEM(tx, r + ty) = getPixel(x - r, y, inTex);
        // right
        SMEM(r + bw + tx, r + ty) = getPixel(x + bw, y, inTex);
    }

    if (threadIdx.y < r) {
        // top
        SMEM(r + tx, ty) = getPixel(x, y - r, inTex);
        // bottom
        SMEM(r + tx, r + bh + ty) = getPixel(x, y + bh, inTex);
    }

    // load corners
    if ((threadIdx.x < r) && (threadIdx.y < r)) {
        // tl
        SMEM(tx, ty) = getPixel(x - r, y - r, inTex);
        // bl
        SMEM(tx, r + bh + ty) = getPixel(x - r, y + bh, inTex);
        // tr
        SMEM(r + bw + tx, ty) = getPixel(x + bh, y - r, inTex);
        // br
        SMEM(r + bw + tx, r + bh + ty) = getPixel(x + bw, y + bh, inTex);
    }

    // wait for loads to complete
    cg::sync(cta);

    // perform convolution
    float rsum    = 0.0f;
    float gsum    = 0.0f;
    float bsum    = 0.0f;
    float samples = 0.0f;

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            uchar4 pixel = SMEM(r + tx + dx, r + ty + dy);

            // only sum pixels within disc-shaped kernel
            float l = (float)(dx * dx + dy * dy);

            if (l <= (float)(r * r)) {
                float fr = float(pixel.x);
                float fg = float(pixel.y);
                float fb = float(pixel.z);

                // brighten highlights
                float lum = (fr + fg + fb) / (255.0f * 3.0f);

                if (lum > threshold) {
                    fr *= highlight;
                    fg *= highlight;
                    fb *= highlight;
                }

                rsum += fr;
                gsum += fg;
                bsum += fb;
                samples += 1.0f;
            }
        }
    }

    if (samples > 0.0f) {
        rsum /= samples;
        gsum /= samples;
        bsum /= samples;
    }

    // ABGR
    if (x < imgw && y < imgh) {
        g_odata[y * imgw + x] = rgbToInt(rsum, gsum, bsum);
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
void read_binary(const std::string& filename, void* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Can not open: " << filename << std::endl; exit(1); }
    in.read(reinterpret_cast<char*>(data), size);
    in.close();
}

void write_binary(const std::string& filename, const void* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "Can not write: " << filename << std::endl; exit(1); }
    out.write(reinterpret_cast<const char*>(data), size);
    out.close();
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    const int width = 128;
    const int height = 128;
    const int radius = 4;
    const float threshold = 0.8f;
    const float highlight = 2.0f;

    size_t img_size_bytes = width * height * sizeof(uchar4);
    size_t out_size_bytes = width * height * sizeof(unsigned int);

    // Host alloc
    uchar4* h_in = new uchar4[width * height];
    unsigned int* h_out = new unsigned int[width * height];

    // Read Input
    read_binary("data/input_img.bin", h_in, img_size_bytes);

    // CUDA Array for Texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, h_in, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice);


    // Create Texture Object
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp to edge
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; // Pixel exact access
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // Use integer coords [0, width)

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Output Buffer
    unsigned int* d_out;
    cudaMalloc(&d_out, out_size_bytes);

    // Kernel Launch Config
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);

    // Shared memory size calculation:
    // Tile width includes the halo (radius) on both sides.
    // tile_w = blockDim.x + 2 * r
    // tile_h = blockDim.y + 2 * r
    int tile_w = dimBlock.x + 2 * radius;
    int tile_h = dimBlock.y + 2 * radius;
    size_t shm_size = tile_w * tile_h * sizeof(uchar4);

    cudaProcess<<<dimGrid, dimBlock, shm_size>>>(
        d_out,
        width,
        height,
        tile_w, // Passed as 'tilew' used in SMEM macro
        radius,
        threshold,
        highlight,
        texObj
    );

    cudaDeviceSynchronize();

    // Read back
    cudaMemcpy(h_out, d_out, out_size_bytes, cudaMemcpyDeviceToHost);

    // Write output
    write_binary("data/output_img.bin", h_out, out_size_bytes);

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
````


### 4.8 xpuoj

- `sample_id`: `xpuoj__04fb76810382e077`
- `source_split`: `eval`
- `task_family`: `text_to_kernel`
- `target_backend`: `triton`
- `upstream_id`: `20::triton`
- `file_path`: ``
- `reference_code_length`: `739`

#### Query（完整）

````text
You are an expert CUDA kernel engineer.

Task family: text_to_kernel
Target backend: triton

Problem specification:
Title: Packbits

[题目描述]
你需要实现一个 packbits 算子。该算子将一维布尔数组按每 8 个 bit 打包为一个 `uint8` 字节，采用**高位在前**（big-endian）的打包规则。

具体来说，对于输出位置 $i$，其对应的 8 个输入 bit 为 $b_0, b_1, \ldots, b_7$（其中 $b_j$ 表示输入布尔值），打包结果为：

$$
\text{packed}[i] = \sum_{j=0}^{7} b_j \cdot 2^{7-j}
$$

或等价地用位运算表示：

$$
\text{packed}[i] = (b_0 \ll 7) \mid (b_1 \ll 6) \mid (b_2 \ll 5) \mid (b_3 \ll 4) \mid (b_4 \ll 3) \mid (b_5 \ll 2) \mid (b_6 \ll 1) \mid b_7
$$

其中：
* 第 0 个 bit 放在结果字节的第 7 位（最高位）
* 第 1 个 bit 放在结果字节的第 6 位
* ...
* 第 7 个 bit 放在结果字节的第 0 位（最低位）

如果最后一组不足 8 个 bit，则缺失位置按 0 补齐。

评测程序会调用你实现的 `run_kernel` 函数，在函数内部你需要计算合适的 grid/block 并 launch kernel 来完成 bit packing。

如何提交代码详见[评测指南](/d/2)。

[接口约定]
你必须在提交的 Python 代码中提供 `run_kernel` 函数，函数名、参数顺序必须完全一致：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def your_kernel(...):
    ...

def run_kernel(bits, packed, num_bits):
    ...
```

### 参数说明

* `bits`：输入 tensor（bool），shape $(\text{num_bits},)$，底层按 1 字节存储
* `packed`：输出 tensor（uint8），shape $(\lceil \text{num_bits} / 8 \rceil,)$
* `num_bits`：输入 bit 数量

`run_kernel` 内部需要完成以下工作：

* 根据数据规模计算合适的 grid 和 block 配置
* launch 你实现的 triton kernel，完成 packbits

[输入格式]
本题的输入由评测程序在 GPU 上构造并传入 `run_kernel`。评测会生成以下参数并按顺序传入：

* `bits`：bool tensor，shape $(\text{num_bits},)$，连续存储，底层按 1 字节布尔值存储
* `packed`：uint8 tensor，shape $(\lceil \text{num_bits} / 8 \rceil,)$，连续存储，作为输出缓冲区
* `num_bits`：int64 标量，输入 bit 数量

[输出格式]
你需要将计算结果写入 `packed` 指向的缓冲区，得到一个 shape 为 $(\lceil \text{num_bits} / 8 \rceil,)$ 的 uint8 张量。

对每个输出位置 $i$（$0 \le i < \lceil \text{num_bits} / 8 \rceil$）：

$$
\text{packed}[i] = \sum_{j=0}^{7} \text{bit}(i \cdot 8 + j) \cdot 2^{7-j}
$$

其中 $\text{bit}(k)$ 是第 $k$ 个输入布尔值（转换为 0 或 1）。

展开写为：

$$
\text{packed}[i] = (b_0 \ll 7) \mid (b_1 \ll 6) \mid (b_2 \ll 5) \mid (b_3 \ll 4) \mid (b_4 \ll 3) \mid (b_5 \ll 2) \mid (b_6 \ll 1) \mid b_7
$$

其中 $b_j = \text{bit}(i \cdot 8 + j)$，若索引越界则 $b_j = 0$。

**注意**：高位在前（big-endian）规则，即第 0 个 bit 放在字节最高位。

[样例]
## 输入

`bits`（$\text{num_bits} = 10$）：

```
[True, False, True, True, False, False, True, False, True, True]
```

对应的二进制表示（1 表示 True，0 表示 False）：

```
[1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
```

## 输出

`packed`（shape $(\lceil 10 / 8 \rceil,) = (2,)$）：

**计算过程**：

对于 $i = 0$（第一个输出字节），对应输入 $b_0$ 到 $b_7$：`[1, 0, 1, 1, 0, 0, 1, 0]`

按高位在前打包：

$$
\text{packed}[0] = (1 \ll 7) \mid (0 \ll 6) \mid (1 \ll 5) \mid (1 \ll 4) \mid (0 \ll 3) \mid (0 \ll 2) \mid (1 \ll 1) \mid 0
$$

$$
= 128 + 0 + 32 + 16 + 0 + 0 + 2 + 0 = 178
$$

二进制：`10110010`

对于 $i = 1$（第二个输出字节），对应输入 $b_8$ 到 $b_{15}$：`[1, 1, 0, 0, 0, 0, 0, 0]`（不足 8 个补 0）

$$
\text{packed}[1] = (1 \ll 7) \mid (1 \ll 6) \mid (0 \ll 5) \mid (0 \ll 4) \mid (0 \ll 3) \mid (0 \ll 2) \mid (0 \ll 1) \mid 0
$$

$$
= 128 + 64 + 0 + 0 + 0 + 0 + 0 + 0 = 192
$$

二进制：`11000000`

**输出结果**：

```
[178, 192]
```

[数据范围与提示]
## 数据范围

* `bits` 的数据类型为 `torch.bool`，底层按 1 字节存储
* `packed` 的数据类型为 `torch.uint8`
* `bits` 的 shape 为 $(\text{num\_bits},)$
* `packed` 的 shape 为 $(\lceil \text{num\_bits} / 8 \rceil,)$
* 所有 tensor 均为连续存储

## 正确性要求

* 必须按高位在前（big-endian）的顺序打包
* 第 0 个 bit 必须放在结果字节的最高位（第 7 位）
* 必须正确处理最后不足 8 bit 的尾块（缺失位补 0）
* 必须写回 `packed`

## 提示

* 处理尾块时需要检查索引是否越界


## 测试用例尺寸

| 测试用例ID | num_bits | bits形状 | packed形状 |
|---|---|---|---|
| 1 | 4096 | (4096,) | (512,) |
| 2 | 12345 | (12345,) | (1544,) |
| 3 | 65536 | (65536,) | (8192,) |
| 4 | 1000003 | (1000003,) | (125001,) |
| 5 | 5000000 | (5000000,) | (625000,) |

[PyTorch 参考实现]
```python
def baseline(bits, packed, num_bits):
    """
    PyTorch 参考实现：Packbits

    参数说明：
    * bits: 输入张量（bool），shape (num_bits,)，底层按 1 字节布尔值存储，只读
    * packed: 输出张量（uint8），shape (ceil(num_bits / 8),)，需写入结果
    * num_bits: 输入 bit 数量（int64）
    """
    num_packed = (num_bits + 7) // 8
    bits_uint8 = bits.to(torch.uint8)
    if num_bits % 8 != 0:
        pad_size = 8 - (num_bits % 8)
        bits_uint8 = torch.cat([bits_uint8, torch.zeros(pad_size, dtype=torch.uint8, device=bits.device)])
    bits_reshaped = bits_uint8.view(num_packed, 8)

    packed_ref = torch.zeros(num_packed, dtype=torch.uint8, device=bits.device)
    for i in range(8):
        packed_ref = packed_ref | (bits_reshaped[:, i] << (7 - i))

    packed.copy_(packed_ref)
```

Input / output contract:
checker.type: custom
checker.language: python
time_limit_ms: 10000
memory_limit_mb: 4096
run_samples: True
subtasks: 1
testcases: 5

Requirements:
1. Return compilable CUDA/C++ code only.
2. Preserve semantics and constraints.
````

#### Reference Code（完整）

````text
def baseline(bits, packed, num_bits):
    """
    PyTorch 参考实现：Packbits

    参数说明：
    * bits: 输入张量（bool），shape (num_bits,)，底层按 1 字节布尔值存储，只读
    * packed: 输出张量（uint8），shape (ceil(num_bits / 8),)，需写入结果
    * num_bits: 输入 bit 数量（int64）
    """
    num_packed = (num_bits + 7) // 8
    bits_uint8 = bits.to(torch.uint8)
    if num_bits % 8 != 0:
        pad_size = 8 - (num_bits % 8)
        bits_uint8 = torch.cat([bits_uint8, torch.zeros(pad_size, dtype=torch.uint8, device=bits.device)])
    bits_reshaped = bits_uint8.view(num_packed, 8)

    packed_ref = torch.zeros(num_packed, dtype=torch.uint8, device=bits.device)
    for i in range(8):
        packed_ref = packed_ref | (bits_reshaped[:, i] << (7 - i))

    packed.copy_(packed_ref)
````


### 4.9 multikernelbench

- `sample_id`: `multikernelbench__9391019e25f167ba`
- `source_split`: `eval`
- `task_family`: `ref_code_to_kernel`
- `target_backend`: `cuda`
- `upstream_id`: `reference/attention/group_query_attention.py`
- `file_path`: `reference/attention/group_query_attention.py`
- `reference_code_length`: `2016`

#### Query（完整）

````text
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is:

```
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(4096, 393216)
    b = torch.randn(4096, 393216)
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```

The example new arch with custom CUDA kernels looks like this:

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
```

You are given the following architecture:

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Grouped-Query Attention (GQA)
    -----------------------------
    Like LLaMA-style attention: multiple query heads share a smaller set of key/value heads.
    """
    def __init__(self, d_model=1024, num_heads=16, num_kv_heads=4):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.v_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, H_kv = self.num_heads, self.num_kv_heads
        head_dim = D // H

        # projections
        Q = self.q_proj(x).view(B, L, H, head_dim)
        K = self.k_proj(x).view(B, L, H_kv, head_dim)
        V = self.v_proj(x).view(B, L, H_kv, head_dim)

        # Expand K/V to match query groups
        if self.group_size > 1:
            K = K.repeat_interleave(self.group_size, dim=2)
            V = V.repeat_interleave(self.group_size, dim=2)

        # attention
        attn = torch.einsum("blhd,bmhd->bh lm", Q, K) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhlm,bmhd->blhd", attn, V).reshape(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------
# Example kernelbench configuration
# ---------------------------------------------------------------------
batch_size = 8
seq_len = 128
d_model = 1024
num_heads = 16
num_kv_heads = 4

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, num_kv_heads]
```

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
````

#### Reference Code（完整）

````text
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Grouped-Query Attention (GQA)
    -----------------------------
    Like LLaMA-style attention: multiple query heads share a smaller set of key/value heads.
    """
    def __init__(self, d_model=1024, num_heads=16, num_kv_heads=4):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.v_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, H_kv = self.num_heads, self.num_kv_heads
        head_dim = D // H

        # projections
        Q = self.q_proj(x).view(B, L, H, head_dim)
        K = self.k_proj(x).view(B, L, H_kv, head_dim)
        V = self.v_proj(x).view(B, L, H_kv, head_dim)

        # Expand K/V to match query groups
        if self.group_size > 1:
            K = K.repeat_interleave(self.group_size, dim=2)
            V = V.repeat_interleave(self.group_size, dim=2)

        # attention
        attn = torch.einsum("blhd,bmhd->bh lm", Q, K) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhlm,bmhd->blhd", attn, V).reshape(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------
# Example kernelbench configuration
# ---------------------------------------------------------------------
batch_size = 8
seq_len = 128
d_model = 1024
num_heads = 16
num_kv_heads = 4

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, num_kv_heads]
````
