#!/usr/bin/env python3
"""
Test script for correctness module.
"""

from kernelrl.eval.eval import eval as kernelrl_eval

# Test case 1: Simple correct case
reference_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(10, 10).cuda()]

def get_init_inputs():
    return []
"""

generated_code = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(10, 10).cuda()]

def get_init_inputs():
    return []
"""

print("=" * 80)
print("Test 1: Correct implementation (x * 2 vs x * 2)")
print("=" * 80)
result = kernelrl_eval(
    generated_code=generated_code,
    reference_code=reference_code,
    language="python",
    use_compile=True,
    use_correctness=True,
    use_hack_check=False,
    use_timing=False,
)
print(f"Result: {result}")
print(f"Correct: {result.get('correct')}")
print(f"Compile OK: {result.get('compile_ok')}")
print(f"Correctness OK: {result.get('correctness_ok')}")
print()

# Test case 2: Incorrect implementation
generated_code_wrong = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 3  # Wrong! Should be x * 2

def get_inputs():
    return [torch.randn(10, 10).cuda()]

def get_init_inputs():
    return []
"""

print("=" * 80)
print("Test 2: Incorrect implementation (x * 3 vs x * 2)")
print("=" * 80)
result = kernelrl_eval(
    generated_code=generated_code_wrong,
    reference_code=reference_code,
    language="python",
    use_compile=True,
    use_correctness=True,
    use_hack_check=False,
    use_timing=False,
)
print(f"Result: {result}")
print(f"Correct: {result.get('correct')}")
print(f"Compile OK: {result.get('compile_ok')}")
print(f"Correctness OK: {result.get('correctness_ok')}")
print(f"Correctness Error: {result.get('correctness_error_type')}")
print(f"Max Diff: {result.get('correctness_max_diff')}")
print()

print("=" * 80)
print("All tests completed!")
print("=" * 80)
