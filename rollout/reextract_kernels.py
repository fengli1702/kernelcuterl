#!/usr/bin/env python3
"""
重新提取rollout结果中的kernel代码。

使用改进的提取逻辑，提取thinking之后最后一个包含ModelNew的完整代码块。
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add KernelRL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelrl.eval.detector import detector


def extract_kernel_improved(text: str) -> Optional[str]:
    """
    改进的kernel代码提取逻辑。

    策略：
    1. 提取所有代码块
    2. 找到最后一个包含ModelNew的代码块（LLM可能会多次修正）
    3. 如果没有ModelNew，返回最大的代码块
    """
    if not text:
        return None

    try:
        code = detector(text)

        # Handle single code block
        if isinstance(code, str):
            return code.strip() if code and code.strip() else None

        # Handle multiple code blocks
        if isinstance(code, (list, tuple)):
            if not code:
                return None

            # Filter out empty blocks
            non_empty_blocks = [str(c).strip() for c in code if c and str(c).strip()]
            if not non_empty_blocks:
                return None

            # Find blocks containing "class ModelNew"
            model_blocks = [b for b in non_empty_blocks if 'class ModelNew' in b or 'class Model' in b]
            if model_blocks:
                # Return the last (most refined) ModelNew block
                return model_blocks[-1]

            # If no ModelNew found, return the largest block (likely the main code)
            # Filter out small snippet blocks (< 500 chars)
            large_blocks = [b for b in non_empty_blocks if len(b) >= 500]
            if large_blocks:
                return max(large_blocks, key=len)

            # Fallback: return the first non-empty block
            return non_empty_blocks[0]

        return None
    except Exception as exc:
        print(f"Error extracting kernel: {exc}")
        return None


def reextract_file(input_path: str, output_path: str):
    """重新提取一个rollout文件的kernel代码。"""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Processing: {input_file}")
    print(f"Output to: {output_file}")

    total = 0
    improved = 0
    same = 0
    worse = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_no, line in enumerate(f_in, 1):
            try:
                result = json.loads(line)
                total += 1

                # Get original extraction
                sample = result['samples'][0]
                old_kernel = sample.get('kernel_code', '')
                old_len = len(old_kernel) if old_kernel else 0

                # Re-extract with improved logic
                text = sample.get('text', '')
                new_kernel = extract_kernel_improved(text)
                new_len = len(new_kernel) if new_kernel else 0

                # Update the kernel_code
                sample['kernel_code'] = new_kernel

                # Track improvements
                if new_len > old_len:
                    improved += 1
                    if (line_no - 1) < 5:  # Show first few
                        print(f"  #{result['idx']}: {old_len:,} -> {new_len:,} chars (+{new_len-old_len:,})")
                elif new_len == old_len:
                    same += 1
                else:
                    worse += 1

                # Write updated result
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as exc:
                print(f"Error processing line {line_no}: {exc}")
                continue

    print(f"\n{'='*80}")
    print("Summary:")
    print("="*80)
    print(f"Total records: {total}")
    print(f"Improved (longer extraction): {improved}")
    print(f"Same length: {same}")
    print(f"Worse (shorter extraction): {worse}")
    print(f"\nOutput saved to: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Re-extract kernel codes from rollout results')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('output', help='Output JSONL file')

    args = parser.parse_args()

    reextract_file(args.input, args.output)
