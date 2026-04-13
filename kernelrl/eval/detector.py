"""
Code extraction detector for LLM outputs.

Extracts code snippets from LLM-generated text, handling various formats:
- Markdown code blocks (```lang ... ```)
- Plain code blocks
- Mixed text and code
"""

import re
from typing import List, Optional, Union


def detector(text: str) -> Union[str, List[str], None]:
    """
    Extract code from LLM output text.

    Handles multiple formats:
    1. Markdown code blocks (```python ... ```, ```cuda ... ```, etc.)
    2. Plain indented code blocks
    3. Mixed text with code

    Returns:
    - str: Single code block if found
    - List[str]: Multiple code blocks if found
    - None: If no code detected

    Args:
        text: LLM output text containing code

    Example:
        >>> text = "Here's a Python function:\\n```python\\ndef add(a, b):\\n    return a + b\\n```"
        >>> code = detector(text)
        >>> print(code)
        def add(a, b):
            return a + b
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Extract all markdown code blocks (```...```)
    code_blocks = _extract_markdown_blocks(text)

    if not code_blocks:
        # If no markdown blocks, try to extract plain code
        code_blocks = _extract_plain_code(text)

    if not code_blocks:
        return None

    # Return single block or list
    if len(code_blocks) == 1:
        return code_blocks[0]
    else:
        return code_blocks


def _extract_markdown_blocks(text: str) -> List[str]:
    """
    Extract code from markdown code blocks (```language ... ```).

    Supports:
    - ```python ... ```
    - ```cuda ... ```
    - ```cpp ... ```
    - ``` ... ``` (no language specified)
    """
    blocks = []

    # Pattern: ```[optional language]...\n...\n```
    # More flexible pattern that handles various formats
    pattern = r'```(?:[a-z0-9+\-#]*)\n?(.*?)\n?```'

    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        code = match.strip()
        if code:
            blocks.append(code)

    return blocks


def _extract_plain_code(text: str) -> List[str]:
    """
    Extract code from plain indented blocks or code-like sections.

    Heuristics:
    - Continuous lines with leading whitespace
    - Lines containing common code patterns (def, class, import, etc.)
    - Lines containing common programming syntax
    """
    blocks = []
    lines = text.split('\n')

    current_block: List[str] = []
    in_code_block = False
    in_triple_single = False
    in_triple_double = False

    def _update_triple_quote_state(line: str) -> None:
        nonlocal in_triple_single, in_triple_double
        if not in_triple_double:
            single_count = len(re.findall(r"(?<!\\\\)'''", line))
            if single_count % 2 == 1:
                in_triple_single = not in_triple_single
        if not in_triple_single:
            double_count = len(re.findall(r'(?<!\\\\)"""', line))
            if double_count % 2 == 1:
                in_triple_double = not in_triple_double

    def _flush_block() -> None:
        nonlocal current_block, in_code_block, in_triple_single, in_triple_double
        if in_code_block and current_block:
            block = '\n'.join(current_block).strip('\n')
            if block:
                blocks.append(block)
        current_block = []
        in_code_block = False
        in_triple_single = False
        in_triple_double = False

    for line in lines:
        stripped = line.strip()
        looks_like_code = _is_code_line(line)

        if in_code_block and (in_triple_single or in_triple_double):
            looks_like_code = True
        elif in_code_block and not looks_like_code:
            if stripped == "":
                looks_like_code = True
            elif _looks_like_code_continuation(line):
                looks_like_code = True

        if looks_like_code:
            if not in_code_block:
                in_code_block = True
            current_block.append(line.rstrip())
            _update_triple_quote_state(line)
        else:
            _flush_block()

    _flush_block()
    return blocks


def _looks_like_code_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if stripped.startswith(('#', '@', ')', ']', '}', ',', '.', ';')):
        return True

    indent = len(line) - len(line.lstrip())
    if indent > 0:
        return True

    if re.search(r'[=(){}\[\]:,+\-*/%<>]', stripped):
        return True

    return False


def _is_code_line(line: str) -> bool:
    """
    Determine if a line looks like code.

    Checks for:
    - Python/C++/CUDA keywords
    - Function/class definitions
    - Comments
    - Common syntax patterns
    - Significant leading whitespace (likely indentation)
    """
    stripped = line.strip()

    if not stripped or stripped.startswith('#'):
        return False

    # Common code keywords and patterns
    code_patterns = [
        # Python
        r'^\s*(def|class|import|from|if|else|for|while|try|except|with|return|yield)',
        r'^\s*[a-zA-Z_]\w*\s*=',  # Assignment
        r'^\s*[a-zA-Z_]\w*\(',      # Function call
        r'^\s*[a-zA-Z_][\w\.]*\[',   # Indexing

        # C++/CUDA
        r'^\s*(int|float|double|void|bool|char|auto|inline|const|struct|class)\b',
        r'^\s*#include',
        r'^\s*#define',
        r'^\s*\{\s*$',
        r'^\s*\}\s*;?\s*$',
        r'^\s*__global__',
        r'^\s*__device__',

        # Common operators and syntax
        r'\s*(==|!=|<=|>=|&&|\|\||->|::)',
        r'\(\s*\)',  # Empty parens
        r'\[\s*\]',  # Empty brackets
    ]

    for pattern in code_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    # Check for significant indentation (likely code block)
    if len(line) - len(line.lstrip()) >= 4:
        # Has meaningful indentation
        if stripped and not stripped.startswith('-'):  # Not a list item
            return True

    return False


# Aliases for backward compatibility
extract_code = detector
extract_kernel_code = detector
