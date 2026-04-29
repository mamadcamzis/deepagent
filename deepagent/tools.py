"""
deepagent.tools
===============
Built-in tools that an agent can call and utilities for defining custom tools.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool registry and decorator
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, "Tool"] = {}


class Tool:
    """Represents a callable capability exposed to an agent.

    Parameters
    ----------
    name:
        Short, unique identifier used by the LLM to invoke the tool.
    description:
        Human-readable description that explains **when** and **how** to use
        the tool.  Good descriptions lead to better agent decisions.
    func:
        The Python callable that implements the tool logic.  It must accept
        a single *str* argument (the raw input from the LLM) and return a
        *str* result.
    """

    def __init__(self, name: str, description: str, func: Callable[[str], str]) -> None:
        self.name = name
        self.description = description
        self._func = func

    # ------------------------------------------------------------------

    def run(self, tool_input: str) -> str:
        """Execute the tool and return its string output."""
        try:
            return str(self._func(tool_input))
        except Exception as exc:  # noqa: BLE001
            return f"[Tool error] {exc}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tool(name={self.name!r})"


def tool(name: str, description: str) -> Callable:
    """Decorator for registering a function as a :class:`Tool`.

    Usage::

        @tool("calculator", "Evaluate a mathematical expression.")
        def calculator(expr: str) -> str:
            return str(eval(expr))  # noqa: S307
    """

    def decorator(func: Callable) -> Tool:
        t = Tool(name=name, description=description, func=func)
        _REGISTRY[name] = t
        return t

    return decorator


def get_tool(name: str) -> Optional[Tool]:
    """Look up a registered tool by name, or *None* if not found."""
    return _REGISTRY.get(name)


def list_tools() -> List[Tool]:
    """Return all currently registered tools."""
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@tool(
    name="calculator",
    description=(
        "Evaluate a safe arithmetic expression and return the numeric result. "
        "Input: a string like '2 + 2', '(10 * 3) / 2', 'sqrt(16)', etc. "
        "Supports +, -, *, /, **, sqrt, abs, round, and common math constants."
    ),
)
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression."""
    _SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    _SAFE_NAMES["abs"] = abs
    _SAFE_NAMES["round"] = round

    # Allow only safe characters
    if re.search(r"[^0-9\s\+\-\*/\(\)\.\,a-zA-Z_]", expression):
        raise ValueError(f"Unsafe expression: {expression!r}")

    result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
    return str(result)


@tool(
    name="word_count",
    description=(
        "Count the number of words in a given text. "
        "Input: any string of text. Output: integer word count."
    ),
)
def word_count(text: str) -> str:
    """Count words in text."""
    return str(len(text.split()))


@tool(
    name="json_formatter",
    description=(
        "Pretty-print a JSON string. "
        "Input: a valid JSON string. Output: indented, human-readable JSON."
    ),
)
def json_formatter(json_str: str) -> str:
    """Pretty-print a JSON string."""
    parsed = json.loads(json_str)
    return json.dumps(parsed, indent=2)


@tool(
    name="text_reverser",
    description=(
        "Reverse the characters of a given string. "
        "Input: any string. Output: the reversed string."
    ),
)
def text_reverser(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@tool(
    name="list_creator",
    description=(
        "Split a comma-separated string into a numbered list. "
        "Input: items separated by commas. Output: numbered list as text."
    ),
)
def list_creator(items_str: str) -> str:
    """Turn a comma-separated string into a numbered list."""
    items = [item.strip() for item in items_str.split(",") if item.strip()]
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
