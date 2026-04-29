"""
deepagent.config
================
Configuration management for the DeepAgent SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepAgentConfig:
    """Global configuration for DeepAgent.

    Parameters
    ----------
    model:
        The LLM model identifier to use (e.g. ``"gpt-4o"``).
    api_key:
        API key for the model provider.  When *None* the value is read from
        the ``OPENAI_API_KEY`` environment variable at runtime.
    max_iterations:
        Maximum number of reasoning iterations the agent may perform before
        it stops and returns its best answer.
    temperature:
        Sampling temperature passed to the underlying LLM (0 = deterministic).
    verbose:
        When *True* the agent prints a trace of each step to *stdout*.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    max_iterations: int = 10
    temperature: float = 0.0
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolved_api_key(self) -> str:
        """Return the API key, falling back to the environment variable."""
        key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "No API key found. Pass api_key= to DeepAgentConfig or set "
                "the OPENAI_API_KEY environment variable."
            )
        return key

    # Convenience constructor -----------------------------------------

    @classmethod
    def from_env(cls, **overrides) -> "DeepAgentConfig":
        """Build a config from environment variables, applying *overrides*."""
        return cls(
            model=os.getenv("DEEPAGENT_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_iterations=int(os.getenv("DEEPAGENT_MAX_ITERATIONS", "10")),
            temperature=float(os.getenv("DEEPAGENT_TEMPERATURE", "0.0")),
            verbose=os.getenv("DEEPAGENT_VERBOSE", "").lower() in {"1", "true", "yes"},
            **overrides,
        )
