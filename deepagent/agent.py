"""
deepagent.agent
===============
Core Agent implementation for the DeepAgent SDK.

The agent follows a simple **ReAct** (Reasoning + Acting) loop:

1. Build a prompt that includes the available tools and conversation history.
2. Ask the LLM what to do next.
3. If the LLM wants to call a tool → run the tool and feed the result back.
4. If the LLM has a final answer → return it.

The loop repeats until a final answer is produced or ``max_iterations`` is
reached.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Union

from .config import DeepAgentConfig
from .memory import ConversationMemory, SlidingWindowMemory
from .tools import Tool, list_tools


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

class AgentResponse:
    """The result returned by :meth:`Agent.run`.

    Attributes
    ----------
    output:
        The final textual answer produced by the agent.
    steps:
        A list of intermediate reasoning/action steps taken during the run.
    iterations:
        The number of LLM calls made.
    """

    def __init__(self, output: str, steps: List[dict], iterations: int) -> None:
        self.output = output
        self.steps = steps
        self.iterations = iterations

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentResponse(output={self.output!r}, "
            f"iterations={self.iterations}, steps={len(self.steps)})"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to the following tools:

{tool_descriptions}

To use a tool, respond in this exact JSON format (and nothing else):
{{
  "action": "tool_name",
  "action_input": "input string for the tool"
}}

When you have enough information to answer the user's question directly, respond in this exact JSON format:
{{
  "action": "final_answer",
  "action_input": "your complete answer here"
}}

Always respond with valid JSON. Do not include any text outside the JSON object.
"""


class Agent:
    """A ReAct-style LLM agent that can use tools to answer questions.

    Parameters
    ----------
    config:
        SDK configuration object (model, API key, etc.).  If *None* a default
        config is created from environment variables.
    tools:
        Extra :class:`~deepagent.tools.Tool` instances to make available in
        addition to the globally registered built-in tools.
    memory:
        A memory object for storing conversation history.  If *None* a fresh
        :class:`~deepagent.memory.ConversationMemory` is created.
    system_prompt:
        Override the default system prompt.

    Examples
    --------
    >>> from deepagent import Agent, DeepAgentConfig
    >>> config = DeepAgentConfig(model="gpt-4o-mini", verbose=True)
    >>> agent = Agent(config=config)
    >>> response = agent.run("What is 17 * 23?")
    >>> print(response.output)
    391
    """

    def __init__(
        self,
        config: Optional[DeepAgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Union[ConversationMemory, SlidingWindowMemory]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.config = config or DeepAgentConfig.from_env()
        self._extra_tools = tools or []

        # Build tool map: built-ins + extras
        self._tools: Dict[str, Tool] = {t.name: t for t in list_tools()}
        for t in self._extra_tools:
            self._tools[t.name] = t

        # Memory
        tool_descriptions = self._build_tool_descriptions()
        default_system = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions
        )
        self._memory = memory or ConversationMemory(
            system_prompt=system_prompt or default_system
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> AgentResponse:
        """Process *user_input* and return an :class:`AgentResponse`.

        Parameters
        ----------
        user_input:
            The message or question from the user.
        """
        self._memory.add("user", user_input)

        steps: List[dict] = []
        iterations = 0

        for _ in range(self.config.max_iterations):
            iterations += 1
            raw = self._call_llm(self._memory.to_dicts())

            if self.config.verbose:
                print(f"[DeepAgent] iteration={iterations} | llm_output={raw!r}")

            action, action_input = self._parse_llm_output(raw)

            step: dict = {"iteration": iterations, "action": action, "input": action_input}

            if action == "final_answer":
                self._memory.add("assistant", action_input)
                step["output"] = action_input
                steps.append(step)
                return AgentResponse(output=action_input, steps=steps, iterations=iterations)

            # Run tool
            if action in self._tools:
                tool_output = self._tools[action].run(action_input)
            else:
                tool_output = f"[Unknown tool: {action!r}]"

            if self.config.verbose:
                print(f"[DeepAgent] tool={action!r} | output={tool_output!r}")

            step["output"] = tool_output
            steps.append(step)

            # Feed tool result back into memory
            self._memory.add(
                "assistant",
                json.dumps({"action": action, "action_input": action_input}),
            )
            self._memory.add("tool", tool_output, name=action)

        # Fallback if max_iterations reached
        fallback = "I reached the maximum number of steps without a definitive answer."
        return AgentResponse(output=fallback, steps=steps, iterations=iterations)

    def reset(self) -> None:
        """Clear conversation memory (keeps the system prompt)."""
        self._memory.clear(keep_system=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tool_descriptions(self) -> str:
        all_tools = {**{t.name: t for t in list_tools()}, **{t.name: t for t in self._extra_tools}}
        lines = []
        for t in all_tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines) if lines else "(no tools available)"

    def _call_llm(self, messages: List[dict]) -> str:
        """Call the LLM and return the raw text response."""
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use DeepAgent with an LLM. "
                "Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.config.resolved_api_key())
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _parse_llm_output(self, raw: str) -> tuple[str, str]:
        """Parse the LLM output and return (action, action_input)."""
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
            action = str(parsed.get("action", "final_answer"))
            action_input = str(parsed.get("action_input", raw))
            return action, action_input
        except json.JSONDecodeError:
            # Fallback: treat entire output as final answer
            return "final_answer", raw
