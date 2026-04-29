"""
deepagent.memory
================
Conversation memory and context management for DeepAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str
    name: Optional[str] = None  # tool name when role == "tool"

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Memory implementations
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Stores the full conversation history (no summarisation).

    This is the simplest memory strategy: every message is retained and
    passed to the LLM on each call.  For very long conversations the token
    count will grow unbounded – use :class:`SlidingWindowMemory` if you
    need a fixed-size window.

    Parameters
    ----------
    system_prompt:
        Optional system message prepended to every conversation.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self._messages: List[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    # ------------------------------------------------------------------

    def add(self, role: Role, content: str, *, name: Optional[str] = None) -> None:
        """Append a message to the history."""
        self._messages.append(Message(role=role, content=content, name=name))

    def get_messages(self) -> List[Message]:
        """Return the full list of messages."""
        return list(self._messages)

    def to_dicts(self) -> List[dict]:
        """Return messages serialised as plain dicts (for LLM APIs)."""
        return [m.to_dict() for m in self._messages]

    def clear(self, keep_system: bool = True) -> None:
        """Clear the history, optionally preserving the system prompt."""
        if keep_system and self._messages and self._messages[0].role == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConversationMemory(messages={len(self._messages)})"


class SlidingWindowMemory(ConversationMemory):
    """Retains only the *k* most recent non-system messages.

    Parameters
    ----------
    window_size:
        Maximum number of non-system messages to keep in memory.
    system_prompt:
        Optional system message prepended to every conversation.
    """

    def __init__(self, window_size: int = 10, system_prompt: str = "") -> None:
        super().__init__(system_prompt=system_prompt)
        self.window_size = window_size

    def add(self, role: Role, content: str, *, name: Optional[str] = None) -> None:
        super().add(role, content, name=name)
        self._trim()

    def _trim(self) -> None:
        system = [m for m in self._messages if m.role == "system"]
        non_system = [m for m in self._messages if m.role != "system"]
        if len(non_system) > self.window_size:
            non_system = non_system[-self.window_size :]
        self._messages = system + non_system

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SlidingWindowMemory(window_size={self.window_size}, "
            f"messages={len(self._messages)})"
        )
