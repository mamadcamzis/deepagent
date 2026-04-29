"""Tests for deepagent.memory"""
import pytest
from deepagent.memory import ConversationMemory, Message, SlidingWindowMemory


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

def test_message_to_dict_basic():
    m = Message(role="user", content="hello")
    d = m.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_message_to_dict_with_name():
    m = Message(role="tool", content="result", name="calculator")
    d = m.to_dict()
    assert d["name"] == "calculator"


# ---------------------------------------------------------------------------
# ConversationMemory tests
# ---------------------------------------------------------------------------

def test_conversation_memory_empty():
    mem = ConversationMemory()
    assert len(mem) == 0


def test_conversation_memory_system_prompt():
    mem = ConversationMemory(system_prompt="Be helpful.")
    assert len(mem) == 1
    assert mem.get_messages()[0].role == "system"


def test_conversation_memory_add():
    mem = ConversationMemory()
    mem.add("user", "hello")
    mem.add("assistant", "hi")
    assert len(mem) == 2


def test_conversation_memory_to_dicts():
    mem = ConversationMemory()
    mem.add("user", "test")
    dicts = mem.to_dicts()
    assert dicts == [{"role": "user", "content": "test"}]


def test_conversation_memory_clear_keeps_system():
    mem = ConversationMemory(system_prompt="System.")
    mem.add("user", "msg1")
    mem.add("assistant", "reply1")
    mem.clear(keep_system=True)
    assert len(mem) == 1
    assert mem.get_messages()[0].role == "system"


def test_conversation_memory_clear_all():
    mem = ConversationMemory(system_prompt="System.")
    mem.add("user", "msg1")
    mem.clear(keep_system=False)
    assert len(mem) == 0


# ---------------------------------------------------------------------------
# SlidingWindowMemory tests
# ---------------------------------------------------------------------------

def test_sliding_window_respects_size():
    mem = SlidingWindowMemory(window_size=3)
    for i in range(6):
        mem.add("user", f"message {i}")
    non_sys = [m for m in mem.get_messages() if m.role != "system"]
    assert len(non_sys) == 3


def test_sliding_window_keeps_latest():
    mem = SlidingWindowMemory(window_size=2)
    mem.add("user", "old message")
    mem.add("user", "newer message")
    mem.add("user", "newest message")
    non_sys = [m for m in mem.get_messages() if m.role != "system"]
    contents = [m.content for m in non_sys]
    assert "newest message" in contents
    assert "old message" not in contents


def test_sliding_window_preserves_system_prompt():
    mem = SlidingWindowMemory(window_size=2, system_prompt="System prompt.")
    for i in range(5):
        mem.add("user", f"msg {i}")
    msgs = mem.get_messages()
    assert msgs[0].role == "system"
    assert msgs[0].content == "System prompt."
