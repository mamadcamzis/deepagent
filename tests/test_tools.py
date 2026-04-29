"""Tests for deepagent.tools"""
import json
import pytest
from deepagent.tools import (
    Tool,
    calculator,
    json_formatter,
    list_creator,
    list_tools,
    text_reverser,
    tool,
    word_count,
)


# ---------------------------------------------------------------------------
# Built-in tool tests
# ---------------------------------------------------------------------------

def test_calculator_addition():
    assert calculator.run("2 + 3") == "5"


def test_calculator_complex():
    assert calculator.run("(10 * 3) / 2") == "15.0"


def test_calculator_sqrt():
    assert calculator.run("sqrt(16)") == "4.0"


def test_calculator_power():
    assert calculator.run("2 ** 10") == "1024"


def test_calculator_unsafe_expression():
    result = calculator.run("__import__('os')")
    assert "[Tool error]" in result


def test_word_count_basic():
    assert word_count.run("hello world") == "2"


def test_word_count_empty():
    assert word_count.run("") == "0"


def test_json_formatter_valid():
    result = json_formatter.run('{"a":1,"b":2}')
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": 2}
    assert "\n" in result  # indented


def test_json_formatter_invalid():
    result = json_formatter.run("not json")
    assert "[Tool error]" in result


def test_text_reverser():
    assert text_reverser.run("abc") == "cba"
    assert text_reverser.run("") == ""


def test_list_creator():
    result = list_creator.run("apple, banana, cherry")
    assert "1. apple" in result
    assert "2. banana" in result
    assert "3. cherry" in result


# ---------------------------------------------------------------------------
# Tool / @tool decorator tests
# ---------------------------------------------------------------------------

def test_custom_tool_via_constructor():
    t = Tool("upper", "Uppercase text.", lambda s: s.upper())
    assert t.run("hello") == "HELLO"
    assert repr(t) == "Tool(name='upper')"


def test_tool_decorator_registers_globally():
    @tool("test_echo", "Echo input back.")
    def echo(s: str) -> str:
        return s

    tools_map = {t.name: t for t in list_tools()}
    assert "test_echo" in tools_map
    assert tools_map["test_echo"].run("hi") == "hi"


def test_tool_error_handling():
    def bad_func(s: str) -> str:
        raise ValueError("oops")

    t = Tool("bad", "Always fails.", bad_func)
    result = t.run("anything")
    assert "[Tool error]" in result
    assert "oops" in result


def test_list_tools_returns_list():
    tools = list_tools()
    assert isinstance(tools, list)
    names = [t.name for t in tools]
    assert "calculator" in names
    assert "word_count" in names
