"""Tests for deepagent.agent (using a mocked LLM)"""
import json
import pytest
from unittest.mock import MagicMock, patch

from deepagent.agent import Agent, AgentResponse
from deepagent.config import DeepAgentConfig
from deepagent.memory import ConversationMemory
from deepagent.tools import Tool


def _make_agent(mock_responses: list[str], **cfg_kwargs) -> tuple[Agent, MagicMock]:
    """Helper: create an Agent whose _call_llm returns canned responses."""
    config = DeepAgentConfig(api_key="test-key", verbose=False, **cfg_kwargs)
    agent = Agent(config=config)

    responses = iter(mock_responses)

    def fake_call_llm(messages):
        return next(responses)

    agent._call_llm = fake_call_llm
    return agent, None


# ---------------------------------------------------------------------------
# AgentResponse tests
# ---------------------------------------------------------------------------

def test_agent_response_str():
    resp = AgentResponse(output="hello", steps=[], iterations=1)
    assert str(resp) == "hello"


# ---------------------------------------------------------------------------
# Agent.run tests
# ---------------------------------------------------------------------------

def test_agent_direct_final_answer():
    """LLM immediately returns a final answer."""
    agent, _ = _make_agent([
        json.dumps({"action": "final_answer", "action_input": "Paris"})
    ])
    response = agent.run("What is the capital of France?")
    assert response.output == "Paris"
    assert response.iterations == 1
    assert len(response.steps) == 1


def test_agent_uses_calculator_tool():
    """LLM calls calculator once then gives final answer."""
    agent, _ = _make_agent([
        json.dumps({"action": "calculator", "action_input": "2 + 2"}),
        json.dumps({"action": "final_answer", "action_input": "The answer is 4."}),
    ])
    response = agent.run("What is 2 plus 2?")
    assert "4" in response.output
    assert response.iterations == 2


def test_agent_max_iterations_fallback():
    """Agent stops and returns fallback when max_iterations reached."""
    # Always return a tool call so the loop never terminates naturally
    always_tool = json.dumps({"action": "calculator", "action_input": "1+1"})
    agent, _ = _make_agent([always_tool] * 3, max_iterations=3)
    response = agent.run("Loop forever.")
    assert response.iterations == 3
    assert "maximum" in response.output.lower()


def test_agent_unknown_tool():
    """Unknown tool name results in error message fed back to LLM, then final answer."""
    agent, _ = _make_agent([
        json.dumps({"action": "nonexistent_tool", "action_input": "anything"}),
        json.dumps({"action": "final_answer", "action_input": "Recovered."}),
    ])
    response = agent.run("Try an unknown tool.")
    assert response.output == "Recovered."


def test_agent_malformed_json_fallback():
    """Malformed JSON from LLM is treated as a final answer."""
    agent, _ = _make_agent(["This is not JSON at all."])
    response = agent.run("Give me something weird.")
    assert response.output == "This is not JSON at all."


def test_agent_reset_clears_non_system_messages():
    memory = ConversationMemory(system_prompt="You are helpful.")
    config = DeepAgentConfig(api_key="test-key", verbose=False)
    agent = Agent(config=config, memory=memory)
    agent._call_llm = lambda _: json.dumps({"action": "final_answer", "action_input": "ok"})

    agent.run("Hello")
    assert len(memory) > 1

    agent.reset()
    assert len(memory) == 1  # only system prompt remains
    assert memory.get_messages()[0].role == "system"


def test_agent_custom_tool_is_used():
    """A custom tool passed via tools= is callable by the agent."""
    greet_tool = Tool("greeter", "Greet by name.", lambda name: f"Hello, {name}!")
    config = DeepAgentConfig(api_key="test-key", verbose=False)
    agent = Agent(config=config, tools=[greet_tool])
    agent._call_llm = lambda _: json.dumps({"action": "greeter", "action_input": "Alice"})

    # Override to provide final answer on second call
    calls = [
        json.dumps({"action": "greeter", "action_input": "Alice"}),
        json.dumps({"action": "final_answer", "action_input": "Hello, Alice!"}),
    ]
    it = iter(calls)
    agent._call_llm = lambda _: next(it)

    response = agent.run("Greet Alice.")
    assert "Alice" in response.output
