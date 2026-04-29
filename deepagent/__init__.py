"""
deepagent
=========
A lightweight Python SDK for building LLM-powered agents with tools and memory.

Quick start::

    from deepagent import Agent, DeepAgentConfig

    config = DeepAgentConfig(model="gpt-4o-mini", verbose=True)
    agent = Agent(config=config)
    response = agent.run("What is 42 * 17?")
    print(response.output)
"""

from .agent import Agent, AgentResponse
from .config import DeepAgentConfig
from .memory import ConversationMemory, Message, SlidingWindowMemory
from .tools import Tool, list_tools, tool

__all__ = [
    "Agent",
    "AgentResponse",
    "DeepAgentConfig",
    "ConversationMemory",
    "SlidingWindowMemory",
    "Message",
    "Tool",
    "tool",
    "list_tools",
]

__version__ = "0.1.0"
