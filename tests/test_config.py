"""Tests for deepagent.config"""
import os
import pytest
from deepagent.config import DeepAgentConfig


def test_defaults():
    cfg = DeepAgentConfig()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_iterations == 10
    assert cfg.temperature == 0.0
    assert cfg.verbose is False


def test_custom_values():
    cfg = DeepAgentConfig(model="gpt-4o", max_iterations=3, temperature=0.7, verbose=True)
    assert cfg.model == "gpt-4o"
    assert cfg.max_iterations == 3
    assert cfg.temperature == 0.7
    assert cfg.verbose is True


def test_resolved_api_key_explicit():
    cfg = DeepAgentConfig(api_key="test-key")
    assert cfg.resolved_api_key() == "test-key"


def test_resolved_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    cfg = DeepAgentConfig()
    assert cfg.resolved_api_key() == "env-key"


def test_resolved_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = DeepAgentConfig()
    with pytest.raises(EnvironmentError, match="No API key found"):
        cfg.resolved_api_key()


def test_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("DEEPAGENT_MODEL", "gpt-4o")
    monkeypatch.setenv("DEEPAGENT_MAX_ITERATIONS", "3")
    monkeypatch.setenv("DEEPAGENT_TEMPERATURE", "0.5")
    monkeypatch.setenv("DEEPAGENT_VERBOSE", "true")
    cfg = DeepAgentConfig.from_env()
    assert cfg.model == "gpt-4o"
    assert cfg.max_iterations == 3
    assert cfg.temperature == 0.5
    assert cfg.verbose is True
