import os
import sys
import types
import importlib
import pytest

# Provide a stub openai module before importing interactive_agent
openai_stub = types.SimpleNamespace()
class ChatCompletion:
    @staticmethod
    async def acreate(*args, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
        )
openai_stub.ChatCompletion = ChatCompletion
openai_stub.api_key = None
sys.modules['openai'] = openai_stub

# Ensure API key env var so interactive_agent does not raise
os.environ.setdefault("OPENAI_API_KEY", "test")

interactive_agent = importlib.import_module("interactive_agent")


@pytest.mark.asyncio
async def test_orchestrate_runs_all_agents(monkeypatch):
    async def fake_ask_agent(role, messages):
        if role == "solution_evaluation":
            return "APPROVED"
        return f"{role} response"

    monkeypatch.setattr(interactive_agent, "_ask_agent", fake_ask_agent)

    state, evaluation = await interactive_agent._orchestrate("build")

    expected_roles = [
        "market_research",
        "demand_analysis",
        "requirement_def",
        "system_design",
        "server_design",
        "infrastructure",
        "db_tuning",
        "code_generation",
        "code_review",
        "test_generation",
        "security_audit",
        "deployment",
        "project_manager",
        "solution_evaluation",
    ]

    assert evaluation == "APPROVED"
    assert set(state.keys()) == set(expected_roles)
    for role in expected_roles[:-1]:
        assert state[role] == f"{role} response"
    assert state["solution_evaluation"] == "APPROVED"
