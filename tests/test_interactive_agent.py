import os
import sys
import types
import importlib
import pytest

# Stub openai so module import succeeds without network
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

os.environ.setdefault("OPENAI_API_KEY", "test")

interactive_agent = importlib.import_module("interactive_agent")


@pytest.mark.asyncio
async def test_orchestrate_uses_adk(monkeypatch):
    created = []

    class FakeAgent:
        def __init__(self, name, prompt):
            self.name = name
            created.append(name)
        async def run(self, inp):
            return f"{self.name} response"

    class FakeSequential:
        def __init__(self, sub_agents):
            self.sub_agents = sub_agents

    class FakeLoop:
        def __init__(self, main_agent, improvement_agents, evaluator, max_loops):
            self.main_agent = main_agent
            self.improvement_agents = improvement_agents
            self.evaluator = evaluator
            self.max_loops = max_loops
        async def run(self, user_input):
            state = {agent.name: f"{agent.name} response" for agent in self.main_agent.sub_agents}
            state[self.evaluator.name] = "APPROVED"
            return state, "APPROVED"

    monkeypatch.setattr(interactive_agent, "LlmAgent", FakeAgent)
    monkeypatch.setattr(interactive_agent, "SequentialAgent", FakeSequential)
    monkeypatch.setattr(interactive_agent, "LoopAgent", FakeLoop)

    state, evaluation = await interactive_agent._orchestrate("build")

    expected_order = interactive_agent.WORKFLOW_ORDER + interactive_agent.IMPROVEMENT_FLOW + ["solution_evaluation"]
    assert created == expected_order
    assert evaluation == "APPROVED"
    for role in interactive_agent.WORKFLOW_ORDER:
        assert state[role] == f"{role} response"
    assert state["solution_evaluation"] == "APPROVED"
