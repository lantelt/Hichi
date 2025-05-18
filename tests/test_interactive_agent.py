import os
import sys
import types
import importlib
import asyncio
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
os.environ.setdefault("ADK_MCP_URL", "http://mcp")
os.environ.setdefault("ADK_MCP_TOKEN", "token")

# Stub flask and adk modules so interactive_agent imports succeed
flask_stub = types.SimpleNamespace(
    Flask=type(
        "Flask",
        (),
        {
            "__init__": lambda self, name: None,
            "before_request": lambda self, f: f,
            "route": lambda self, *a, **kw: (lambda func: func),
            "run": lambda self, *a, **kw: None,
        },
    ),
    request=types.SimpleNamespace(method="GET", form={}),
    render_template_string=lambda *a, **kw: "",
    session={},
)
adk_stub = types.SimpleNamespace(
    LlmAgent=object,
    SequentialAgent=object,
    LoopAgent=object,
)
sys.modules.setdefault("flask", flask_stub)
sys.modules.setdefault("adk", adk_stub)

interactive_agent = importlib.import_module("interactive_agent")


class OrchestrateUsesAdkTest(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrate_uses_adk(self):
        created = []
        used_models = []
        used_kwargs = []

        class FakeAgent:
            def __init__(self, name, prompt, model=None, **kwargs):
                self.name = name
                created.append(name)
                used_models.append(model)
                used_kwargs.append(kwargs)

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
                state = {
                    agent.name: f"{agent.name} response"
                    for agent in self.main_agent.sub_agents
                }
                state[self.evaluator.name] = "APPROVED"
                return state, "APPROVED"

        patcher1 = mock.patch.object(interactive_agent, "LlmAgent", FakeAgent)
        patcher2 = mock.patch.object(interactive_agent, "SequentialAgent", FakeSequential)
        patcher3 = mock.patch.object(interactive_agent, "LoopAgent", FakeLoop)
        patcher1.start(); patcher2.start(); patcher3.start()
        try:
            state, evaluation = await interactive_agent._orchestrate("build")
        finally:
            patcher1.stop(); patcher2.stop(); patcher3.stop()

        expected_order = (
            interactive_agent.WORKFLOW_ORDER
            + interactive_agent.IMPROVEMENT_FLOW
            + ["solution_evaluation"]
        )
        self.assertEqual(created, expected_order)
        self.assertEqual(used_models, [interactive_agent.MODEL] * len(expected_order))
        for kwargs in used_kwargs:
            self.assertEqual(kwargs.get("mcp_url"), os.environ["ADK_MCP_URL"])
            self.assertEqual(kwargs.get("mcp_token"), os.environ["ADK_MCP_TOKEN"])
        self.assertEqual(evaluation, "APPROVED")
        for role in interactive_agent.WORKFLOW_ORDER:
            self.assertEqual(state[role], f"{role} response")
        self.assertEqual(state["solution_evaluation"], "APPROVED")


if __name__ == "__main__":
    unittest.main()
