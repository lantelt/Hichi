import os
import asyncio
import openai
from flask import Flask, request, render_template_string, session
from adk import LlmAgent, SequentialAgent, LoopAgent, MCPToolset
from uuid import uuid4

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

openai.api_key = API_KEY
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Optional MCP configuration for the ADK
MCP_URL = os.getenv("ADK_MCP_URL")
MCP_TOKEN = os.getenv("ADK_MCP_TOKEN")

_MCP_KWARGS = {}
if MCP_URL:
    _MCP_KWARGS["mcp_url"] = MCP_URL
if MCP_TOKEN:
    _MCP_KWARGS["mcp_token"] = MCP_TOKEN

_TOOLSET = None
if MCP_URL and MCP_TOKEN:
    _TOOLSET = MCPToolset(mcp_url=MCP_URL, mcp_token=MCP_TOKEN)
    _MCP_KWARGS["toolset"] = _TOOLSET


SYSTEM_PROMPTS = {
    "market_research": (
        "You are the market research agent. Analyse trends, competitors and "
        "customer demand for the given business idea and report your findings."
    ),
    "demand_analysis": (
        "Using the market research, define the main problem to solve and "
        "describe user pain points."
    ),
    "requirement_def": (
        "List key features and outline a high level solution addressing the "
        "identified problem."
    ),
    "system_design": (
        "Design the overall architecture including major components and their "
        "interactions."
    ),
    "server_design": (
        "Choose frameworks, define API structure and sketch the database schema."
    ),
    "infrastructure": (
        "Provide Docker and Kubernetes configuration snippets for the target "
        "environment."
    ),
    "db_tuning": (
        "Refine the database schema and suggest indexing or optimisation."
    ),
    "code_generation": (
        "Generate or update application source code based on the design. "
        "Use available tools like 'python' for quick execution when helpful."
    ),
    "code_review": (
        "Review the current code and suggest fixes or improvements."
    ),
    "test_generation": (
        "Write unit tests and report the results of running them."
    ),
    "security_audit": (
        "Identify potential security vulnerabilities in code and configuration."
    ),
    "deployment": (
        "Describe the steps required to deploy the application."
    ),
    "project_manager": (
        "Summarise progress so far and coordinate next steps."
    ),
    "solution_evaluation": (
        "Evaluate whether the solution satisfies the original request. Respond "
        "with 'APPROVED' or 'IMPROVE:' followed by feedback."
    ),
}





app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev")
chat_logs = {}

# Maximum number of entries kept in memory per session.
MAX_LOG_LENGTH = int(os.getenv("MAX_LOG_LENGTH", "100"))

# Directory for persisting pruned log entries
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _store_old_entry(sid: str, entry: dict):
    """Append a pruned log entry to a session log file."""
    path = os.path.join(LOG_DIR, f"{sid}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{entry['sender']}: {entry['text']}\n")


def _append_and_prune(log: list, entry: dict, sid: str):
    """Append an entry to the in-memory log and remove old ones."""
    log.append(entry)
    while len(log) > MAX_LOG_LENGTH:
        old = log.pop(0)
        _store_old_entry(sid, old)





WORKFLOW_ORDER = [
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
]

IMPROVEMENT_FLOW = ["code_generation", "code_review", "test_generation"]


def _build_workflow(max_iterations: int = 1):
    sub_agents = [
        LlmAgent(
            name=role,
            prompt=SYSTEM_PROMPTS[role],
            model=MODEL,
            output_key=role,
            **_MCP_KWARGS,
        )
        for role in WORKFLOW_ORDER
    ]
    seq = SequentialAgent(sub_agents=sub_agents)
    improvement_agents = [
        LlmAgent(
            name=role,
            prompt=SYSTEM_PROMPTS[role],
            model=MODEL,
            output_key=role,
            **_MCP_KWARGS,
        )
        for role in IMPROVEMENT_FLOW
    ]
    evaluator = LlmAgent(
        name="solution_evaluation",
        prompt=SYSTEM_PROMPTS["solution_evaluation"],
        model=MODEL,
        output_key="solution_evaluation",
        **_MCP_KWARGS,
    )
    return LoopAgent(
        main_agent=seq,
        improvement_agents=improvement_agents,
        evaluator=evaluator,
        max_loops=max_iterations,
    )


async def _orchestrate(user_input: str, max_iterations: int = 1):
    workflow = _build_workflow(max_iterations)
    state, evaluation = await workflow.run(user_input)
    return state, evaluation


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Interactive Coding Agent</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      .message { margin-bottom: 1em; }
      .sender { font-weight: bold; }
      pre { background: #f4f4f4; padding: 10px; }
    </style>
  </head>
  <body>
    <h1>Interactive Coding Agent</h1>
    <div id="log">
    {% for entry in log %}
      <div class="message">
        <span class="sender">{{ entry['sender'] }}:</span>
        <pre><code>{{ entry['text'] }}</code></pre>
      </div>
    {% endfor %}
    </div>
    <form method="post">
      <textarea name="message" rows="4" style="width:80%" autofocus></textarea><br/>
      <button type="submit">Send</button>
    </form>
  </body>
</html>
"""


@app.before_request
def ensure_session():
    session.permanent = True
    if "id" not in session:
        session["id"] = str(uuid4())


@app.route("/", methods=["GET", "POST"])
def index():
    sid = session["id"]
    log = chat_logs.setdefault(sid, [])
    if request.method == "POST":
        msg = request.form.get("message", "").strip()
        if msg:
            _append_and_prune(log, {"sender": "User", "text": msg}, sid)
            state, evaluation = asyncio.run(_orchestrate(msg))
            for role, text in state.items():
                _append_and_prune(log, {"sender": role, "text": text}, sid)
            _append_and_prune(log, {"sender": "Evaluator", "text": evaluation}, sid)
    return render_template_string(TEMPLATE, log=log)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
