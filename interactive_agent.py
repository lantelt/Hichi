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
        "You are the System Architect. Your responsibility is to design a robust, "
        "scalable, and maintainable overall system architecture. Clearly define "
        "major components, their responsibilities, and the interactions between "
        "them. Consider non-functional requirements like performance, security, "
        "and scalability. Output detailed diagrams or descriptions as appropriate."
    ),
    "server_design": (
        "You are the Server Design Specialist. Your role involves choosing "
        "appropriate server-side frameworks, defining a clear and consistent API "
        "structure, and creating an initial database schema. For the database "
        "schema, focus on logical data organization; the DBA will further refine "
        "it for performance and optimization. Ensure your design choices support "
        "the architectural goals."
    ),
    "infrastructure": (
        "Provide Docker and Kubernetes configuration snippets for the target "
        "environment."
    ),
    "db_tuning": (
        "You are the Database Administrator (DBA). Your task is to refine the "
        "database schema for optimal performance, integrity, and scalability. "
        "Suggest appropriate indexing strategies, query optimizations, and "
        "address data security considerations. Ensure the schema aligns with "
        "the application's data access patterns."
    ),
    "code_generation": (
        "You are the Coding Specialist. Your responsibility is to generate "
        "high-quality, clean, maintainable, and efficient source code based on "
        "the provided design and specifications. Follow best coding practices "
        "and conventions for the chosen language/framework. Use available tools "
        "like 'python' for code execution or validation if needed. Ensure your "
        "code is well-commented."
    ),
    "code_review": (
        "You are the QA Specialist focused on Code Review. Meticulously review "
        "the generated source code for correctness, adherence to design "
        "specifications, potential bugs, performance issues, and "
        "maintainability. Provide specific, actionable suggestions for fixes or "
        "improvements."
    ),
    "test_generation": (
        "You are the QA Specialist focused on Test Generation. Your task is to "
        "create comprehensive unit tests that cover critical aspects of the "
        "generated code. Aim for high test coverage. Report the results of "
        "running these tests clearly, indicating any failures."
    ),
    "security_audit": (
        "You are the QA Specialist focused on Security Audit. Your "
        "responsibility is to identify potential security vulnerabilities in the "
        "generated code, infrastructure configuration, and data handling. Focus "
        "on common web application vulnerabilities (e.g., OWASP Top 10) and "
        "suggest mitigations."
    ),
    "deployment": (
        "Describe the steps required to deploy the application."
    ),
    "project_manager": (
        "Summarise progress so far and coordinate next steps."
    ),
    "solution_evaluation": (
        "You are the Solution Evaluation Agent. Your role is to critically assess whether the generated solution "
        "fully satisfies the original user request and meets quality standards.\n"
        "- If the solution is satisfactory and complete, respond with only the word 'APPROVED'.\n"
        "- If the solution needs improvement, respond with 'IMPROVE:' followed by a detailed, actionable critique. "
        "Clearly specify which aspects are lacking (e.g., bugs in code, unmet requirements, poor test coverage, "
        "security concerns) and provide concrete suggestions for what the 'code_generation', 'code_review', or "
        "'test_generation' agents should do next to address the issues. Vague feedback is not helpful."
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


async def _orchestrate(user_input: str, max_iterations: int = 3):
    workflow = _build_workflow(max_iterations)
    state, evaluation = await workflow.run(user_input)
    return state, evaluation


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Interactive Coding Agent</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 20px;
        background-color: #f9f9f9;
        color: #333;
        font-size: 16px; /* Slightly increased base font size */
      }
      h1 {
        text-align: center;
        color: #444;
      }
      #log {
        max-width: 800px;
        margin: 20px auto;
        padding: 15px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      .message {
        margin-bottom: 1.5em; /* Increased spacing */
        padding: 12px;
        border-radius: 6px;
        line-height: 1.6;
      }
      .sender {
        font-weight: bold;
        display: block; /* Make sender take full width for better structure */
        margin-bottom: 5px;
      }
      .timestamp {
        font-size: 0.8em;
        color: #777; /* Darker grey for better contrast */
        display: block;
        margin-bottom: 8px;
      }
      .user-message {
        background-color: #e1f5fe; /* Light blue */
        border-left: 5px solid #03a9f4; /* Accent border */
        /* text-align: right; /* This would align text, often message block is aligned instead */
      }
      .agent-message {
        background-color: #f0f0f0; /* Light grey */
        border-left: 5px solid #757575; /* Accent border */
      }
      /* Styling for user message container if we want to align the whole block */
      .message-container.user {
        display: flex;
        justify-content: flex-end;
      }
      .message-container.user .message {
         max-width: 80%; /* Prevent user message from taking full width */
      }

      pre {
        background: #2d2d2d; /* Darker background for code */
        color: #f0f0f0; /* Light text for code */
        padding: 15px;
        border-radius: 4px;
        overflow-x: auto; /* Enable horizontal scrolling for long lines */
        font-family: "Courier New", Courier, monospace;
        font-size: 0.95em;
      }
      code {
        font-family: "Courier New", Courier, monospace;
        /* background: #eee;  Optional: if inline code needs different bg */
        /* padding: 2px 4px; */
        /* border-radius: 3px; */
      }
      form {
        max-width: 800px;
        margin: 20px auto;
        padding: 15px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        display: flex; /* For aligning textarea and button */
        align-items: flex-start;
      }
      textarea {
        flex-grow: 1;
        min-height: 60px; /* Increased height */
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 1em;
        margin-right: 10px;
      }
      button {
        padding: 10px 20px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1em;
        transition: background-color 0.2s ease;
      }
      button:hover {
        background-color: #0056b3;
      }
    </style>
  </head>
  <body>
    <h1>Interactive Coding Agent</h1>
    <div id="log">
    {% for entry in log %}
      {# Determine if the message is from the user to apply container-level alignment #}
      <div class="message-container {% if entry['sender'] == 'User' %}user{% else %}agent{% endif %}">
        <div class="message {{ 'user-message' if entry['sender'] == 'User' else 'agent-message' }}">
          <span class="sender">{{ entry['sender'] }}</span>
          {# Display timestamp if available, otherwise nothing #}
          {% if entry.timestamp %}
            <span class="timestamp">{{ entry.timestamp }}</span>
          {% else %}
            <span class="timestamp">Timestamp not available</span> {# Placeholder #}
          {% endif %}
          <pre><code>{{ entry['text'] }}</code></pre>
        </div>
      </div>
    {% endfor %}
    </div>
    <form method="post">
      <textarea name="message" rows="4" autofocus placeholder="Type your message here..."></textarea>
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
