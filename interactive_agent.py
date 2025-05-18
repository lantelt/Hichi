import os
import asyncio
import openai
from flask import Flask, request, render_template_string, session
from uuid import uuid4

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

openai.api_key = API_KEY
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


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
        "Generate or update application source code based on the design."
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



class Agent:
    def __init__(self, role: str, prompt: str):
        self.role = role
        self.prompt = prompt

    async def ask(self, messages):
        try:
            response = await openai.ChatCompletion.acreate(
                model=MODEL,
                messages=[{"role": "system", "content": self.prompt}] + messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error from {self.role} agent: {e}]"


AGENTS = {role: Agent(role, prompt) for role, prompt in SYSTEM_PROMPTS.items()}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev")
chat_logs = {}


async def _ask_agent(role: str, messages):
    return await AGENTS[role].ask(messages)


async def _run_agent(role: str, conversation, state):
    """Call an agent and record its output in state."""
    reply = await _ask_agent(role, conversation)
    conversation.append({"role": "assistant", "content": reply})
    state[role] = reply
    return reply


async def _orchestrate(user_input: str, max_iterations: int = 1):
    """Run the defined agents in sequence and collect their outputs."""
    conversation = [{"role": "user", "content": user_input}]
    state = {}

    workflow = [
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

    for role in workflow:
        await _run_agent(role, conversation, state)

    evaluation = await _run_agent("solution_evaluation", conversation, state)
    iteration = 0

    improve_flow = ["code_generation", "code_review", "test_generation"]
    while "IMPROVE" in evaluation.upper() and iteration < max_iterations:
        conversation.append({"role": "assistant", "content": evaluation})
        for role in improve_flow:
            await _run_agent(role, conversation, state)
        evaluation = await _run_agent("solution_evaluation", conversation, state)
        iteration += 1

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
            log.append({"sender": "User", "text": msg})
            state, evaluation = asyncio.run(_orchestrate(msg))
            for role, text in state.items():
                log.append({"sender": role, "text": text})
            log.append({"sender": "Evaluator", "text": evaluation})
    return render_template_string(TEMPLATE, log=log)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
