import os
import asyncio
import openai
from flask import Flask, request, render_template_string, session
from uuid import uuid4

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

openai.api_key = API_KEY

SYSTEM_PROMPTS = {
    "architect": (
        "You are the lead software architect of a development team. "
        "Analyze the user's request and outline a high level design that the "
        "other team members can build upon."
    ),
    "dba": (
        "You are the team's database expert. Based on the architect's plan, "
        "propose relevant schema designs or queries."
    ),
    "coding": (
        "You are the implementation specialist. Cooperate with the architect "
        "and DBA plans to produce clear code snippets."
    ),
    "qa": (
        "You are the QA engineer. Review the proposed implementation and "
        "suggest tests to verify quality."
    ),
    "project_manager": (
        "You are the project manager overseeing the team. Summarize objectives, "
        "coordinate tasks, and ensure each role stays aligned."
    ),
    "evaluator": (
        "You review the entire discussion. Respond with 'APPROVED' if the "
        "solution is satisfactory, otherwise reply with 'IMPROVE:' followed by "
        "specific feedback."
    ),
}


class Agent:
    def __init__(self, role: str, prompt: str):
        self.role = role
        self.prompt = prompt

    async def ask(self, messages):
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
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


async def _orchestrate(user_input: str, max_iterations: int = 1):
    """Coordinate the agents so each builds on the prior responses."""
    conversation = [{"role": "user", "content": user_input}]

    # The architect sets the direction for the team
    arch_reply = await _ask_agent("architect", conversation)
    conversation.append({"role": "assistant", "content": arch_reply})

    # DBA responds after seeing the architect's plan
    dba_reply = await _ask_agent("dba", conversation)
    conversation.append({"role": "assistant", "content": dba_reply})

    # Coding specialist implements using prior advice
    code_reply = await _ask_agent("coding", conversation)
    conversation.append({"role": "assistant", "content": code_reply})

    # QA reviews the implementation
    qa_reply = await _ask_agent("qa", conversation)
    conversation.append({"role": "assistant", "content": qa_reply})

    # Project manager coordinates based on the latest updates
    pm_reply = await _ask_agent("project_manager", conversation)
    conversation.append({"role": "assistant", "content": pm_reply})

    evaluation = await _ask_agent("evaluator", conversation)
    iteration = 0

    # Short improvement cycle if requested by the evaluator
    while "IMPROVE" in evaluation.upper() and iteration < max_iterations:
        conversation.append({"role": "assistant", "content": evaluation})

        code_reply = await _ask_agent("coding", conversation)
        conversation.append({"role": "assistant", "content": code_reply})

        qa_reply = await _ask_agent("qa", conversation)
        conversation.append({"role": "assistant", "content": qa_reply})

        pm_reply = await _ask_agent("project_manager", conversation)
        conversation.append({"role": "assistant", "content": pm_reply})

        evaluation = await _ask_agent("evaluator", conversation)
        iteration += 1

    responses = {
        "Architect": arch_reply,
        "DBA": dba_reply,
        "Coding Specialist": code_reply,
        "QA Specialist": qa_reply,
        "Project Manager": pm_reply,
    }

    return responses, evaluation


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
            responses, evaluation = asyncio.run(_orchestrate(msg))
            for role, text in responses.items():
                log.append({"sender": role, "text": text})
            log.append({"sender": "Evaluator", "text": evaluation})
    return render_template_string(TEMPLATE, log=log)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
