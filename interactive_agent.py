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
    "architect": "You are an experienced software architect. Provide high level design guidance for the requirement.",
    "dba": "You are an expert DBA. Offer database schema ideas and query suggestions related to the request.",
    "coding": "You are a coding specialist. Produce clear implementation code when asked.",
    "qa": "You are a QA specialist. Suggest tests and ways to verify code quality.",
    "evaluator": (
        "You evaluate the assistant results. Respond with 'APPROVED' if the solution looks good. "
        "Otherwise respond with 'IMPROVE' followed by improvement suggestions."
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
    conversation = [{"role": "user", "content": user_input}]

    arch_task = asyncio.create_task(_ask_agent("architect", conversation))
    dba_task = asyncio.create_task(_ask_agent("dba", conversation))
    code_task = asyncio.create_task(_ask_agent("coding", conversation))
    qa_task = asyncio.create_task(_ask_agent("qa", conversation))

    arch_reply, dba_reply, code_reply, qa_reply = await asyncio.gather(
        arch_task, dba_task, code_task, qa_task
    )

    conversation.extend(
        [
            {"role": "assistant", "content": arch_reply},
            {"role": "assistant", "content": dba_reply},
            {"role": "assistant", "content": code_reply},
            {"role": "assistant", "content": qa_reply},
        ]
    )

    combined = (
        f"Architect:\n{arch_reply}\n\nDBA:\n{dba_reply}\n\n"
        f"Coding Specialist:\n{code_reply}\n\nQA Specialist:\n{qa_reply}"
    )

    evaluation = await _ask_agent(
        "evaluator", [{"role": "assistant", "content": combined}]
    )
    iteration = 0

    while "IMPROVE" in evaluation.upper() and iteration < max_iterations:
        feedback = evaluation + "\nPlease incorporate these improvements."
        code_reply, qa_reply = await asyncio.gather(
            _ask_agent("coding", [{"role": "user", "content": feedback}]),
            _ask_agent("qa", [{"role": "assistant", "content": feedback}]),
        )
        combined += (
            f"\n\nUpdated Coding Specialist:\n{code_reply}\n\nUpdated QA Specialist:\n{qa_reply}"
        )
        evaluation = await _ask_agent(
            "evaluator", [{"role": "assistant", "content": combined}]
        )
        iteration += 1

    return combined, evaluation


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Interactive Coding Agent</title>
  </head>
  <body>
    <h1>Interactive Coding Agent</h1>
    <div>
    {% for entry in log %}
      <p><strong>{{ entry['sender'] }}:</strong> {{ entry['text'] }}</p>
    {% endfor %}
    </div>
    <form method="post">
      <input name="message" autofocus style="width:80%%" />
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
            combined, evaluation = asyncio.run(_orchestrate(msg))
            log.append({"sender": "Agents", "text": combined})
            log.append({"sender": "Evaluator", "text": evaluation})
    return render_template_string(TEMPLATE, log=log)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
