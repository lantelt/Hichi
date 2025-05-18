# Interactive Coding Agent

This repository provides a simple web interface for a multi-agent coding assistant. The system now includes many more roles (market research, system design, code generation, testing and others) which are described in `AGENTS.md`. The project manager agent orchestrates the flow so the conversation mimics a collaborative development team. An evaluation agent may request improvements before final approval, triggering an automatic improvement cycle where code is regenerated and retested. Each visitor has an isolated chat session and conversation state is stored in memory.
See `AGENTS.md` for details about each agent role.

## Orchestrator

The workflow is built with the Google **Agent Development Kit (ADK)**.  Each
stage in `WORKFLOW_ORDER` of `interactive_agent.py` becomes an `LlmAgent` with a
specific system prompt.  These agents run sequentially via `SequentialAgent` and
are wrapped in a `LoopAgent` so the evaluator can trigger automatic improvement
loops.  ADK handles passing state between agents and executing the full
pipeline, allowing new roles to be inserted easily.  Each agent now specifies an
`output_key` so its response is written to the shared state automatically.

## Requirements

- Python 3.8+
- Required Python packages: `openai` and `flask` (installed automatically when using the Docker container)
- An OpenAI API key provided via the `OPENAI_API_KEY` environment variable

### Environment variables

The application uses several variables that can be set via Docker or your shell:

- `OPENAI_API_KEY` – key used by ADK to call OpenAI models.
- `OPENAI_MODEL` – model name passed to the API (defaults to `gpt-3.5-turbo`).
- `FLASK_SECRET` – secret value for Flask sessions.
- `MAX_LOG_LENGTH` – number of chat entries kept in memory.
- `ADK_MCP_URL` – base URL to the Model Context Protocol service.
- `ADK_MCP_TOKEN` – authentication token for MCP requests.

When set, these variables are passed to every `LlmAgent` so the ADK can
communicate with an MCP service for tool execution and state sharing.

## Usage

### Local execution

Start the web server and open your browser to `http://localhost:5000`:

```bash
python interactive_agent.py
```

### Running with Docker

You can also build and run the agent inside a container. This keeps any API keys out of the repository and allows them to be supplied at runtime:

```bash
docker build -t interactive-agent .
docker run --rm -p 5000:5000 -e OPENAI_API_KEY=your-key-here interactive-agent
```

You can optionally set `FLASK_SECRET` to specify the Flask session secret key.
The server will keep up to `MAX_LOG_LENGTH` messages in memory for each session
(default 100). Older messages are written to files under `logs/`.

### Running on Kubernetes

Example manifests are available under `k8s/`. Build and push the Docker image to
a registry accessible by your cluster and then apply the manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The deployment expects secrets named `openai`, `flask` and `adk` containing the
environment variables listed above. The service exposes the web interface on
port 80 inside the cluster.

## Note

This script requires an internet connection and a valid OpenAI API key. Error handling is minimal and provided as a basic example. Feel free to expand upon it for your own use.
