# Interactive Coding Agent

This repository provides a simple web interface for a multi-agent coding assistant. The system now includes many more roles (market research, system design, code generation, testing and others) which are described in `AGENTS.md`. The project manager agent orchestrates the flow so the conversation mimics a collaborative development team. An evaluation agent may request improvements before final approval, triggering an automatic improvement cycle where code is regenerated and retested. Each visitor has an isolated chat session and conversation state is stored in memory.
See `AGENTS.md` for details about each agent role.

## Requirements

- Python 3.8+
- Required Python packages: `openai` and `flask` (installed automatically when using the Docker container)
- An OpenAI API key provided via the `OPENAI_API_KEY` environment variable

## Usage

### Local execution

Start the web server and open your browser to `http://localhost:5000`:

```bash
python interactive_agent.py
```
The page now includes a field allowing you to select how many improvement cycles
should run. Leave it blank to use the `MAX_ITERATIONS` environment default.

### Running with Docker

You can also build and run the agent inside a container. This keeps any API keys out of the repository and allows them to be supplied at runtime:

```bash
docker build -t interactive-agent .
docker run --rm -p 5000:5000 -e OPENAI_API_KEY=your-key-here interactive-agent
```

You can optionally set `FLASK_SECRET` to specify the Flask session secret key.
`MAX_ITERATIONS` can also be set to change how many improvement cycles run by default.

## Note

This script requires an internet connection and a valid OpenAI API key. Error handling is minimal and provided as a basic example. Feel free to expand upon it for your own use.
