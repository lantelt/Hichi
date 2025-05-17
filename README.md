# Interactive Coding Agent

This repository now contains an example multi‑agent coding assistant that can be accessed through a simple web interface. Several specialized agents – an Architect, DBA, Coding Specialist and QA Specialist – collaborate on each request. Their work is reviewed by an evaluation agent which can trigger a short improvement cycle. The agents now build on one another's replies so the flow resembles a real development team. Each visitor has an isolated chat session.

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

### Running with Docker

You can also build and run the agent inside a container. This keeps any API keys out of the repository and allows them to be supplied at runtime:

```bash
docker build -t interactive-agent .
docker run --rm -p 5000:5000 -e OPENAI_API_KEY=your-key-here interactive-agent
```

You can optionally set `FLASK_SECRET` to specify the Flask session secret key.

## Note

This script requires an internet connection and a valid OpenAI API key. Error handling is minimal and provided as a basic example. Feel free to expand upon it for your own use.
