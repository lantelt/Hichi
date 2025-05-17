# Interactive Coding Agent

This repository now contains an example interactive coding agent built in Python. The agent uses the OpenAI API to provide conversational coding assistance directly from the command line.

## Requirements

- Python 3.8+
- `openai` Python package (`pip install openai`) *(installed automatically when using the Docker container)*
- An OpenAI API key provided via the `OPENAI_API_KEY` environment variable

## Usage

### Local execution

Run the agent directly in your terminal:

```bash
python interactive_agent.py
```

Type your questions or coding requests and the assistant will respond. Enter `exit` or `quit` to end the session.

### Running with Docker

You can also build and run the agent inside a container. This keeps any API keys out of the repository and allows them to be supplied at runtime:

```bash
docker build -t interactive-agent .
docker run --rm -e OPENAI_API_KEY=your-key-here interactive-agent
```

## Note

This script requires an internet connection and a valid OpenAI API key. Error handling is minimal and provided as a basic example. Feel free to expand upon it for your own use.
