FROM python:3.10-slim

WORKDIR /app

COPY interactive_agent.py /app/

RUN pip install --no-cache-dir openai

CMD ["python", "interactive_agent.py"]
