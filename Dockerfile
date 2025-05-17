FROM python:3.10-slim

WORKDIR /app

COPY interactive_agent.py /app/

RUN pip install --no-cache-dir openai flask

EXPOSE 5000

CMD ["python", "interactive_agent.py"]
