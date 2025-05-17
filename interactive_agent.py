import os
import openai

API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    print('Missing OPENAI_API_KEY environment variable.')
    exit(1)

openai.api_key = API_KEY

SYSTEM_PROMPT = "You are a helpful coding assistant."
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

print("Type 'exit' or 'quit' to end the conversation.")
while True:
    user_input = input('User: ').strip()
    if user_input.lower() in {'exit', 'quit'}:
        break
    conversation.append({"role": "user", "content": user_input})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        reply = response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        continue
    print(f'Assistant: {reply}\n')
    conversation.append({"role": "assistant", "content": reply})
