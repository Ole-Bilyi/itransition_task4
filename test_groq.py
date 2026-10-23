import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

tools = [{
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo a value back",
        "parameters": {
            "type": "object",
            "properties": {"v": {"type": "string"}},
            "required": ["v"],
        },
    },
}]

r = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Echo hello"}],
    tools=tools,
    tool_choice="auto",
)

print(r.choices[0].message.tool_calls)