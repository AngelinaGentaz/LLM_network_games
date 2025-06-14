from openai import OpenAI
import json

def call_openai_api(api_key, system_prompt, user_prompt, player_id, cost):
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4o",
        instructions=system_prompt,                 # replaces the 'system' role
        input=[                                     # replaces the 'messages' list
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt}
                ],
            }
        ],
        text={"format": {"type": "json_object"}}, 
        temperature=0.7,
        max_output_tokens=1024,
    )

    raw = response.output_text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw}



