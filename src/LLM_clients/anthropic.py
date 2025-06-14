import anthropic
import json

def call_anthropic_api(api_key, system_prompt, user_prompt, player_id, cost):
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1500,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    response_text = response.content[0].text
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_output": response_text}
