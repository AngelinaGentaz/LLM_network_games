# llm_clients/google_client.py

import json
from openai import OpenAI

def call_gemini_api(api_key, system_prompt, user_prompt, player_id, cost):
    """
    Send system + user prompts to Gemini 2.5 Flash via the OpenAI-compatible endpoint,
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )

    text = response.choices[0].message.content

    # 1. If text starts with ``` (and maybe "json"), remove the first and last lines
    if text.lstrip().startswith("```"):
        lines = text.splitlines()
        # drop the fence lines
        lines = lines[1:-1]
        text = "\n".join(lines)

    # 2. Trim any extra whitespace
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text}
