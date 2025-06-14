import os
import json
from mistralai import Mistral

def call_mistral_api(api_key, system_prompt, user_prompt, player_id, cost):
    """
    Send system + user prompts to a Mistral model and return parsed JSON or raw text.
    """
    # 1) Instantiate the Mistral client
    client = Mistral(api_key=api_key)  # pip install mistralai :contentReference[oaicite:2]{index=2}

    # 2) Issue a chat completion
    response = client.chat.complete(
        model="mistral-small-latest",    # switch to "mistral-large-latest" or others as needed
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt}
        ],
        temperature=0.7,
    )

    # 3) Extract and parse
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
