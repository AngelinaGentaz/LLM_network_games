import os
import json
from mistralai import Mistral

def call_mistral_api(api_key, system_prompt, user_prompt, player_id, cost):
    """
    Send system + user prompts to a Mistral model and return parsed JSON or raw text.
    """
    # Instantiate the Mistral client
    client = Mistral(api_key=api_key)

    # Prepare messages in the correct format
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]

    response = client.chat.complete(
        model="mistral-medium-2505",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    # Extract the response content
    raw = response.choices[0].message.content.strip()
    
    # If it's fenced…
    if raw.startswith("```"):
        # drop first and last lines
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1])
    
    # If it starts and ends with quotes, remove them and try to parse as JSON
    if raw.startswith('"') and raw.endswith('"'):
        # Remove outer quotes
        inner_content = raw[1:-1]
        # Unescape the content (handle \" and \n)
        try:
            # Use json.loads to properly decode the escaped string
            unescaped_content = json.loads('"' + inner_content + '"')
            # Now try to parse as JSON by wrapping in braces
            json_string = "{" + unescaped_content + "}"
            return json.loads(json_string)
        except json.JSONDecodeError:
            # If that fails, try manual unescaping as fallback
            try:
                manual_unescaped = inner_content.replace('\\"', '"').replace('\\n', '\n')
                json_string = "{" + manual_unescaped + "}"
                return json.loads(json_string)
            except json.JSONDecodeError:
                return {"raw_output": inner_content}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw}