import json

def chat_oepnai(doc, client, system_prompt):
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": 'system', "content": system_prompt},
                    {'role': 'user', 'content': f'Json File: {json.dumps(doc)}. Response: '}],
        )
    message = response.choices[0].message.content
    return message