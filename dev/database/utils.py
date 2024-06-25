import json

def chat_oepnai(doc, client, system_prompt, mode='json'):
    if mode=='json':
        response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": 'system', "content": system_prompt},
                        {'role': 'user', 'content': f'Json File: {json.dumps(doc, ensure_ascii=False)}. Response: '}],
                temperature=0
            )
    elif mode=='text':
        response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": 'system', "content": system_prompt},
                        {'role': 'user', 'content': f'text chunk: {doc}. Response: '}],
                temperature=0
            )
    message = response.choices[0].message.content
    return message