import json
import pandas as pd
import openai
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from prompt_template.prompts import SUMMARY_PROMPT, QA_PAIR_PROMPT

from dotenv import load_dotenv
load_dotenv()

data_save_path = Path('/Users/yangkaiwen/Documents/hypergai-chatbot-data/mock')
csv_path = '/Users/yangkaiwen/Documents/hypergai-chatbot-data/mock/real_estate_listings.csv'

df = pd.read_csv(csv_path)
docs = df.to_dict(orient='records')

# Save docs as json files
for i, doc in enumerate(docs):
    doc['property_id'] = i
    with open(data_save_path / f'docs/{i}.json', 'w') as f:
        json.dump(doc, f)

print(f'Number of Docs: {len(docs)}')

client = openai.Client()

def chat_oepnai(doc, client, system_prompt):
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": 'system', "content": system_prompt},
                    {'role': 'user', 'content': f'Json File: {json.dumps(doc, ensure_ascii=False)}. Response: '}],
            temperature=0
        )
    message = response.choices[0].message.content
    return message

text_summaries = []
qa_pairs = []
print("Generating summaries")
with ThreadPoolExecutor(max_workers=8) as executor:
    gpt_partial = partial(chat_oepnai, client=client, system_prompt=SUMMARY_PROMPT)
    text_summaries = list(executor.map(gpt_partial, docs))

# Save summaries
for i, summary in enumerate(text_summaries):
    with open(data_save_path / f'summary/{i}.txt', 'w') as f:
        f.write(summary)

print("Generating QA pairs")
with ThreadPoolExecutor(max_workers=8) as executor:
    gpt_partial = partial(chat_oepnai, client=client, system_prompt=QA_PAIR_PROMPT)
    qa_pairs = list(executor.map(gpt_partial, docs))

# Save Q&A pairs
for i, qa_pair in enumerate(qa_pairs):
    with open(data_save_path / f'qa_pairs/{i}.txt', 'w') as f:
        f.write(qa_pair)

print("Generation finished")