import json
import openai
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from prompt_template.prompts import SUMMARY_PROMPT, QA_PAIR_PROMPT
from database.utils import chat_oepnai

from dotenv import load_dotenv
load_dotenv()

starting_page_number = 3
pdf_path = "/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo/adobe_handbook.pdf"
data_save_path = Path('/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo')

text_splitter = SemanticChunker(OpenAIEmbeddings())
loader = UnstructuredPDFLoader(pdf_path, mode="elements", chunking_strategy='by_title')
# Returns a List[Element] present in the pages of the parsed pdf document
docs = loader.load_and_split(text_splitter)
docs = [doc.dict() for doc in docs if doc.metadata['page_number']>=starting_page_number]

print(f'Number of Docs: {len(docs)}')

# Save documents
for i, doc in enumerate(docs):
    with open(data_save_path / f'docs/{i}.json', 'w') as f:
        json.dump(doc, f)

client = openai.Client()

docs_text = [doc['page_content'] for doc in docs]

text_summaries = []
qa_pairs = []
print("Generating summaries")
with ThreadPoolExecutor(max_workers=8) as executor:
    gpt_partial = partial(chat_oepnai, client=client, system_prompt=SUMMARY_PROMPT, mode='text')
    text_summaries = list(executor.map(gpt_partial, docs_text))

# Save summaries
for i, summary in enumerate(text_summaries):
    with open(data_save_path / f'summary/{i}.txt', 'w') as f:
        f.write(summary)

print("Generating QA pairs")
with ThreadPoolExecutor(max_workers=8) as executor:
    gpt_partial = partial(chat_oepnai, client=client, system_prompt=QA_PAIR_PROMPT, mode='text')
    qa_pairs = list(executor.map(gpt_partial, docs_text))

# Save Q&A pairs
for i, qa_pair in enumerate(qa_pairs):
    with open(data_save_path / f'qa_pairs/{i}.txt', 'w') as f:
        f.write(qa_pair)

print("Generation finished")