import json
import redis
import openai
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from prompt_template.prompts import SUMMARY_PROMPT, QA_PAIR_PROMPT

from dotenv import load_dotenv
load_dotenv()

docs_path = Path("data/docs")
docs = []

for p in docs_path.rglob("*.json"):
    with open(p, 'r') as f:
        doc = json.load(f)
    docs.append(doc)

client = openai.Client()
client.chat.completions.create(temperature=0)

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
print("Generating QA pairs")
with ThreadPoolExecutor(max_workers=8) as executor:
    gpt_partial = partial(chat_oepnai, client=client, system_prompt=QA_PAIR_PROMPT)
    qa_pairs = list(executor.map(gpt_partial, docs))
print("Generation finished")

# print("Generating QA pairs")
# qa_pairs = qa_chain.batch(docs, {"max_concurrency": 8})

# # Load embedding model
# model_name = "BAAI/bge-m3"
# model_kwargs = {'device': "cuda:0"}
# embed_model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
# )

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="data/chroma_openai",
)
host = 'redis-10020.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com'
port = '10020'
password = 'yfT9uQWDa3BFAA871OmLhhUbLv3oETWh'

client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
store = RedisStore(client=client)
id_key = "doc_id"
ori_text = "ori_text"
summary = "summary"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id",
)

# Add texts
summary_texts = [Document(page_content=s, metadata={"doc_id": i, 
                                                    "ori_text": json.dumps(docs[i]),
                                                    "summary": s,
                                                    "qa_pairs": qa_pairs[i]}) for i, s in enumerate(text_summaries)]
qa_texts = [Document(page_content=s, metadata={"doc_id": i, 
                                                "ori_text": json.dumps(docs[i]),
                                                "summary": text_summaries[i],
                                                "qa_pairs": s}) for i, s in enumerate(qa_pairs)]
text_docs = [json.dumps({"page_content": s, "metadata": {"doc_id": i, 
                                                        "ori_text": docs[i],
                                                        "summary": s,
                                                        "qa_pairs": qa_pairs[i]}}) for i, s in enumerate(text_summaries)]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(range(len(docs)), text_docs)))
