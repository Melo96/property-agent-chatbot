import os
import json
import redis
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

from dotenv import load_dotenv
load_dotenv()

db_name = 'hypergai'
doc_src_path = Path(f'/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo/{db_name}')
data_save_path = Path('/Users/yangkaiwen/Documents/hypergai-chatbot/dev/data/handbook_db')
id_key = 'doc_id'

# # Load embedding model
# model_name = "BAAI/bge-m3"
# model_kwargs = {'device': "cuda:0"}
# embed_model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
# )

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name=db_name,
    embedding_function=OpenAIEmbeddings(),
    persist_directory=str(data_save_path),
)
# index_name='mock-real-estate'
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(), namespace='mock-real-estate')

host = os.environ['REDIS_HOST']
port = os.environ['REDIS_PORT']
password = os.environ['REDIS_PASSWORD']

client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
store = RedisStore(client=client)

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Load documents
all_files = os.listdir(doc_src_path / 'qa_pairs')

docs = []
qa_pairs = []
text_summaries = []
doc_ids = []

for file in all_files:
    with open(doc_src_path / f'qa_pairs/{file}', 'r') as f:
        qa_pair = f.read()
    
    with open(doc_src_path / f'summary/{file}', 'r') as f:
        summary = f.read()

    with open(doc_src_path / f'docs/{Path(file).stem}.json', 'r') as f:
        doc = json.load(f)

    qa_pairs.append(qa_pair)
    text_summaries.append(summary)
    doc_ids.append(f'{db_name}_{Path(file).stem}')
    docs.append(doc)

# Add texts
summary_texts = [Document(page_content=s, metadata={"summary": s,
                                                    "qa_pairs": qa_pairs[i],
                                                    "ori_text": docs[i]['page_content'],
                                                    "doc_id": doc_ids[i],
                                                    "orig_elements": docs[i]['metadata']['orig_elements'],
                                                    "page_number": docs[i]['metadata']['page_number']}) for i, s in enumerate(text_summaries)]
qa_texts = [Document(page_content=s, metadata={"summary": text_summaries[i],
                                               "qa_pairs": s,
                                               "ori_text": docs[i]['page_content'],
                                               "doc_id": doc_ids[i],
                                               "orig_elements": docs[i]['metadata']['orig_elements'],
                                               "page_number": docs[i]['metadata']['page_number']}) for i, s in enumerate(qa_pairs)]
text_docs = [json.dumps({"page_content": docs[i]['page_content'], 
                         "metadata": {"summary": s,
                                     "qa_pairs": qa_pairs[i],
                                     "ori_text": docs[i]['page_content'],
                                     "doc_id": doc_ids[i],
                                     "orig_elements": docs[i]['metadata']['orig_elements'],
                                     "page_number": docs[i]['metadata']['page_number']}}) for i, s in enumerate(text_summaries)]
print(f'Total number of summary text: {len(summary_texts)}')
print(f'Total number of qa-pairs text: {len(qa_texts)}')
retriever.vectorstore.add_documents(summary_texts)
retriever.vectorstore.add_documents(qa_texts)
retriever.docstore.mset(list(zip(doc_ids, text_docs)))
