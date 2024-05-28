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

data_save_path = Path('/Users/yangkaiwen/Documents/hypergai-chatbot/dev/data/0528')

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
    persist_directory=os.path.join(data_save_path, "chroma_openai_0528"),
)
host = os.environ['REDIS_HOST']
port = '19817'
password = os.environ['REDIS_PASSWORD']

client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
store = RedisStore(client=client)

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="楼盘ID",
)

# Load documents
all_files = Path('/Users/yangkaiwen/Documents/hypergai-chatbot/dev/data/0528/qa_pairs').glob('*.txt')

docs = []
qa_pairs = []
text_summaries = []
doc_ids = []

for file in all_files:
    doc_id = file.stem
    with open(f'/Users/yangkaiwen/Documents/hypergai-chatbot/dev/data/0528/qa_pairs/{doc_id}.txt', 'r') as f:
        qa_pair = f.read()
    
    with open(f'/Users/yangkaiwen/Documents/hypergai-chatbot/dev/data/0528/summary/{doc_id}.txt', 'r') as f:
        summary = f.read()

    with open(f'/Users/yangkaiwen/Documents/hypergai-chatbot/data/docs_0528/{doc_id}.json', 'r') as f:
        doc = json.load(f)

    qa_pairs.append(qa_pair)
    text_summaries.append(summary)
    doc_ids.append(doc_id)
    docs.append(doc)

# Add texts
summary_texts = [Document(page_content=s, metadata={"summary": s,
                                                    "qa_pairs": qa_pairs[i],
                                                    **docs[i]}) for i, s in enumerate(text_summaries)]
qa_texts = [Document(page_content=s, metadata={"summary": text_summaries[i],
                                               "qa_pairs": s,
                                               **docs[i]}) for i, s in enumerate(qa_pairs)]
text_docs = [json.dumps({"page_content": s, "metadata": {"summary": s,
                                                         "qa_pairs": qa_pairs[i],
                                                         **docs[i]}}) for i, s in enumerate(text_summaries)]
retriever.vectorstore.add_documents(summary_texts)
retriever.vectorstore.add_documents(qa_texts)
retriever.docstore.mset(list(zip(doc_ids, text_docs)))
