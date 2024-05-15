__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import openai
import json
import streamlit as st
import time
import redis
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore
from prompt_template.prompts import *

from dotenv import load_dotenv
load_dotenv()

llm = "gpt-4o"
collection_name = "summaries"
persist_directory = "data/chroma_openai"
redis_host = 'redis-10020.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com'
redis_port = '10020'
redis_password = 'yfT9uQWDa3BFAA871OmLhhUbLv3oETWh'
top_k = 10
doc_id_key = "doc_id"

def chat_oepnai(user_input, client, system_prompt):
    response = client.chat.completions.create(
            model=llm,
            messages=[{"role": 'system', "content": system_prompt},
                    {'role': 'user', 'content': user_input}],
        )
    message = response.choices[0].message.content
    return message

# Load embedding model
@st.cache_resource
def initialize_chain():
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )

    # The storage layer for the parent documents
    redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    docstore = RedisStore(client=redis_client)

    llm_client = openai.Client()
    return [vectorstore, docstore, llm_client]

@st.cache_resource
def rag(ori_query):
    st.text("正在查找答案...")
    # Multi-query generation
    multi_query = chat_oepnai(ori_query, st.session_state['llm_client'], MULTI_QUERY_PROMPT)
    queries = [ori_query] + multi_query.split("\n")[1:-1]
    st.markdown(f"""<h5>为您生成相关问题:</h5>
        {str("<br />".join(queries))}
        <br />
        <br />
    """,
    unsafe_allow_html=True
    )

    # Get the matches for each query
    doc_ids = set()
    for query in queries:
        matches = st.session_state['vectorstore'].max_marginal_relevance_search(query, k=top_k, lambda_mult=0.5)
        doc_ids.update([match.metadata['doc_id'] for match in matches])
    doc_ids = list(doc_ids)
    matches = st.session_state['docstore'].mget(doc_ids)
    match_list = [json.loads(match) for match in matches]
    match_list_text = [match['page_content'] for match in match_list]
    result_text = ''.join([f'<context>{t}</context>' for t in match_list_text])

    st.text(f"已为您找到{len(match_list)}个相关来源")
    # Reranking and Reordering
    response = chat_oepnai(RAG_USER_PROMPT.format(ori_query, result_text), st.session_state['llm_client'], RAG_SYSTEM_PROMPT)
    st.markdown(f"""<h3>Agent answers:</h3>
        {str(response)}
        <br />
        <br />
    """,
    unsafe_allow_html=True
    )
    return response

# Initialization
if 'files' not in st.session_state:
    st.session_state['files'] = set()

st.title("爱房网智能客服DEMO")
retrieval_tab, chat_tab = st.tabs(
        ["Retrieval 知识查找", "Chat about File 文件细节"]
    )

with retrieval_tab:
    st.subheader("知识查找")
    sesstion_state_name = ['vectorstore', 'docstore', 'llm_client']
    init = initialize_chain()
    for name, func in zip(sesstion_state_name, init):
        st.session_state[name] = func
    ori_query = st.text_input("问题")
    if st.button('搜索') and ori_query:
        st.session_state['ori_query'] = ori_query
        s = time.time()
        response = rag(ori_query)

        e = time.time()
        print("total time:", e-s)

with chat_tab:
    st.subheader("文件细节")

