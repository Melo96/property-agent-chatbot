# # For streamlit cloud deployment
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import openai
import cohere
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

rerank = False

co = cohere.Client('LaLtYhX3dsRsCxURaEPcidgL9Se9gNfG0h9lLorf')
llm = "gpt-4o"
reranker = 'rerank-multilingual-v3.0'
collection_name = "summaries"
persist_directory = "data/chroma_openai"
redis_host = 'redis-10020.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com'
redis_port = '10020'
redis_password = 'yfT9uQWDa3BFAA871OmLhhUbLv3oETWh'
top_k = 10
doc_id_key = "doc_id"

def chat_oepnai(user_input, client, system_prompt='', chat_history=[], temperature=0.2):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=chat_history
    messages += [{'role': 'user', 'content': user_input}]

    response = client.chat.completions.create(
            model=llm,
            messages=messages,
            temperature=temperature
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
    # Coreference Resolution
    print(ori_query)
    user_input_history = '['
    for i, item in enumerate(st.session_state['messages']):
        if i!=0:
            user_input_history+=', '
        if item['role']=='user':
            user_input_history+=f'Q: {item['content']}'
        elif item['role']=='assistant':
            user_input_history+=f'A: {item['content']}'
    print(user_input_history)
    if st.session_state['messages']:
        output = chat_oepnai(COREFERENCE_RESOLUTION.format(history=user_input_history,
                                                              question=ori_query),
                                st.session_state['llm_client'], 
                                temperature=0)
        if 'OUTPUT QUESTION: ' in output:
            ori_query = output.split('OUTPUT QUESTION: ')[1]
    print(ori_query)

    # Multi-query generation
    s = time.time()
    multi_query = chat_oepnai(ori_query, st.session_state['llm_client'], MULTI_QUERY_PROMPT, temperature=0)
    e = time.time()
    print(f'Multi queries: {e-s} seconds')
    queries = [ori_query] + multi_query.split("\n")[1:-1]

    s = time.time()
    # Get the matches for each query
    doc_ids = set()
    for query in queries:
        matches = st.session_state['vectorstore'].max_marginal_relevance_search(query, k=top_k, lambda_mult=0.5)
        # matches = st.session_state['vectorstore'].similarity_search_with_relevance_scores(query, k=top_k)
        doc_ids.update([match.metadata['doc_id'] for match in matches])
    doc_ids = list(doc_ids)

    matches = st.session_state['docstore'].mget(doc_ids)
    match_list = [json.loads(match) for match in matches]
    match_list_text = [match['page_content'] for match in match_list]
    st.text(f"已为您找到{len(match_list)}个相关来源")
    e = time.time()
    print(f'Vector search: {e-s} seconds')

    # Reranking
    if rerank:
        s = time.time()
        results = co.rerank(model=reranker, query=ori_query, documents=match_list_text, top_n=20, return_documents=True)
        result_text_list = [item.document.text for item in results.results]
        result_text = ''.join([f'<context>{t}</context>' for t in result_text_list])
        e = time.time()
        print(f'Reranking: {e-s} seconds')
    else:
        result_text = ''.join([f'<context>{t}</context>' for t in match_list_text])

    # Response
    s = time.time()
    response = chat_oepnai(RAG_USER_PROMPT.format(ori_query, result_text), st.session_state['llm_client'], RAG_SYSTEM_PROMPT, st.session_state['messages'])
    st.markdown(f"""
        {str(response)}
        <br />
        <br />
    """,
    unsafe_allow_html=True
    )
    e = time.time()
    print(f'Response: {e-s} seconds')
    return response

st.title("爱房网智能客服DEMO")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

sesstion_state_name = ['vectorstore', 'docstore', 'llm_client']
init = initialize_chain()
for name, func in zip(sesstion_state_name, init):
    st.session_state[name] = func

if prompt := st.chat_input("您好！请问有什么可以帮助您的吗？"):
    # Add user message to chat history
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = rag(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({"role": "assistant", "content": response})
