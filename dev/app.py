# # For streamlit cloud deployment
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import openai
import cohere
import json
import streamlit as st
import time
import redis
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore
from prompt_template.prompts import *

from dotenv import load_dotenv
load_dotenv()

rerank = False
router_type = 'llm'

co = cohere.Client(os.environ['COHERE_API_KEY'])
llm = "gpt-4o"
reranker = 'rerank-multilingual-v3.0'
collection_name = "summaries"
persist_directory = "data/chroma_openai"
redis_host = os.environ['REDIS_HOST']
redis_port = '10020'
redis_password = os.environ['REDIS_PASSWORD']
top_k = 10
doc_id_key = "doc_id"

def chat_openai(user_input, system_prompt='', chat_history=[], temperature=0.2):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=chat_history
    messages += [{'role': 'user', 'content': user_input}]

    response = st.session_state['llm_client'].chat.completions.create(
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

def coreference_resolution(ori_query):
    user_input_history = '['
    print(f'Coreference resolution: {ori_query}')
    if st.session_state['messages']:
        for i, item in enumerate(st.session_state['messages']):
            if i!=0:
                user_input_history+=', '
            if item['role']=='user':
                user_input_history+=f'Q: {item['content']}'
            elif item['role']=='assistant':
                user_input_history+=f'A: {item['content']}'

        output = chat_openai(COREFERENCE_RESOLUTION.format(history=user_input_history,
                                                              question=ori_query),
                             temperature=0)
        if 'OUTPUT QUESTION: ' in output:
            ori_query = output.split('OUTPUT QUESTION: ')[1]
    return ori_query

def multi_queries_retrieval(ori_query):
    # Multi-query generation
    s = time.time()
    multi_query = chat_openai(ori_query, MULTI_QUERY_PROMPT, temperature=0)
    e = time.time()
    print(f'Multi queries: {e-s} seconds')
    queries = [ori_query] + multi_query.split("\n")[1:-1]

    # Get the matches for each query
    doc_ids = set()
    for query in queries:
        matches = st.session_state['vectorstore'].max_marginal_relevance_search(query, k=top_k, lambda_mult=0.5)
        # matches = st.session_state['vectorstore'].similarity_search_with_relevance_scores(query, k=top_k, score_threshold=0.7)
        doc_ids.update([match.metadata['doc_id'] for match in matches])
    doc_ids = list(doc_ids)

    matches = st.session_state['docstore'].mget(doc_ids)
    match_list = [json.loads(match) for match in matches]
    match_list_text = [match['page_content'] for match in match_list]
    return match_list_text

def rag(ori_query):
    # Coreference Resolution
    s = time.time()
    ori_query = coreference_resolution(ori_query)
    e = time.time()
    print(f'Coreference resolution: {ori_query}, {e-s} seconds')

    # Multi-query
    s = time.time()
    match_list_text = multi_queries_retrieval(ori_query)
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
    response = chat_openai(RAG_USER_PROMPT.format(ori_query, result_text), RAG_SYSTEM_PROMPT, st.session_state['messages'])
    return response

@st.cache_resource
def chat(ori_query):
    # Router
    s = time.time()
    if router_type=='embedding':
        pass
    else:
        router_result = chat_openai(ROUTER_PROMPT.format(question=ori_query), temperature=0)
        if 'Output: ' in router_result:
            router_result = router_result.split('Output: ')[1]
    e = time.time()
    print(f"Router: {router_result}, {e-s} seconds")

    s = time.time()
    if router_result=='query':
        response = rag(ori_query)
    else:
        response = chat_openai(ori_query, CHAT_SYSTEM_PROMPT, chat_history=st.session_state['messages'])
    
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
        response = chat(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({"role": "assistant", "content": response})
