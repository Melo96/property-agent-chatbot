import streamlit as st
import os
import re
import openai
import cohere
import json
import time
import redis
import boto3
import random
from io import BytesIO
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore

from prompt_template.prompts_handbook import *
from prompt_template.response import *
from utils.utils import output_parser, get_routes

from dotenv import load_dotenv
load_dotenv()

rerank = False
router_type = 'llm'

reranker = 'rerank-multilingual-v3.0'
db_name = "adobe_handbook_db"
db_path = Path(__file__).parent / 'data'
persist_directory = db_path / db_name
doc_id_key = "doc_id"
redis_host = os.environ['REDIS_HOST']
redis_port = os.environ['REDIS_PORT']
redis_password = os.environ['REDIS_PASSWORD']
top_k = 5
reranker_top_k = 10

bucket_name = 'hypergai-data'

def chat_llm_stream(user_input, system_prompt='', chat_history=[], temperature=0.2, llm="gpt-4o"):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=chat_history
    messages += [{'role': 'user', 'content': user_input}]

    with st.chat_message("assistant"):
        stream = st.session_state['llm_client'].chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                stream=True
            ) 
        response = st.write_stream(stream)
    return response

def chat_llm(user_input, system_prompt='', chat_history=[], temperature=0.2, display_textbox=False, llm="gpt-4o"):
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

    if display_textbox:
        with st.chat_message("assistant"):
            st.write(message)
        st.session_state['display_messages'].append({"role": "assistant", "content": message})
    return message

@st.cache_resource
def initialize_chain():
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=db_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(persist_directory),
    )
    # The storage layer for the parent documents
    redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    docstore = RedisStore(client=redis_client)

    # Load models
    reranker = cohere.Client(os.environ['COHERE_API_KEY'])
    llm_client = openai.Client()

    # Load S3 client
    s3_client = boto3.client(
        's3', 
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    return [vectorstore, docstore, llm_client, reranker, s3_client]

def multi_queries_retrieval(ori_query):
    # Multi-query generation
    s = time.time()
    multi_query = chat_llm(MULTI_QUERY_PROMPT.format(question=ori_query), temperature=0)
    pattern = re.compile(r'<question>(.*?)</question>')
    queries = [ori_query] + pattern.findall(multi_query)
    e = time.time()
    print(f'Multi queries: {e-s} seconds')
    print(queries)

    # Get the matches for each query
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        vectore_search_partial = partial(st.session_state['vectorstore'].similarity_search_with_relevance_scores, 
                                         k=top_k, 
                                         score_threshold=0.7
                                         )
        nested_results = executor.map(vectore_search_partial, queries)

    matches = [item for sub_list in nested_results for item in sub_list]
    match_list_text = ''
    if matches:
        doc_ids = set(map(lambda d: d[0].metadata[doc_id_key], matches))
        matches = st.session_state['docstore'].mget(list(doc_ids))
        print(f'Found {len(matches)} relevant docs')
        match_list = [json.loads(match) for match in matches]
        match_list_text = [match['page_content'] for match in match_list]
    return match_list_text

def chat(ori_query):
    # Multi-query generation
    s = time.time()
    multi_query = chat_llm(MULTI_QUERY_PROMPT.format(question=ori_query), temperature=0)
    pattern = re.compile(r'<question>(.*?)</question>')
    queries = [ori_query] + pattern.findall(multi_query)
    e = time.time()
    print(f'Multi queries: {e-s} seconds')
    print(queries)

    # Get the matches for each query
    s = time.time()
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        vectore_search_partial = partial(st.session_state['vectorstore'].similarity_search_with_relevance_scores, 
                                         k=top_k, 
                                         score_threshold=0.7
                                         )
        nested_results = executor.map(vectore_search_partial, queries)
    e = time.time()
    print(f'Vector search: {e-s} seconds')

    matches = [item for sub_list in nested_results for item in sub_list]
    match_list_text = ''
    if matches:
        doc_ids = set(map(lambda d: d[0].metadata[doc_id_key], matches))
        matches = st.session_state['docstore'].mget(list(doc_ids))
        print(f'Found {len(matches)} relevant docs')
        match_list = [json.loads(match) for match in matches]
        match_list_text = [match['page_content'] for match in match_list]
    else:
        with st.chat_message("assistant"):
            st.write(NO_RELEVANT_FILES)
            st.session_state['display_messages'].append({"role": "assistant", "content": NO_RELEVANT_FILES})
            return

    result_text = ''.join(f'<context>{t}</context>' for t in match_list_text)
    s = time.time()
    response = chat_llm_stream(CHAT_USER_PROMPT.format(context=result_text, question=ori_query), 
                               CHAT_SYSTEM_PROMPT, 
                               st.session_state['messages']
                               )
    e = time.time()
    print(f'Response: {e-s} seconds')
    return response

# Begin of Streamlit UI Code
st.title("Employee Handbook Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['display_messages'] = []
    st.session_state['messages'] = []
    st.session_state['context'] = ''

sesstion_state_name = ['vectorstore', 'docstore', 'llm_client', 'reranker', 's3_client']
init = initialize_chain()
for name, func in zip(sesstion_state_name, init):
    st.session_state[name] = func

if prompt := st.chat_input('Message'):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    chat(prompt)
