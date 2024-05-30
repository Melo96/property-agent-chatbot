import os
import re
import openai
import cohere
import json
import streamlit as st
import time
import redis
import boto3
from PIL import Image
from io import BytesIO
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore

from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

from prompt_template.prompts import *
from prompt_template.response import *
from prompt_template.costar_prompts import REPHRASING_DECISION_PROMPT, REPHRASING_PROMPT
from utils.utils import *

from dotenv import load_dotenv
load_dotenv()

rerank = False
router_type = 'llm'

reranker = 'rerank-multilingual-v3.0'
collection_name = "summaries"
persist_directory = Path(__file__).parent / "data/0528/chroma_openai_0528"
redis_host = os.environ['REDIS_HOST']
redis_port = os.environ['REDIS_PORT']
redis_password = os.environ['REDIS_PASSWORD']
top_k = 10
doc_id_key = "楼盘ID"

bucket_name = 'hypergai-data'

def chat_llm_stream(user_input, system_prompt='', chat_history=[], temperature=0.2, llm="gpt-4o"):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=chat_history
    messages += [{'role': 'user', 'content': user_input}]

    response = st.session_state['llm_client'].chat.completions.create(
            model=llm,
            messages=messages,
            temperature=temperature,
            stream=True
        )

    message_complete = ''
    message = ''
    for chunk in response:
        text = chunk.choices[0].delta.content
        if text:
            message_complete+=text
            if '\n\n' in text:
                with st.chat_message("assistant"):
                    st.write(message)
                st.session_state['display_messages'].append({"role": "assistant", "content": message})
                message = ''
            else:
                message+=text
    # Print last chunk of text
    if message:
        with st.chat_message("assistant"):
            st.write(message)
        print(message_complete)
        st.session_state['display_messages'].append({"role": "assistant", "content": message})
    return message_complete

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
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(persist_directory),
    )
    # The storage layer for the parent documents
    redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    docstore = RedisStore(client=redis_client)

    # Load models
    encoder = OpenAIEncoder()
    reranker = cohere.Client(os.environ['COHERE_API_KEY'])
    llm_client = openai.Client()

    # Load routers
    routers = get_routes()

    # Load S3 client
    s3_client = boto3.client(
        's3', 
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    # Load name2id map
    with open(Path(__file__).parent / "data/0528/name2id.json", 'r') as f:
        name2id = json.load(f)
    return [vectorstore, docstore, llm_client, encoder, reranker, routers, s3_client, name2id]

def coreference_resolution(ori_query):
    if st.session_state['messages']:
        user_input_history = '[' + ', '.join(
                                f"user: {item['content']}\n" if item['role'] == 'user' else f"assistant: {item['content']}\n"
                                for item in st.session_state['messages']
                            ) + ']'

        output = chat_llm(COREFERENCE_RESOLUTION.format(history=user_input_history,
                                                        question=ori_query),
                          temperature=0
                          )
        ori_query = output_parser(output, 'OUTPUT QUESTION: ')
    return ori_query

def multi_queries_retrieval(ori_query):
    # Multi-query generation
    s = time.time()
    multi_query = chat_llm(ori_query, MULTI_QUERY_PROMPT, st.session_state['messages'], temperature=0)
    e = time.time()
    print(f'Multi queries: {e-s} seconds')
    queries = [ori_query] + multi_query.split("\n")[1:-1]

    # Get the matches for each query
    with ThreadPoolExecutor(max_workers=6) as executor:
        vectore_search_partial = partial(st.session_state['vectorstore'].similarity_search_with_relevance_scores, 
                                         k=top_k, 
                                         score_threshold=0.7)
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

def query_rephrase(query):
    decision = chat_llm(REPHRASING_DECISION_PROMPT.format(chat_history=st.session_state['history'], question=query))
    if 'true' in decision:
        query = chat_llm(REPHRASING_PROMPT.format(chat_history=st.session_state['history'], question=query))
    return query

@st.spinner('正在输入...')
def rag(ori_query):
    if st.session_state['messages']:
        st.session_state['history'] = json.dumps(st.session_state['messages'], ensure_ascii=False)
    # Query Rephrasing
    s = time.time()
    ori_query = query_rephrase(ori_query)
    e = time.time()
    print(f'Query rephrasing: {ori_query}, {e-s} seconds')

    # Coreference Resolution
    s = time.time()
    ori_query = coreference_resolution(ori_query)
    e = time.time()
    print(f'Coreference resolution: {ori_query}, {e-s} seconds')

    # RAG Router
    s = time.time()
    router_result = chat_llm(RAG_ROUTER_PROMPT.format(question=ori_query, history=st.session_state['history']), temperature=0)
    e = time.time()
    print(f"RAG Router: {router_result}, {e-s} seconds")

    result_text = ''
    if 'no' in router_result:
        # Multi-query
        s = time.time()
        match_list_text = multi_queries_retrieval(ori_query)
        e = time.time()
        print(f'Vector search: {e-s} seconds')

        if not match_list_text:
            with st.chat_message("assistant"):
                st.write(NO_RELEVANT_FILES)

            col1, col2 = st.columns(2)
            with col1:
                st.text_input(label="姓名")
            with col2:
                st.text_input(label="手机号")
            return NO_RELEVANT_FILES

        # Reranking
        if rerank:
            s = time.time()
            results = st.session_state['reranker'].rerank(model=reranker, query=ori_query, documents=match_list_text, top_n=20, return_documents=True)
            result_text_list = [item.document.text for item in results.results]
            result_text = ''.join(f'<context>{t}</context>' for t in result_text_list)
            e = time.time()
            print(f'Reranking: {e-s} seconds')
        else:
            result_text = ''.join(f'<context>{t}</context>' for t in match_list_text)

    # Response
    response = chat_llm_stream(RAG_USER_PROMPT.format(ori_query, result_text), RAG_SYSTEM_PROMPT, st.session_state['messages'])
    return response

def chat(ori_query):
    st.session_state['display_messages'].append({"role": "user", "content": ori_query})
    # Limit the number of chat history
    st.session_state['messages'] = st.session_state['messages'][:20]
    # Query Router
    s = time.time()
    router_result = chat_llm(QUERY_ROUTER_PROMPT.format(question=ori_query), chat_history=st.session_state['messages'], temperature=0)
    e = time.time()
    print(f"Query Router: {router_result}, {e-s} seconds")

    # Call RAG or directly call LLM
    s1 = time.time()
    if 'query' in router_result:
        response = rag(ori_query)
        st.session_state['messages'].append({"role": "user", "content": ori_query})
        st.session_state['messages'].append({"role": "assistant", "content": response})
    else:
        response = chat_llm_stream(ori_query, CHAT_SYSTEM_PROMPT, chat_history=st.session_state['messages'])
    
    # Get house images
    router_result = chat_llm(IMAGE_ROUTER_PROMPT, chat_history=st.session_state['messages'], temperature=0)
    if '<house>' in router_result:
        pattern = re.compile(r'<house>(.*?)</house>')
        houses_list = pattern.findall(router_result)
        for house in houses_list:
            with st.chat_message("assistant"):
                st.write(HOUSE_IMAGE_RESPONSE.format(house_name=house))
            if house in st.session_state['name2id']:
                house_id = st.session_state['name2id'][house]
                img_list = st.session_state['s3_client'].list_objects_v2(Bucket=bucket_name, Prefix=f'img_house/{house_id}')
                for img in img_list['Contents']:
                    img_response = st.session_state['s3_client'].get_object(Bucket=bucket_name, Key=img['Key'])
                    image_data = img_response['Body'].read()
                    with st.chat_message("assistant"):
                        st.image(BytesIO(image_data))
    
    e1 = time.time()
    print(f'Response: {e1-s1} seconds')
    return response

st.title("爱房网智能客服DEMO")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['display_messages'] = [{"role": "assistant", "content": '您好！感谢您选择爱房网。我是您的专属房产咨询助手小盖。请问有什么可以帮助您的吗？'}]
    st.session_state['messages'] = []
    st.session_state['history'] = ''

# Display chat messages from history on app rerun
for message in st.session_state['display_messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

sesstion_state_name = ['vectorstore', 'docstore', 'llm_client', 'encoder', 'reranker', 'routers', 's3_client', 'name2id']
init = initialize_chain()
for name, func in zip(sesstion_state_name, init):
    st.session_state[name] = func

if prompt := st.chat_input('请在这里输入消息，点击Enter发送'):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    response = chat(prompt)
