import streamlit as st
import os
import re
import openai
import cohere
import json
import time
import redis
import boto3
from PIL import Image
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from unstructured.staging.base import elements_from_base64_gzipped_json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore

from prompt_template.prompts_handbook import *
from prompt_template.response import *
from utils.utils import draw_bounding_box, merge_elements_metadata, output_parser, read_svg

from dotenv import load_dotenv
load_dotenv()

rerank = True
router_type = 'llm'
sesstion_state_name = ['vectorstore', 'docstore', 'llm_client', 'reranker', 's3_client']

options = ('adobe',)
reranker = 'rerank-english-v3.0'
db_path = Path(__file__).parent / 'data/handbook_db'
doc_id_key = "doc_id"
redis_host = os.environ['REDIS_HOST']
redis_port = os.environ['REDIS_PORT']
redis_password = os.environ['REDIS_PASSWORD']

bucket_name = 'hypergai-data'

assistant_icon = Image.open(Path(__file__).parent / 'data/logos/LogoIcon.png')

def chat_llm_stream(user_input, system_prompt='', chat_history=[], temperature=0.2, llm="gpt-4o"):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=[msg for msg in chat_history if msg['role']!='image']
    messages += [{'role': 'user', 'content': user_input}]

    with st.chat_message("assistant", avatar=assistant_icon):
        stream = st.session_state['llm_client'].chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                stream=True
            ) 
        response = st.write_stream(stream)
    return response

def chat_llm(user_input, system_prompt='', chat_history=[], temperature=0.2, display_textbox=False, llm="gpt-4o", response_format=None):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=chat_history
    messages += [{'role': 'user', 'content': user_input}]

    response = st.session_state['llm_client'].chat.completions.create(
            model=llm,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
    message = response.choices[0].message.content

    if display_textbox:
        with st.chat_message("assistant", avatar=assistant_icon):
            st.write(message)
        st.session_state['messages'].append({"role": "assistant", "content": message})
    return message

def initialize_chain():
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=st.session_state['db_name'],
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(db_path),
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
    return {'vectorstore': vectorstore, 
            'docstore': docstore, 
            'llm_client': llm_client, 
            'reranker': reranker,
            's3_client': s3_client}

def multiquery_retrieval(ori_query):
    # Top-K router
    s = time.time()
    top_k_json = chat_llm(TOPK_ROUTER_PROMPT.format(question=ori_query), temperature=0, response_format={"type": "json_object"})
    try:
        top_k_json = json.loads(top_k_json)
        top_k = top_k_json['top_k']
        reranker_top_k = top_k // 2
    except:
        top_k = 10
        reranker_top_k = 5
    e = time.time()
    print(f"Get top-k param: {e-s} seconds, {top_k_json}")
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
    match_list = []
    if matches:
        doc_ids = set(map(lambda d: d[0].metadata[doc_id_key], matches))
        matches = st.session_state['docstore'].mget(list(doc_ids))
        print(f'Found {len(matches)} relevant docs')
        match_list = [json.loads(match) for match in matches]
        # Reranking
        if rerank:
            s = time.time()
            rerank_results = st.session_state['reranker'].rerank(model=reranker, query=ori_query, documents=match_list, rank_fields=['page_content'], top_n=reranker_top_k, return_documents=False)
            rerank_results_index = [result.index for result in rerank_results.results]
            match_list = [match_list[i] for i in rerank_results_index]
            e = time.time()

            print(f'Rerank {e-s} seconds')
        match_list_text = [match['page_content'] for match in match_list]
        result_text = '\n\n'.join(f'{i+1}. {t}' for i, t in enumerate(match_list_text))
    s = time.time()
    if match_list:
        response = chat_llm_stream(CHAT_USER_PROMPT.format(context=result_text, question=ori_query), 
                                CHAT_SYSTEM_PROMPT, 
                                chat_history=st.session_state['messages'],
                                )
    else:
        response = chat_llm_stream(CHAT_USER_PROMPT_2.format(question=ori_query), 
                                chat_history=st.session_state['messages'],
                                llm='gpt-3.5-turbo'
                                )
    e = time.time()
    print(f"Response: {e-s} seconds")
    return response, match_list

def chat(ori_query):
    match_list = []
    # Intent router
    s = time.time()
    router_result = chat_llm(INTENT_ROUTER_PROMPT.format(context=st.session_state['context'], question=ori_query),
                             temperature=0)
    e = time.time()
    print(f'Tool router: {router_result}, {e-s} seconds')
    router_result = output_parser(router_result, 'Decision:')
    if 'handbook_query' in router_result:
        response, match_list = multiquery_retrieval(ori_query)
    else:
        response = chat_llm_stream(ori_query, 
                                   CHAT_SYSTEM_PROMPT, 
                                   chat_history=st.session_state['messages'], 
                                   llm='gpt-3.5-turbo'
                                   )
    e = time.time()
    print(f'Response: {e-s} seconds')
    st.session_state['display_messages'].append({"role": "assistant", "content": response})
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response, match_list

# Begin of Streamlit UI Code
st.title("Employee Handbook Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['display_messages'] = []
    st.session_state['messages'] = []
    st.session_state['context'] = ''
    st.session_state['db_name'] = options[0]
    init = initialize_chain()
    for name in sesstion_state_name:
        st.session_state[name] = init[name]

with st.sidebar:
    svg_html = read_svg(Path(__file__).parent / 'data/logos/Logo.svg')
    st.write(svg_html, unsafe_allow_html=True)
    st.write('\n')
    option = st.selectbox('Choose which handbook you are querying about', options)
    st.write(f'Current Handbook: {option}')
    st.text("Reference related to the latest\nresponse will be displayed here")

    if option!=st.session_state['db_name']:
        st.session_state['db_name'] = option
        init = initialize_chain()
        for name in sesstion_state_name:
            st.session_state[name] = init[name]

# Display chat messages from history on app rerun
for message in st.session_state['display_messages']:
    if message["role"]=='image':
        with st.chat_message('assistant'):
            st.image(message["content"])
    elif message["role"]=='assistant':
        with st.chat_message('assistant', avatar=assistant_icon):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input('Message'):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})
        st.session_state['display_messages'].append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    response, match_list = chat(prompt)

    # Display reference
    for i, doc in enumerate(match_list):
        base64_elements_str = doc['metadata']['orig_elements']
        elements = elements_from_base64_gzipped_json(base64_elements_str)
        page, bbox = merge_elements_metadata(elements)
        size = (int(elements[0].metadata.coordinates.system.width), int(elements[0].metadata.coordinates.system.height))
        image = convert_from_path(db_path / f'{st.session_state['db_name']}_handbook.pdf', first_page=page, last_page=page)[0]
        size = (int(elements[0].metadata.coordinates.system.width), int(elements[0].metadata.coordinates.system.height))

        img_with_bbox = draw_bounding_box(image, list(bbox), size)
        with st.sidebar.expander(f"reference {i+1}"):
            st.write(f"**Page {page}**: {doc['metadata']['summary']}")
            st.image(img_with_bbox)
