# For streamlit cloud deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import re
import openai
import cohere
import json
import redis
import boto3
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
from utils.utils import draw_bounding_box, merge_elements_metadata

from dotenv import load_dotenv
load_dotenv()

rerank = True
router_type = 'llm'

reranker = 'rerank-multilingual-v3.0'
db_name = "adobe_handbook_db"
db_path = Path(__file__).parent / 'data'
persist_directory = db_path / db_name
doc_id_key = "doc_id"
redis_host = os.environ['REDIS_HOST']
redis_port = os.environ['REDIS_PORT']
redis_password = os.environ['REDIS_PASSWORD']
top_k = 10
reranker_top_k = 5

bucket_name = 'hypergai-data'

def chat_llm_stream(user_input, system_prompt='', chat_history=[], temperature=0.2, llm="gpt-4o"):
    messages = [{"role": 'system', "content": system_prompt}] if system_prompt else []
    if chat_history:
        messages+=[msg for msg in chat_history if msg['role']!='image']
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
        st.session_state['messages'].append({"role": "assistant", "content": message})
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

def chat(ori_query):
    # Multi-query generation
    multi_query = chat_llm(MULTI_QUERY_PROMPT.format(question=ori_query), temperature=0)
    pattern = re.compile(r'<question>(.*?)</question>')
    queries = [ori_query] + pattern.findall(multi_query)

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
        match_list = [json.loads(match) for match in matches]
        # Reranking
        if rerank:
            rerank_results = st.session_state['reranker'].rerank(model=reranker, query=ori_query, documents=match_list, rank_fields=['page_content'], top_n=reranker_top_k, return_documents=False)
            rerank_results_index = [result.index for result in rerank_results.results]
            match_list = [match_list[i] for i in rerank_results_index]
        match_list_text = [match['page_content'] for match in match_list]
    else:
        with st.chat_message("assistant"):
            st.write(NO_RELEVANT_FILES)
            st.session_state['messages'].append({"role": "assistant", "content": NO_RELEVANT_FILES})
            return

    result_text = '\n\n'.join(f'{i+1}. {t}' for i, t in enumerate(match_list_text))
    response = chat_llm_stream(CHAT_USER_PROMPT.format(context=result_text, question=ori_query), 
                               CHAT_SYSTEM_PROMPT, 
                               st.session_state['messages']
                               )
    return response, match_list

# Begin of Streamlit UI Code
st.title("Employee Handbook Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['context'] = ''

sesstion_state_name = ['vectorstore', 'docstore', 'llm_client', 'reranker', 's3_client']
init = initialize_chain()
for name, func in zip(sesstion_state_name, init):
    st.session_state[name] = func

# Display chat messages from history on app rerun
for message in st.session_state['messages']:
    if message["role"]=='image':
        with st.chat_message('assistant'):
            st.image(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input('Message'):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    response, match_list = chat(prompt)
    # Display reference
    with st.chat_message("assistant"):
        reference_response = "You can refer to the following highlighted context:"
        st.write(reference_response)
        st.session_state['messages'].append({"role": "assistant", "content": reference_response})
    doc = match_list[0]
    base64_elements_str = doc['metadata']['orig_elements']
    elements = elements_from_base64_gzipped_json(base64_elements_str)
    page, bbox = merge_elements_metadata(elements)
    image = convert_from_path(persist_directory / 'adobe_handbook.pdf', first_page=page, last_page=page)[0]
    
    size = (int(elements[0].metadata.coordinates.system.width), int(elements[0].metadata.coordinates.system.height))
    img_with_bbox = draw_bounding_box(image, list(bbox), size)
    with st.chat_message("assistant"):
        reference_response = f'Page {page}, {doc['metadata']['summary']}'
        st.write(reference_response)
        st.image(img_with_bbox)
        st.session_state['messages'].append({"role": "assistant", "content": reference_response})
        st.session_state['messages'].append({"role": "image", "content": img_with_bbox})
