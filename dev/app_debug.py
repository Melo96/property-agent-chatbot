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
from io import BytesIO
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.storage import RedisStore

from prompt_template.prompts import *
from prompt_template.response import *
from utils.utils import output_parser

from dotenv import load_dotenv
load_dotenv()

rerank = False
router_type = 'llm'

reranker = 'rerank-english-v3.0'
db_name = "mock_db"
db_path = Path(__file__).parent / 'data/mock'
persist_directory = db_path / 'mock_db'
doc_id_key = "property_id"
# db_name = "summaries"
# db_path = Path(__file__).parent / 'data/0528'
# persist_directory = db_path / 'chroma_openai_0528'
# doc_id_key = "楼盘ID"
redis_host = os.environ['REDIS_HOST']
redis_port = os.environ['REDIS_PORT']
redis_password = os.environ['REDIS_PASSWORD']
top_k = 10
reranker_top_k = 10

bucket_name = 'hypergai-data'

assistant_icon = Image.open(Path(__file__).parent / 'data/logos/LogoIcon.png')
user_icon = Image.open(Path(__file__).parent / 'data/logos/image.png')

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
                with st.chat_message("assistant", avatar=assistant_icon):
                    st.markdown(message, unsafe_allow_html=True)
                st.session_state['display_messages'].append({"role": "assistant", "content": message})
                message = ''
            else:
                message+=text
    # Print last chunk of text
    if message:
        with st.chat_message("assistant", avatar=assistant_icon):
            st.write(message, unsafe_allow_html=True)
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
        with st.chat_message("assistant", avatar=assistant_icon):
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

    # Load name2id map
    with open(db_path / "name2id.json", 'r') as f:
        name2id = json.load(f)
    return [vectorstore, docstore, llm_client, reranker, s3_client, name2id]

def retrive_img(ori_query):
    # Get house names
    result = chat_llm(
        HOUSE_NAME_PROMPT.format(
            chat_history=json.dumps(st.session_state['messages'], ensure_ascii=False),
            question=ori_query
        ),
        temperature=0
    )
    print(f'Image house names: {result}')
    result = output_parser(result, 'Result:')
    pattern = re.compile(r'<property>(.*?)</property>')
    houses_list = [house_id for house_id in pattern.findall(result) if house_id in st.session_state['name2id']]
    if houses_list:
        for house in houses_list:
            try:
                house_id = st.session_state['name2id'][house]
                img_response = st.session_state['s3_client'].get_object(Bucket=bucket_name, Key=f'mock-real-estate/images_resized/{house_id}.png')
                with st.chat_message("assistant", avatar=assistant_icon):
                    response = HOUSE_IMAGE_RESPONSE.format(house_name=house)
                    st.write(response)
                    st.session_state['display_messages'].append({"role": "assistant", "content": response})
                image_data = img_response['Body'].read()
                with st.chat_message("assistant", avatar=assistant_icon):
                    st.image(BytesIO(image_data))
                    # Add the image to the dispayed chat history
                    st.session_state['display_messages'].append({"role": "image", "content": image_data})
            except:
                with st.chat_message("assistant", avatar=assistant_icon):
                    st.write(f'The requested preperty {house} does not have the corresponding floor plans.')
                    st.session_state['display_messages'].append({"role": "assistant", "content": f'The requested property {house} does not have the corresponding floor plans.'})
    else:
        with st.chat_message("assistant", avatar=assistant_icon):
            st.write(HOUSE_IMAGE_NOE_FOUND_RESPONSE)
            st.session_state['display_messages'].append({"role": "assistant", "content": HOUSE_IMAGE_NOE_FOUND_RESPONSE})

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

@st.spinner('Typing...')
def rag(ori_query):
    # Add rephrased query to the chat history
    st.session_state['messages'].append({"role": "user", "content": ori_query})
    # Tools router
    s = time.time()
    router_result = chat_llm(TOOL_ROUTER_PROMPT.format(context=st.session_state['context'], question=ori_query),
                             temperature=0)
    e = time.time()
    print(f'Tool router: {router_result}, {e-s} seconds')
    router_result = output_parser(router_result, 'Decision:')

    result_text = ''
    if router_result=='images':
        retrive_img(ori_query)
        return
    elif router_result=='rag':
        # Multi-query
        s = time.time()
        match_list_text = multi_queries_retrieval(ori_query)
        e = time.time()
        print(f'Vector search: {e-s} seconds')

        # No relevant context
        if not match_list_text:
            with st.chat_message("assistant", avatar=assistant_icon):
                st.write(NO_RELEVANT_FILES)
                st.session_state['display_messages'].append({"role": "assistant", "content": NO_RELEVANT_FILES})
            return

        # Reranking
        if rerank:
            s = time.time()
            results = st.session_state['reranker'].rerank(model=reranker, query=ori_query, documents=match_list_text, top_n=20, return_documents=True)
            match_list_text = [item.document.text for item in results.results]
            e = time.time()
            print(f'Reranking: {e-s} seconds')
        result_text = ''.join(f'<context>{t}</context>' for t in match_list_text)
        # Store relevant context
        st.session_state['context']+=result_text

    # Response
    result_text = st.session_state['context'] if not result_text else result_text
    response = chat_llm_stream(RAG_USER_PROMPT.format(context=result_text, question=ori_query), 
                               RAG_SYSTEM_PROMPT, 
                               st.session_state['messages']
                               )
    # Add the llm response to the chat history
    st.session_state['messages'].append({"role": "assistant", "content": response})

def chat(ori_query):
    # Add the original query to the dispayed chat history
    st.session_state['display_messages'].append({"role": "user", "content": ori_query})
    # Limit the number of chat history
    st.session_state['messages'] = st.session_state['messages'][-10:]

    # Coreference Resolution
    if st.session_state['messages']:
        s = time.time()
        output = chat_llm(COREFERENCE_RESOLUTION.format(history=json.dumps(st.session_state['messages'], ensure_ascii=False),
                                                        question=ori_query),
                          temperature=0
                        )
        e = time.time()
        print(f'Coreference resolution: {output}, {e-s} seconds')
        ori_query = output_parser(output, 'Result:')

    # Intent Router
    s = time.time()
    # rerank_result = st.session_state['reranker'].rerank(model=reranker, 
    #                                                      query=ori_query, 
    #                                                      documents=st.session_state['routes']['intent_router'], 
    #                                                      top_n=1,
    #                                                      return_documents=False
    #                                                      )
    # rerank_result_index = rerank_result.results[0].index
    # rerank_result = st.session_state['routes']['intent_router'][rerank_result_index]
    # router_result = output_parser(rerank_result, 'Name:')
    router_result = chat_llm(INTENT_ROUTER_PROMPT.format(question=ori_query),
                             temperature=0
                             )
    router_result = output_parser(router_result, 'Decision:')
    e = time.time()
    print(f"Intent Router: {router_result}, {e-s} seconds")

    # Call RAG or directly call LLM
    s1 = time.time()
    if 'real_estate_inquiry' in router_result:
        rag(ori_query)
    else:
        chat_llm_stream(ori_query, CHAT_SYSTEM_PROMPT, chat_history=st.session_state['messages'])

    e1 = time.time()
    print(f'Response: {e1-s1} seconds')

# Begin of Streamlit UI Code
st.title("Intelligent Real Estate Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state['display_messages'] = []
    st.session_state['messages'] = []
    st.session_state['context'] = ''

# Display the user and the assistant's message box in the opposite side
st.markdown(
    """
    <style>
        div.stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4
            { 
                display: flex;
                text-align: right;
                flex-direction: row-reverse;
            }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display chat messages from history on app rerun
for message in st.session_state['display_messages']:
    if message["role"]=='image':
        with st.chat_message('assistant', avatar=assistant_icon):
            st.image(BytesIO(message["content"]))
    elif message["role"]=='assistant':
        with st.chat_message('assistant', avatar=assistant_icon):
            st.write(message["content"])
    elif message["role"]=='user':
        with st.chat_message('user', avatar=user_icon):
            st.write(message["content"])

sesstion_state_name = ['vectorstore', 'docstore', 'llm_client', 'reranker', 's3_client', 'name2id']
init = initialize_chain()
for name, func in zip(sesstion_state_name, init):
    st.session_state[name] = func

if prompt := st.chat_input('Message'):
    # Display user message in chat message container
    with st.chat_message("user", avatar=user_icon):
        st.markdown(prompt)

    # Display assistant response in chat message container
    chat(prompt)
