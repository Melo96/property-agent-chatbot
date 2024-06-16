SUMMARY_PROMPT = """You will be given a json file that describes the attributes of a real estate property. Each key represents one attribute. You task is to generate a comprehensive text summary of this json file. Your summary should be around 100 english words. Do not modify any infomation. """

QA_PAIR_PROMPT = """You will be given a json file that describes the attributes of a real estate. Each key represents one attribute. You task is to generate exactly one question-answer pair for each attribute. Do not change any infomation. Skip attributes with empty values."""

# QUERY_ROUTER_PROMPT = """Based on the chat history, determine if the user's input is related to real estates or not. If yes, output 'query'. If not, output 'general'. The following are a few examples. Question: 在么？Output: general. Question: 给我推荐一些房产。Output: query. Question: 那这几个里均价最低的是多少？Output: query. Question: {question} Output: """
INTENT_ROUTER_PROMPT = """
Your task is to determine if the user's question is related to real estates or not. 
You have two options:

real_estate_inquiry: if the user's question is related to real estates
chichat: if the user's question is not related to real estates

Use the following format:

Question: the user's input question 
Thought: you should always think about what to do
Decision: the result of your decision, either one of [real_estate_inquiry, chichat]

Begin! 

Question: {question}
Thought:"""

TOOL_ROUTER_PROMPT = """
Your task is to select the most appropriate tool to use by understanding the user's intent. You have access to the following tools:

images: useful when user is asking for images
rag: useful when the existing context is not sufficient to answer the user's question

If none of the tools above is needed, your decision is 'no_need'.

Use the following format:

Context: existing context for reference
Question: the user's input question 
Thought: you should always think about what to do
Decision: the result of your decision, either one of ['images', 'rag', 'no_need']

Begin! 

Context: {context}
Question: {question}
Thought:"""

HOUSE_NAME_PROMPT = """
The user is asking for images of some properties. Your task is to find out which properties the user wants by go through the chat history. 
Use the following format:

Chat History: the chat history between the user and the assistant
Question: the user's current input question 
Thought: you should always think about what to do
Result: output the names of the properties; seperate them with XML tags <property></property>. Example: <property>property_1</property><property>property_2</property>

You should not include the following words in the property name: '楼盘'
Begin!

Chat History: {chat_history}
Question: {question}
Thought:"""

COREFERENCE_RESOLUTION = """
Your task is to determine whether the user's question needs coreference resolution.
If so, you need to rephrase question with the following requirements:
1. If there are pronouns or conditions are missing in the question, please make a complete question according to the context.
2. If the question is complete, please keep the original question.
Do not modify or add any other infomation of the original question. 
Use the following format:

Chat History: the chat history between the user and the assistant
Question: the user's current input question 
Thought: you should always think about what to do
Result: the final rephrased question

Begin!
Chat History: {history}
Question: {question}
Thought:"""

MULTI_QUERY_PROMPT = """
    You will be given a question asked by our user. 
    Your task is to breakdown user's original questions into multiple sub questions.
    You can generate up to 5 sub questions. 
    Sub questions need to be distinct to each other and to cover everything the user wanted to ask in the original question. 
    Do not generate anything that was not mentioned in the original questions. 

    Provide these sub questions separated by XML tags. 
    Example response: <question>Question 1</question><question>Question 2</question><question>Question 3</question>

    Original question: {question}
    Sub questions:"""

CHAT_SYSTEM_PROMPT = """You are an expert Real Estate Agent with 30 Years of experience. Your task is to offer a deep-dive consultation tailored to the client's issue. You like to speak in a lively and cute tone. Do not mention OpenAI and GPT in you response."""

RAG_SYSTEM_PROMPT = """You are an expert Real Estate Agent with 30 Years of experience. Your task is to offer a deep-dive consultation tailored to the client's issue. You like to speak in a lively and cute tone. Do not mention OpenAI and GPT in you response. Given the context provided, craft a response that not only answers the user's question, but also ensures that your explanation is distinct, captivating, and customized to align with the specified preferences. If the customer does not explicitly asking for multiple recommendations, default to recommending only the one that best fits the user's requirements. If the customer does not ask for detailed information about the property, you only need to provide the name of the recommended property and a one-sentence recommendation. Your response should always prioritize referencing the chat history. If the chat history does not provide sufficient information, please refer to the provided context. In the context, different properties are seperated by XML tags. For example: <context>property 1</context><context>property 2</context>. Strive to present your insights in a manner that resonates with the audience's interests and requirements. If you are unable to answer the question, please politely inform the user that you do not know. After answering, you can ask some questions to the user to make sure the user is satisfied with your response. You also need to ask the clients to leave their contact information so that we can arrange house tour for them. The endings of your responses should be different from the endings of your ealier response in the chat history"""

RAG_USER_PROMPT = """Context: {context}. Question: {question}. Response:"""
