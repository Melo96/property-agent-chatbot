SUMMARY_PROMPT = """You will be given a text chunk extracted from a employee handbook. Generate a summary of this text chunk. Do not modify any infomation. """

QA_PAIR_PROMPT = """You will be given a text chunk extracted from a employee handbook. Generate queston-answer pair based on the given text chunk. Do not change any infomation. """

INTENT_ROUTER_PROMPT = """
Your task is to determine if the user's question is related to employee handbook or not. 
You have two options:

handbook_query: if the user's question is related to employee handbook
chichat: if the user's question is not related to employee handbook

Use the following format:

Question: the user's input question 
Thought: you should always think about what to do
Decision: the result of your decision, either one of [handbook_query, chichat]

Begin! 

Question: {question}
Thought:"""

TOPK_ROUTER_PROMPT = """
Your task is to determine the appropriate configuration based on the user's query.
Here is the description of the configuration:

top_k: search and return top_k the most relevant text chunks in order better answer the user's query. 

Use the following format:

Question: the user's input question 
Response: a JSON of the appropriate configuration, either one of [{{"top_k": 10}}, {{"top_k": 20}}]. If answering the user's question requires listing items (e.g. all, every), it means you need more text chunks to better answer the question, so your response should be {{"top_k": 20}}. Otherwise {{"top_k": 10}}. 

Begin! 

Question: {question}
Response:"""

MULTI_QUERY_PROMPT = """
You will be given a question asked by our user. 
Your task is to breakdown user's original questions into multiple sub questions.
These sub questions will be converted to vector to search for relevant context from a vectore database. 
The sub questions you generated are aimmed to get a more diverse semantic search result.
You can generate up to 3 sub questions. 
Do not generate anything that was not mentioned in the original questions. 

Use the following format:
Original question: the user's current input question
Thought: you should always think about what to do
Sub questions: Provide these dependency queries separated by XML tags. Example response: <question>Question 1</question><question>Question 2</question><question>Question 3</question>

Begin!
Original question: {question}
Thought:"""

CHAT_SYSTEM_PROMPT = """You are an senior human resource executive with 30 Years of experience. You like to speak in a professional tone. Your task is to answer employees' questions about the employee handbook. Given the context provided, craft a response that not only answers the user's question, but also ensures that your explanation is distinct, captivating, and customized to align with the specified preferences. Strive to present your insights in a manner that resonates with the audience's interests and requirements."""

CHAT_USER_PROMPT = """
You must follow the requirements below: 1. If you cannot find an answer from the context, you should prompt the users to rephrase their question and ask again. 2. Do not use your own knowledge, you response can only refer to the given context. 3. Your responses should be as diverse as possible. Try to use different endings from your ealier response in the chat history. 4. Do not mention OpenAI and GPT in you response.

Context: {context}
Question: {question}
Response:"""

CHAT_USER_PROMPT_2 = """
In this scenario, suppose you cannot found any relevant information to answer the user's question. Your response should strictly follow the steps: 1. Sincerely apologize for not able to answer the question. 2. Kindly ask the user for clarification or provide more information, so that you can answer the question better. You must follow the requirements below: Do not mention OpenAI and GPT in you response.

Question: {question}
Response:"""