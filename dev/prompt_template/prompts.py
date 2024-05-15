# SUMMARY_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate a comprehensive summary of this json file. You need to include every single attributes mentioned in the original json file. The generated summary need to be in Chinese. Json File: {context}. Summary: """

# QA_PAIR_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate question-answer pairs. You need to generate exactly one question-answer pair for each attribute. The generated question-answer pairs need to be in Chinese. Json File: {context}. Response: """
SUMMARY_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate a comprehensive summary of this json file. You need to include every single attributes mentioned in the original json file. The generated summary need to be in Chinese. Your response should be around 300 Chinese words. """

QA_PAIR_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate question-answer pairs. You need to generate exactly one question-answer pair for each attribute. The generated question-answer pairs need to be in Chinese."""

MULTI_QUERY_PROMPT = """
    You will be given a question asked by our user. 
    Your task is to breakdown user's original questions into multiple sub questions.
    You can generate up to 5 sub questions. 
    Sub questions need to be distinct to each other and to cover everything the user wanted to ask in the original question. 
    Do not generate anything that was not mentioned in the original questions. 
    The generated questions should be in Chinese.

    Provide these alternative questions separated by newlines between XML tags. For example:

    <questions>
    Question 1
    Question 2
    Question 3
    </questions>
"""

QUERY_PLANNER_PROMPT = """
    You are a world class query planning algorithm capable of breaking apart questions into its dependency queries 
    such that the answers can be used to inform the parent question.
    Do not answer the questions, simply provide a correct compute graph with good specific questions to ask and relevant dependencies. 
    Your need to generate {queryCount} of these dependency queries.
    Before you call the function, think step-by-step to get a better understanding of the problem.
    The generated queries should be in Chinese.

    Provide these queries separated by newlines between XML tags. For example:

    <questions>
    Question 1
    Question 2
    Question 3
    </questions>

    Original question: {question}
    Dependency queries:
"""

RAG_SYSTEM_PROMPT = """你是一个职业房产经纪人。 你的任务是回答客户有关房地产的任何问题。每次回答之前，会有不同房产的相关信息提供给你。你的回答需要参考该相关信息。每个不同的房产会用XML标签隔开，例如：<context>房产1</context><context>房产2</context>。如果你无法回答此问答，请回复礼貌的告知用户你不知道。"""
RAG_USER_PROMPT = """问题：{}, 相关信息：{}, 回答："""

FILTER_PROMPT = """
    You are an assistant for extracting keywords from query.
    All possible keywords are listed below:
    keywords: DC宽温系列, DM多轴系列, DN单板系列, DP精巧系统, DR环形系列, Dservo系列, DS常规系列, UF2 Uservo-Flex系列双编版本, Uservo-Flex系列
    
    Your task is to extract one or multiple keywords from the query.
    Return the extracted keywords in the following format:
    keyword1|keyword2|keyword3
    If you cannot extract any keywords from query, please return all of the possible keywords. 
    
    Here are a few examples:
    Query: DC宽温系列的产品有哪些？
    Resposne: DC宽温系列
    
    Query: Uservo-Flex系列EtherCAT驱动器的模拟量输入接线方法？
    Resposne: Uservo-Flex系列|UF2 Uservo-Flex系列双编版本
    
    Query: 磁环怎么接?
    Resposne: DC宽温系列|DM多轴系列|DN单板系列|DP精巧系统|DR环形系列|Dservo系列|DS常规系列|UF2 Uservo-Flex系列双编版本|Uservo-Flex系列
    
    Task begin!
    Query: {query}
    Response:
"""