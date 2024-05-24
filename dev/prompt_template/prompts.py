SUMMARY_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate a comprehensive summary of this json file. You need to include every single attributes mentioned in the original json file. Do not change any infomation. The generated summary need to be in Chinese. Your response should be around 300 Chinese words. """

QA_PAIR_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate question-answer pairs. You need to generate exactly one question-answer pair for each attribute. Do not change any infomation. The generated question-answer pairs need to be in Chinese."""

QUERY_ROUTER_PROMPT = """Determine if the user's question is about real estates or not. If yes, output 'query'. If not, output 'general'. The following are a few examples. Question: 在么？Output: general. Question: 给我推荐一些房产。Output: query. Question: 那这几个里均价最低的是多少？Output: query. Question: {question} Output: """

RAG_ROUTER_PROMPT = """You will be given a query input by our user. Your task is to determine if the user's question can be answered only by infomation mentioned in the chat history. If so, output 'yes'. If not, output 'no'. Your output should be exactly one word. Question: {question} Output: """

COREFERENCE_RESOLUTION = """
    Please return a new question with the following requirements:
    1. If there are pronouns or conditions are missing in the question, please make a complete question according to the context.
    2. If the question is complete, please keep the original question.
    Think step by step before you generate the new question.
    Here are a few examples:
    HISTORY:
    [Q: 给我推荐几套三亚的房产,
     A: 根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。]
    NOW QUESTION: 这些房产里哪些适合养老度假?
    THOUGHT: “这些房产”指代的是上文中提及的4个房产。所以我需要把“这些房产”替换为上文提到的“三亚繁华里”，“君和君泰”，“三亚中央公馆”和“三亚凤凰苑”.
    NEED COREFERENCE RESOLUTION: Yes
    OUTPUT QUESTION: “三亚繁华里”，“君和君泰”，“三亚中央公馆”和“三亚凤凰苑”里哪些适合养老度假？
    -------------------
    HISTORY:
    [Q: 给我推荐几套三亚的房产,
     A: 根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。
    NOW QUESTION: 再给我推荐多几个
    THOUGHT: 用户问题中没有指代。所以不需要替换
    NEED COREFERENCE RESOLUTION: No
    OUTPUT QUESTION: 再给我推荐多几个
    -------------------
    HISTORY:
    [Q: 给我推荐几套三亚的房产,
     A: 根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。
     Q: 这些房产里哪些带精装修？,
     A: 根据提供的信息，以下房产带有精装修：
        1.三亚繁华里：简装
        2.君和君泰：精装修
        3.三亚中央公馆：未提及
        4.三亚凤凰苑：精装修
        所以，君和君泰和三亚凤凰苑是带精装修的。]
    NOW QUESTION: 那这两个里哪个的均价较低?
    THOUGHT: “这两个”指代的是上文中提到的带精装修的两个房产，“君和君泰”和“三亚凤凰苑”。所以我需要把“这两个”替换为上文提到的“君和君泰”和“三亚凤凰苑”.
    NEED COREFERENCE RESOLUTION: Yes
    OUTPUT QUESTION: 那“君和君泰”和“三亚凤凰苑”里哪个均价较低？
    -------------------
    HISTORY:
    [{history}]
    NOW QUESTION: {question}
    THOUGHT: """

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

CHAT_SYSTEM_PROMPT = """你是一个有着30年从业经验的职业房产客服。你的名字叫“小盖”。你在“爱房网”工作。"""

# 你需要精心撰写你的回答，还要确保你的解释与众不同，引人入胜，并符合特定的偏好，努力以一种能与用户产生共鸣并吸引用户兴趣的方式来表达见解，最终吸引用户购买你的房产
RAG_SYSTEM_PROMPT = """你是一个有着30年从业经验的职业房产客服。你的名字叫“小盖”。你在“爱房网”工作。你的任务是帮助客户找到最符合他们需求的房产。你必须让用户相信你的专业水平，并且最终满意你提供的方案，吸引用户购买我们的提供的房源。每次回答之前，会有不同房产的相关信息提供给你。你的回答需要优先参考聊天记录。如果聊天记录不能提供足够的信息，请参考提供的房产相关信息。每个不同的房产会用XML标签隔开，例如：<context>房产1</context><context>房产2</context>。请严格按照用户问题的要求回答，不要回答额外的信息。你的回复里请不要包含XML标签。如果你无法回答此问答，请回复礼貌的告知用户你不知道。你每次回复的结尾可以尽可能地更多样化。"""

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