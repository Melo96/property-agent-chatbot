SUMMARY_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate a comprehensive summary of this json file. You need to include every single attributes mentioned in the original json file. Do not change any infomation. The generated summary need to be in Chinese. Your response should be around 300 Chinese words. """

QA_PAIR_PROMPT = """You will be given a json file that uses Chinese to describe the attributes of a real estate. Each key represents one attribute. You task is to generate exactly one question-answer pair for each attribute. Do not change any infomation. Skip attributes with empty values. The generated question-answer pairs need to be in Chinese."""

# QUERY_ROUTER_PROMPT = """Based on the chat history, determine if the user's input is related to real estates or not. If yes, output 'query'. If not, output 'general'. The following are a few examples. Question: 在么？Output: general. Question: 给我推荐一些房产。Output: query. Question: 那这几个里均价最低的是多少？Output: query. Question: {question} Output: """
QUERY_ROUTER_PROMPT = """
    Based on the chat history, determine if the user's question is related to real estates or not. 
    If yes, output 'query'. If not, output 'general'. 
    Your output should be either 'query' or 'general'.
    Think step by step before you make the final decision.
    The following are a few examples:
    HISTORY:
    []
    NOW INPUT: 在么？
    THOUGHT: 用户输入与房产无关
    OUTPUT: general. 
    -------------------
    HISTORY:
    [{{"role": "user", "content": "三亚有房子吗"}},
     {{"role": "assistant", "content": "当然有啦~ 三亚有很多优质的房子供你选择哦~ 根据你的需求，我推荐君和君泰楼盘。这个楼盘位于三亚市吉阳区荔枝沟出口路，靠近琼州学院，地理位置非常优越。"}}]
    NOW INPUT: 均价多少？
    THOUGHT: 根据上下文，用户此处是在询问上文提到的“君和君泰”的均价。所以输入与房产有关。
    OUTPUT: query
    -------------------
    HISTORY:
    [{{"role": "user", "content": "三亚有房子吗"}},
     {{"role": "assistant", "content": "当然有啦~ 三亚有很多优质的房子供你选择哦~ 根据你的需求，我推荐君和君泰楼盘。这个楼盘位于三亚市吉阳区荔枝沟出口路，靠近琼州学院，地理位置非常优越。"}}]
    NOW INPUT: 看看图？
    THOUGHT: 根据上下文，用户此处是在询问上文提到的“君和君泰”的图片。所以输入与房产有关。
    OUTPUT: query
    -------------------
    HISTORY:
    {history}
    NOW INPUT: {question}
    THOUGHT:"""


RAG_ROUTER_PROMPT = """
    You will be given a query input by our user. 
    Your task is to determine if the user's question can be answered only by infomation mentioned in the chat history. 
    Think step by step before you generate the new question.
    If so, output 'yes'. If not, output 'no'. Your output should be exactly one word. Here are a few examples:
    HISTORY:
    []
    NOW QUESTION: 推荐两个三亚的房子?
    THOUGHT: 聊天历史为空，因此没有足够的信息来回答
    SUFFICIENT?: no
    -------------------
    HISTORY:
    [{{"role": "user", "content": "详细介绍一下融创日月湾"}},
     {{"role": "assistant", "content": "融创日月湾楼盘位于海南省万宁市日月湾旅游度假区，属于万宁市区域。该楼盘目前在售，均价为24000元/平方米，主力户型建筑面积为66㎡-153㎡，现房产品。楼盘的开发商是万宁南山融创实业有限公司，物业公司为融创物业，物业费为5.8元/平方米。楼盘的预售证号为【2017】万房预字(0056)号。"}}]
    NOW QUESTION: 融创日月湾的均价是多少?
    THOUGHT: 聊天历史中有提到融创日月湾的均价，因此信息足够
    SUFFICIENT?: yes
    -------------------
    HISTORY:
    [{{"role": "user", "content": "推荐万宁房产"}},
     {{"role": "assistant", "content": "根据您的需求，我推荐融创日月湾～这个楼盘位于海南省万宁市日月湾旅游度假区，交通便利，周边配套设施齐全，适合度假和居住哦～希望您喜欢这个推荐～如果需要更多信息，请随时告诉我哦～"}}]
    NOW QUESTION: 详细介绍一下融创日月湾，
    THOUGHT: 上文有提到融创日月湾，但并没有详细信息，因此信息并不足够
    SUFFICIENT?: no
    -------------------
    HISTORY: 
    {history}
    NOW QUESTION: {question}
    THOUGHT: """

IMAGE_ROUTER_PROMPT = """
    Your task is to determine whether the user is explicitly asking for house images. 
    If so, output the names of the houses and seperate them with XML tags <house> </house>. 
    If the user did not ask for images, or the user did not provide the house name, just output an empty string. 
    You should not include the following words in the house name: '楼盘'
    Here are a few examples:
    HISTORY:
    []
    NOW QUESTION: 发个户型图
    THOUGHT: 用户提问中明确要求发送户型图，但并没有提供对应的楼盘名称，因此无法发送
    OUTPUT JSON: {{"need_image": "yes", "result": ""}}
    -------------------
    HISTORY:
    [{{"role": "user", "content": "请推荐三亚的房产"}},
     {{"role": "assistant", "content": "好的呀～我再推荐一个三亚的房产给您哦～这个项目是**鲁能三亚湾港湾二区**，位于三亚市天涯区，交通便利，周边配套设施齐全，适合各种需求的购房者哦～希望您喜欢这个推荐～如果需要更多信息，随时告诉我哦～"}}]
    NOW QUESTION: 发个户型图
    THOUGHT: 用户提问中明确要求发送户型图。根据上下文，需要给用户发送”鲁能三亚湾港湾二区“的户型图。
    OUTPUT JSON: {{"need_image": "yes", "result": "<house>鲁能三亚湾港湾二区</house>"}}
    -------------------
    HISTORY:
    [{{"role": "user", "content": "给我推荐三亚的两个房产"}},
     {{"role": "assistant", "content": "根据您的需求，我推荐以下两套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。"}}]
    NOW QUESTION: 可以发图片给我参考一下吗
    THOUGHT: 用户提问中明确要求发送户型图。根据上下文，需要给用户发送”三亚繁华里“和“君和君泰”的户型图。
    OUTPUT JSON: {{"need_image": "yes", "result": "<house>三亚繁华里</house><house>君和君泰</house>"}}
    -------------------
    HISTORY:
    [{{"role": "user", "content": "给我推荐三亚的两个房产"}},
     {{"role": "assistant", "content": "根据您的需求，我推荐以下两套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。"}}]
    NOW QUESTION: 均价最低的是哪个
    THOUGHT: 用户提问中没有要求发送户型图。
    OUTPUT JSON: {{"need_image": "no", "result": ""}}
    -------------------
    HISTORY:
    {history}
    NOW QUESTION: {question}
    THOUGHT: """

COREFERENCE_RESOLUTION = """
    Please return a new question with the following requirements:
    1. If there are pronouns or conditions are missing in the question, please make a complete question according to the context.
    2. If the question is complete, please keep the original question.
    Do not modify or add any other infomation of the original question. 
    Think step by step before you generate the new question.
    Here are a few examples:
    HISTORY:
    [{{"role": "user", "content": "给我推荐几套三亚的房产"}},
     {{"role": "assistant", "content": "根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。"}}]
    NOW QUESTION: 这些房产里哪些适合养老度假?
    THOUGHT: “这些房产”指代的是上文中提及的4个房产。所以我需要把“这些房产”替换为上文提到的“三亚繁华里”，“君和君泰”，“三亚中央公馆”和“三亚凤凰苑”.
    NEED COREFERENCE RESOLUTION: Yes
    OUTPUT QUESTION: “三亚繁华里”，“君和君泰”，“三亚中央公馆”和“三亚凤凰苑”里哪些适合养老度假？
    -------------------
    HISTORY:
    [{{"role": "user", "content": "给我推荐几套三亚的房产"}},
     {{"role": "assistant", "content": "根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。"}}]
    NOW QUESTION: 再给我推荐多几个
    THOUGHT: 根据上下文，此处用户在询问能否推荐其他的三亚房产。因此需要添加指代
    NEED COREFERENCE RESOLUTION: No
    OUTPUT QUESTION: 再给我推荐多几个三亚的房产
    -------------------
    HISTORY:
    [{{"role": "user", "content": "给我推荐几套三亚的房产,
     {{"role": "assistant", "content": "根据您的需求，我推荐以下几套位于三亚的房产：
        1.三亚繁华里
        2.君和君泰
        3.三亚中央公馆
        4.三亚凤凰苑
        这几套房产各具特色，您可以根据自己的需求和预算进行选择。如果需要更多信息或有其他问题，请随时联系我。"}},
     {{"role": "user", "content": "这些房产里哪些带精装修？"}},
     {{"role": "assistant", "content": "根据提供的信息，以下房产带有精装修：
        1.三亚繁华里：简装
        2.君和君泰：精装修
        3.三亚中央公馆：未提及
        4.三亚凤凰苑：精装修
        所以，君和君泰和三亚凤凰苑是带精装修的。"}}]
    NOW QUESTION: 那这两个里哪个的均价较低?
    THOUGHT: “这两个”指代的是上文中提到的带精装修的两个房产，“君和君泰”和“三亚凤凰苑”。所以我需要把“这两个”替换为上文提到的“君和君泰”和“三亚凤凰苑”.
    NEED COREFERENCE RESOLUTION: Yes
    OUTPUT QUESTION: 那“君和君泰”和“三亚凤凰苑”里哪个均价较低？
    -------------------
    HISTORY:
    [{{"role": "user", "content": "请推荐三亚的房产"}},
     {{"role": "assistant", "content": "好的呀～我再推荐一个三亚的房产给您哦～这个项目是**鲁能三亚湾港湾二区**，位于三亚市天涯区，交通便利，周边配套设施齐全，适合各种需求的购房者哦～希望您喜欢这个推荐～如果需要更多信息，随时告诉我哦～"}}]
    NOW QUESTION: 发个户型图
    THOUGHT: 根据上下文，这里用户需要的是“鲁能三亚湾港湾二区”的户型图。所以我需要在问题中加上这个指代
    OUTPUT QUESTION: 发个'鲁能三亚湾港湾二区'的户型图
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

    Provide these sub questions separated by XML tags. 
    Example response: <question>Question 1</question><question>Question 2</question><question>Question 3</question>

    Original question: {question}
    Sub questions:"""

CHAT_SYSTEM_PROMPT = """You are an expert Real Estate Agent with 30 Years of experience. Your task is to offer a deep-dive consultation tailored to the client's issue. You like to speak in a lively and cute tone. Do not mention OpenAI and GPT in you response."""
RAG_USER_PROMPT = """问题：{question}, 相关信息：{context}, 回答："""
RAG_SYSTEM_PROMPT = """你是一个有着30年从业经验的职业房产客服。你的名字叫“小盖”。你喜欢用活泼可爱的语气说话。你习惯在句尾加波浪号“～”。你在“爱房网”工作。你的任务是帮助客户找到最符合他们需求的房产。如果客户没有明确提到需要你推荐多个房源，默认只推荐最符合用户要求的那一个。如果用户没有要你提供房源的详细信息，你只需要告诉客户推荐的房源名称，并在回复中加粗表示房源名称。你的回答需要优先参考聊天记录。如果聊天记录不能提供足够的信息，请参考提供的房产相关信息。每个不同的房产会用XML标签隔开，例如：<context>房产1</context><context>房产2</context>。请严格按照用户问题的要求回答，不要回答额外的信息。你的回复里请不要包含XML标签。如果你无法回答此问答，请回复礼貌的告知用户你不知道。你每次回复的结尾可以尽可能地更多样化。不要在你的回复里提到OpenAI和GPT"""

# RAG_SYSTEM_PROMPT = """You are an expert Real Estate Agent with 30 Years of experience. Your task is to offer a deep-dive consultation tailored to the client's issue. You like to speak in a lively and cute tone. Do not mention OpenAI and GPT in you response. Given the context provided above, craft a response that not only answers the user's question, but also ensures that your explanation is distinct, captivating, and customized to align with the specified preferences. If the customer does not explicitly asking for multiple recommendations, default to recommending only the one that best fits the user's requirements. If the customer does not ask for detailed information about the property, you only need to provide the name of the recommended property and a one-sentence recommendation. Your response should always prioritize referencing the chat history. If the chat history does not provide sufficient information, please refer to the provided context. In the context, different properties are seperated by XML tags. For example: <context>property 1</context><context>property 2</context>. Strive to present your insights in a manner that resonates with the audience's interests and requirements. If you are unable to answer the question, please politely inform the user that you do not know. After answering, you can ask some questions to the user to make sure the user is satisfied with your response. You also need to ask the clients to leave their contact information so that we can arrange house tour for them. The endings of your responses should be different from the endings of your ealier response in the chat history"""

# RAG_USER_PROMPT = """Context: {context}. Question: {question}. Response:"""

# QUERY_PLANNER_PROMPT = """
#     You are a world class query planning algorithm capable of breaking apart questions into its dependency queries 
#     such that the answers can be used to inform the parent question.
#     Do not answer the questions, simply provide a correct compute graph with good specific questions to ask and relevant dependencies. 
#     Your need to generate {queryCount} of these dependency queries.
#     Before you call the function, think step-by-step to get a better understanding of the problem.
#     The generated queries should be in Chinese.

#     Provide these queries separated by newlines between XML tags. For example:

#     <questions>
#     Question 1
#     Question 2
#     Question 3
#     </questions>

#     Original question: {question}
#     Dependency queries:
# """

# FILTER_PROMPT = """
#     You are an assistant for extracting keywords from query.
#     All possible keywords are listed below:
#     keywords: DC宽温系列, DM多轴系列, DN单板系列, DP精巧系统, DR环形系列, Dservo系列, DS常规系列, UF2 Uservo-Flex系列双编版本, Uservo-Flex系列
    
#     Your task is to extract one or multiple keywords from the query.
#     Return the extracted keywords in the following format:
#     keyword1|keyword2|keyword3
#     If you cannot extract any keywords from query, please return all of the possible keywords. 
    
#     Here are a few examples:
#     Query: DC宽温系列的产品有哪些？
#     Resposne: DC宽温系列
    
#     Query: Uservo-Flex系列EtherCAT驱动器的模拟量输入接线方法？
#     Resposne: Uservo-Flex系列|UF2 Uservo-Flex系列双编版本
    
#     Query: 磁环怎么接?
#     Resposne: DC宽温系列|DM多轴系列|DN单板系列|DP精巧系统|DR环形系列|Dservo系列|DS常规系列|UF2 Uservo-Flex系列双编版本|Uservo-Flex系列
    
#     Task begin!
#     Query: {query}
#     Response:
# """

# IMAGE_ROUTER_PROMPT = """
#     Suppose we need to send house images to the user if any of the following two scenarios were detected in the last message from the user or the last message from the assistant: 
#     1, When the user explicitly asking for house images. 
#     2, When the assistant mentions a new house that were never seen before in the chat history. 
#     You task is to determine whether we need to send house images to the user. 
#     If so, output the names of the houses and seperate them with XML tags <house> </house>. 
#     If not, just output one word 'empty'. 
#     You should not include the following words in your output as the house name: '楼盘'
#     Here are a few examples:
#     HISTORY:
#     [user: 请推荐三亚的房产,
#      assistant: 好的呀～我再推荐一个三亚的房产给您哦～这个项目是**鲁能三亚湾港湾二区**，位于三亚市天涯区，交通便利，周边配套设施齐全，适合各种需求的购房者哦～希望您喜欢这个推荐～如果需要更多信息，随时告诉我哦～]
#     THOUGHT: 最新的回复中提到**鲁能三亚湾港湾二区**，此前没有提到这个楼盘，因此需要给用户发送户型图。
#     NEED IMAGE: yes
#     OUTPUT:  <house>鲁能三亚湾港湾二区</house>
#     -------------------
#     HISTORY:
#     [user: 请推荐三亚的房产,
#      assistant: 好的呀～我再推荐一个三亚的房产给您哦～这个项目是**鲁能三亚湾港湾二区**，位于三亚市天涯区，交通便利，周边配套设施齐全，适合各种需求的购房者哦～希望您喜欢这个推荐～如果需要更多信息，随时告诉我哦～,
#      assistant: 下面给您展示一些鲁能三亚湾港湾二区的简介和户型图，您可以参考看看～,
#      user: 鲁能三亚湾港湾二区的均价是多少？,
#      assistant: 鲁能三亚湾港湾二区的均价大约是每平方米3万元左右哦～具体价格可能会根据楼层、朝向和户型有所不同～如果您需要更详细的信息或有其他问题，请随时告诉我哦～希望这个信息对您有帮助～]
#     THOUGHT: 最新回复中没有提到新的楼盘，因此不需要给用户发送户型图
#     NEED IMAGE: no
#     OUTPUT: empty
#     -------------------
#     HISTORY:
#     {history}
#     THOUGHT: """