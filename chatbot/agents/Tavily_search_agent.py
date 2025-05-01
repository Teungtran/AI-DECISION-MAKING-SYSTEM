import os
from langchain_community.tools import TavilySearchResults
import datetime
import warnings
warnings.filterwarnings('ignore')
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_openai import AzureChatOpenAI

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Initialize Azure OpenAI model
amazon_help_url  = [
    "https://www.amazon.in/gp/help/customer/display.html?nodeId=202111910",
    "https://www.amazon.com/gp/help/customer/display.html?nodeId=GLHXEX85MENUE4XFhttps://www.cnn.com",
    "https://www.amazon.com/gp/help/customer/display.html?nodeId=GZ5R2Y8QHENSLW75",
    "https://www.amazon.com/gp/help/customer/display.html?nodeId=GZXW7X6AKTHNUP6H",
    "https://www.amazon.com/gp/help/customer/display.html?nodeId=G8UUV583T397XQ3R"
]

genllm = AzureChatOpenAI(openai_api_key=OPENAI_API_KEY,
                            openai_api_version="2024-08-01-preview",
                            model ="gpt-4o-mini",
                            openai_api_base=AZURE_ENDPOINT, temperature=0) 


# Initialize Tavily search tool
tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_images=False,
    include_answer=True,
    include_raw_content=True,
    api_key=TAVILY_API_KEY,
    description="A search engine optimized for comprehensive, accurate, and trusted results. Input should be a search query.",
    source_urls=amazon_help_url 
)

def create_chain():
    prompt = ChatPromptTemplate(
    [
        ("system", f"You are a search engine.Your task is to find the latest news about the user's query on Amazon news, market news, and other e-commerce and marketing topics. Use the following format to answer the user's query.The timeline of the years are from 2023-2025. Use that  timeline to find the latest news."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)
    llm_with_tools = genllm.bind_tools([tool])
    llm_chain = prompt | llm_with_tools
    return llm_chain

@chain
def Search_logic(user_input: str):
    config = RunnableConfig()
    llm_chain = create_chain()
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    response = llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
    return response.content

def Search_agent(question: str):
    response = Search_logic.invoke(question)
    return response
