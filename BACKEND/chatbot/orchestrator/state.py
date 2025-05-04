import os
from pydantic import  Field
from typing import TypedDict, Literal, Optional
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
import warnings
warnings.filterwarnings('ignore')
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
agents = ['amazon_policy','sql_agent','search_engine',"sale_expert",'recall_memory']
def open_ai_model():
    llm = AzureChatOpenAI(openai_api_key=OPENAI_API_KEY,
                        openai_api_version="2024-08-01-preview",
                        model ="gpt-4o-mini",
                        openai_api_base=AZURE_ENDPOINT, temperature=0) 
    return llm
def gg_model():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, max_tokens=2048,top_p=0.95)
    return llm

class Router(TypedDict):
    next: Literal[*agents] = Field(
        None, description="The next agents in the routing process"
    )

class AgentState(TypedDict):
    user_id: str  
    input: str
    decision: Optional[str]  
    output: Optional[ToolMessage]
    reflection_feedback: Optional[str]
    needs_refinement: Optional[bool]
