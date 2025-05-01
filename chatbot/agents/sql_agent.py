import os
from langchain_community.utilities import SQLDatabase
from ..prompt_template  import SQL_AGENT_PROMPT  

import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType, create_sql_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

OPEN_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

model = None


# Initialize database connection
def connect_to_db(server: str, database: str) -> SQLDatabase:
    db_uri = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    return SQLDatabase.from_uri(db_uri)

# Initialize SQL agent
def setup_query_engine(db: SQLDatabase):
    llm = AzureChatOpenAI(openai_api_key=OPEN_API_KEY,
                        model = 'gpt-4o-mini',
                        openai_api_version="2024-08-01-preview",openai_api_base=AZURE_ENDPOINT,temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    def execute_query(input_text: str):
        prompt = ChatPromptTemplate.from_messages(
            [("system", SQL_AGENT_PROMPT), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")])
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            top_k=None,
            prompt=prompt,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        result = agent_executor.invoke(input_text)
        return result
    return execute_query
#______________________Generate answer___________________
def SQL_tool(user_query: str):
    try:
        db = connect_to_db(server="DESKTOP-LU731VP\SQLEXPRESS", database="DE_AN")
        execute_with_query = setup_query_engine(db)
        result = execute_with_query(user_query)
        return result["output"]
    except Exception as e:
        return None, f"Error generating response: {str(e)}" 
