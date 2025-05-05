from langgraph.graph import StateGraph, START
import warnings
warnings.filterwarnings('ignore')
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from .state import AgentState
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from .memory import save_conversation_to_memory, initialize_checkpointer,get_agent_memory
from.nodes import (
    amazon_policy,
    sale_expert,
    sql_agent,
    search_engine,
    recall_memory,
    reflection_node,
    refined_node,
    route_decision,
    should_refine,
    needs_refinement_check,
    supervisor_node
)

def build_graph():
    # Build the graph
    builder = StateGraph(AgentState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("amazon_policy", amazon_policy)
    builder.add_node("sale_expert", sale_expert)
    builder.add_node("sql_agent", sql_agent)
    builder.add_node("search_engine", search_engine)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("reflection_node", reflection_node)
    builder.add_node("refined_node", refined_node)

    builder.add_edge(START, "supervisor_node")
    
    builder.add_conditional_edges(
        "supervisor_node",
        route_decision,
        {
            "amazon_policy": "amazon_policy",
            "sale_expert": "sale_expert",
            "sql_agent": "sql_agent",
            "search_engine": "search_engine",
            "recall_memory": "recall_memory",
        },
    )
    builder.add_conditional_edges(
        "amazon_policy",
        should_refine,
        {
            "check_reflection": "reflection_node",
            "skip_refinement": END,
        },
    )
    builder.add_conditional_edges(
        "sale_expert",
        should_refine,
        {
            "check_reflection": "reflection_node",
            "skip_refinement": END,
        },
    )

    builder.add_conditional_edges(
        "sql_agent",
        should_refine,
        {
            "check_reflection": "reflection_node",
            "skip_refinement": END,
        },
    )
    builder.add_edge("search_engine", END)
    builder.add_edge("recall_memory", END)

    builder.add_conditional_edges(
        "reflection_node",
        needs_refinement_check,
        {
            "refine": "refined_node",  
            "no_refine": END,          
        },
    )
    builder.add_edge("refined_node", END)
    db_path=r'C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\BACKEND\chatbot\database\chat_history.db'
    memory = initialize_checkpointer(db_path)
    graph = builder.compile(checkpointer=memory,store=get_agent_memory)  

    return graph

_GRAPH_INSTANCE = None


def get_graph():
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is None:
        logger.info("Building new graph instance...")
        _GRAPH_INSTANCE = build_graph()
    return _GRAPH_INSTANCE

def call_ai_agentic(user_input: str, user_id: str = None):
    logger.info("Starting call_ai_agentic")

    graph = get_graph()
    logger.info("Graph loaded")

    initial_state = {
        "input": user_input, 
        "user_id": user_id
    }
    config = {"configurable": { "thread_id": user_id} }        
    events = graph.stream(
        initial_state,
        config=config,
        stream_mode="values"
    )
    chat_history = []
    for event in events:
        chat_history.append(event)
    response = event.get("output")
    response = response.content if isinstance(response, AIMessage) else (response or "No response generated.")
    save_conversation_to_memory(user_input, response, user_id)
    return response
