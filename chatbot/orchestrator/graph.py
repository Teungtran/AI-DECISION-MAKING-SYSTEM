from langgraph.graph import StateGraph, START
import warnings
warnings.filterwarnings('ignore')
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from .state import AgentState
from .memory import Semantic_memory, initialize_checkpointer
from.nodes import (
    amazon_policy,
    sales_expert,
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
    builder.add_node("sales_expert", sales_expert)
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
            "sales_expert": "sales_expert",
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
        "sales_expert",
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
    db_path=r'C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\chatbot\database\history_store\history.db'
    memory = initialize_checkpointer(db_path)
    graph = builder.compile(checkpointer=memory, store=Semantic_memory())  

    return graph

def call_ai_agentic(user_input: str, user_id: str = None):
    graph = build_graph()
    events = graph.stream(
    {"input": user_input},
    config = {"configurable": { "thread_id": user_id} },
    stream_mode="values",


    )
    chat_history = []
    for event in events:
        chat_history.append(event)

    response = event["output"].content if isinstance(event["output"], AIMessage) else event["output"]

    memory = Semantic_memory()  
    doc_id = f"msg_{id}_{len(chat_history)}"
    memory.add_texts(
        ids=[doc_id],
        texts=[f"Query: {user_input}\nResponse: {response}"],
        metadatas=[{"query": user_input, "response": response}]
    )

    return response