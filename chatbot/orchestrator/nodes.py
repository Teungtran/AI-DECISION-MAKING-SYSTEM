import warnings
from ..agents.chat_pdf import RAG_QA
from ..agents.sql_agent import SQL_tool
from ..agents.expert import AI_Expert
from ..agents.Tavily_search_agent import Search_agent
warnings.filterwarnings('ignore')
import re
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from .state import AgentState, Router, open_ai_model,gg_model
from .memory import Semantic_memory
from ..prompt_template import SYSTEM_PROMPT
#_______AGENTS NODES______________
LLM = None
def search_engine(state: AgentState):
    try:
        query = state["input"]  
        result = Search_agent(query)  

        return  {"output": result}
    except Exception as e:
        return {"output": AIMessage(content=f"Error in  search: {str(e)}")}

def amazon_policy(state: AgentState):
    try:
        query = state["input"]  
        result = RAG_QA(query)  
        return {"output": result}
    except Exception as e:
        return {"output": f"Error retrieving policy information: {str(e)}"}
def sales_expert(state: AgentState):
    try:
        query = state["input"]  
        result = AI_Expert(query)  
        return {"output": result}
    except Exception as e:
        return {"output": f"Error retrieving expert information: {str(e)}"}
    
def sql_agent(state: AgentState):
    try:
        query = state["input"]  
        result = SQL_tool(query)

        return {"output": result}
    except Exception as e:
        return {"output": AIMessage(content=f"Error accessing database: {str(e)}")}

def recall_memory(state: AgentState):
    """Handles memory recall for previous interactions."""
    global LLM
    LLM = open_ai_model()
    memory_prompt = """ 
    You are a Supervisor handling {agents}. You also have to handle questions about previous interactions, chat history, and follow-up questions.
    If the user asks about previous conversations, retrieve relevant past interactions and provide an answer.
    If the user asks a follow-up question, recall the past response and use it as context before deciding on the next tool to use.
    """
    memory = Semantic_memory()
    past_interactions = memory.search(state["input"], search_type='similarity',k=1)

    history_context = "\n".join([
        f"{item.metadata['query']} → {item.metadata['response']}" for item in past_interactions
    ])

    if history_context:
        prompt_with_memory = f"{memory_prompt}\n\nPrevious Conversations:\n{history_context}\n\nUser: {state['input']}"
    else:
        prompt_with_memory = f"{memory_prompt}\n\nUser: {state['input']}"

    recall = LLM.invoke([
        SystemMessage(content=prompt_with_memory),
        HumanMessage(content=state["input"]),
    ])

    return {"output": AIMessage(content=recall.content)}

#_______SUPERVISOR NODES______________
def supervisor_node(state: AgentState):
    global LLM
    LLM = open_ai_model()
    decision = LLM.with_structured_output(Router).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["input"]),
        ]
    )


    if not isinstance(decision, dict) or "next" not in decision or decision["next"] is None:
        return {"decision": "recall_memory"}  

    return {"decision": decision["next"]}
def route_decision(state: AgentState):
    decision = state.get("decision", None)
    
    if decision == "amazon_policy":
        return "amazon_policy"
    elif decision == "sales_expert":
        return "sales_expert"
    elif decision == "sql_agent":
        return "sql_agent"
    elif decision == "search_engine":
        return "search_engine"
    elif decision == "recall_memory":
        return "recall_memory"
    else:
        print(f"Unexpected decision value: {decision}")
        return "recall_memory"
    
#_____________REFLECTION & REFINEMENT NODES_______________
def reflection_node(state: AgentState):
    """
    Analyze the output from previous node and determine if refinement is needed.
    Provides specific feedback if improvement is required.
    """
    global LLM
    LLM = open_ai_model()
    output = state.get("output", None)
    if isinstance(output, str):
        output_content = output
    elif hasattr(output, "content"):
        output_content = output.content
    else:
        output_content = str(output) if output else ""
    user_input = state.get("input", "")
    
    reflection_prompt = f"""
    Analyze the following output to determine if it correctly and effectively answers the user query:
    
    USER QUERY: "{user_input}"
    
    OUTPUT: {output_content}
    
    First, decide if this output needs improvement by evaluating:
    1. It should not contain any complicated jargon or terms that are not directly related to the user's query.
    2. Does it directly answer the user's query?
    3. If it contains SQL queries, are they optimized and correct?
    4. Is it clear, concise and not too complicated for the user?
    
    INSTRUCTIONS:
    - First, determine YES or NO if refinement is needed
    - If YES, provide specific, actionable feedback on what to improve
    - If SQL queries exist, check for proper indexing, join optimization, and query structure
    - Focus on completeness, clarity, and relevance to the original query
    
    Format your response as:
    NEEDS_REFINEMENT: [YES or NO]
    FEEDBACK: [Your detailed feedback if refinement is needed, otherwise "No refinement needed"]
    """
    
    reflection_response = LLM.invoke(
        [
            SystemMessage(content="You are a critical AI output reviewer. Your job is to determine if content needs improvement and provide specific, actionable feedback."),
            HumanMessage(content=reflection_prompt),
        ]
    )
    
    if isinstance(reflection_response, str):
        response_text = reflection_response
    else:
        response_text = reflection_response.content
    
    needs_refinement = False
    feedback = "No refinement needed"
    
    if "NEEDS_REFINEMENT: YES" in response_text.upper():
        needs_refinement = True
        
        # Extract feedback section
        feedback_pattern = r"FEEDBACK:\s*(.*?)(?=$|NEEDS_REFINEMENT:)"
        feedback_match = re.search(feedback_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            parts = response_text.split("FEEDBACK:", 1)
            if len(parts) > 1:
                feedback = parts[1].strip()
    
    print(f"Reflection analysis - Needs refinement: {needs_refinement}")
    print(f"Feedback: {feedback}")
    
    return {
        "reflection_feedback": feedback,
        "needs_refinement": needs_refinement
    }
def refined_node(state: AgentState):
    """
    Refine the output based on user input and reflection feedback
    """
    global LLM
    LLM = open_ai_model()
    output = state.get("output", None)
    feedback = state.get("reflection_feedback", "")
    user_input = state.get("input", "")
    needs_refinement = state.get("needs_refinement", False)
    
    if not needs_refinement or not output:
        return {"output": output}
    
    original_response = (
        output.content if hasattr(output, "content") else (output if isinstance(output, str) else "")
    )

    refinement_prompt = f"""
    ORIGINAL USER QUERY: {user_input}

    ORIGINAL RESPONSE: {original_response}

    IMPROVEMENT FEEDBACK: {feedback}

    INSTRUCTIONS:
    You must refine the original response strictly based on the feedback above. Do not add any new facts, details, or assumptions that are not present in the original response or the feedback.

    1. Only rewrite the response using the original content and the feedback provided.
    2. Do NOT introduce new content or change the meaning of the original answer.
    3. Ensure the response is improved for clarity, accuracy, and completeness only within the scope of the original answer.
    4. Do NOT add explanations about the refinement process.
    5. If the feedback involves SQL optimization, only modify the query logic without adding new unrelated queries or logic.
    6. Provide only the final refined response.
    """
    
    refined_response = LLM.invoke(
        [
            SystemMessage(content="You are an expert at refining AI outputs based on feedback. Your job is to create a clear, concise and accurate response that addresses all the issues identified in the feedback."),
            HumanMessage(content=refinement_prompt),
        ]
    )
    
    refined_output = AIMessage(content=refined_response.content if hasattr(refined_response, "content") else str(refined_response))
    
    return {"output": refined_output}

def should_refine(state: AgentState):
    """
    Determine if the output needs reflection based on the source node.
    Only amazon_policy, sales_expert, and sql_agent should go through reflection.
    """
    decision = state.get("decision", None)
    
    # Only these specific nodes should go through reflection
    if decision in ["amazon_policy", "sales_expert", "sql_agent"]:
        return "check_reflection"
    else:
        # All other nodes skip reflection entirely
        return "skip_refinement"
def needs_refinement_check(state: AgentState):
    """
    Based on the reflection node's determination, either proceed to refinement
    or skip directly to END
    """
    if state.get("needs_refinement", False):
        return "refine"  
    else:
        return "no_refine"  

