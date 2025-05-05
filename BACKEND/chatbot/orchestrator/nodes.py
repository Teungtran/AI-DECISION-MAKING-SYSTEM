import warnings
from ..agents.chat_pdf import RAG_QA
from ..agents.sql_agent import SQL_tool
from ..agents.expert import AI_Expert
from ..agents.Tavily_search_agent import Search_agent
warnings.filterwarnings('ignore')
import re
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from .state import AgentState, Router, open_ai_model,gg_model
from .memory import get_agent_memory,get_user_history
from BACKEND.chatbot.prompt_template import SYSTEM_PROMPT
import os
import warnings
warnings.filterwarnings('ignore')
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Initialize Azure OpenAI model

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
def sale_expert(state: AgentState):
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

#_____________REFLECTION & REFINEMENT NODES_______________
def recall_memory(state: AgentState):
    """Recall memory from JSON user history, rephrased naturally. Fallback to vector memory only if history is empty."""
    global LLM
    LLM = open_ai_model()
    user_id = state.get("user_id", "default_user")
    query = state["input"]

    if not query.strip():
        return {"output": AIMessage(content="Please provide a query to search in your memory.")}

    try:
        history = get_user_history(user_id, limit=5)  # JSON expected
        if history and isinstance(history, list):
            # Reformat JSON into a readable conversation
            history_text = "\n\n".join([
                f"User: {item.get('query', '')}\nAI: {item.get('response', '')}"
                for item in history
            ])
            print(f"Retrieved {len(history)} items from JSON history")

            # Ask LLM to rephrase this history into natural summary
            rephrase_prompt = f"""
            You are an assistant summarizing past conversations for context.

            Here are the past interactions between the user and the AI:

            {history_text}

            Please summarize or rephrase these interactions naturally as a conversation history:
"""
            rephrased_history = LLM.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=rephrase_prompt),
            ])

            memory_prompt = f"""
            You are a Supervisor AI Agent. The user has interacted before, and their past conversations are summarized below:

            {rephrased_history.content}

            Now the user asks a new question:
            User: {query}
            AI:
            """
            recall = LLM.invoke([
                SystemMessage(content=memory_prompt),
                HumanMessage(content=query),
            ])
            return {"output": AIMessage(content=recall.content)}
    except Exception as e:
        print(f"JSON History handling failed: {str(e)}")

#_______SUPERVISOR NODES______________
def supervisor_node(state: AgentState):
    global LLM
    LLM = open_ai_model()
    agents = ["amazon_policy", "sale_expert", "sql_agent", "search_engine", "recall_memory"]

    decision = LLM.with_structured_output(Router).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["input"]),
        ]
    )
    if not isinstance(decision, dict) or "next" not in decision or decision["next"] not in agents:
        decision["next"] = "recall_memory"
    print(f"Decision from LLM: {decision}")
    return {"decision": decision["next"]}
def route_decision(state: AgentState):
    decision = state.get("decision", None)
    
    if decision == "amazon_policy":
        return "amazon_policy"
    elif decision == "sale_expert":
        return "sale_expert"
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
    Uses simplified criteria to reduce unnecessary refinements.
    """
    global LLM
    LLM = open_ai_model()
    
    # Extract output content
    output = state.get("output", None)
    if isinstance(output, str):
        output_content = output
    elif hasattr(output, "content"):
        output_content = output.content
    else:
        output_content = str(output) if output else ""
    
    user_input = state.get("input", "")
    
    # Simplified prompt with higher threshold for triggering refinement
    reflection_prompt = f"""
    Analyze if this output effectively answers the user query:
    
    USER QUERY: "{user_input}"
    OUTPUT: {output_content}
    
    Evaluate only these critical issues:
    1. Does it directly answer the main question or request?
    2. Does it contain major factual errors or contradictions?
    3. Is the formatting severely broken, making it hard to read?
    4. If it contains SQL, does the query have critical errors?
    5. If it's an error message, mark it as "No refinement needed"
    
    INSTRUCTIONS:
    - Only recommend refinement for MAJOR issues that significantly impact usability
    - For minor style issues, prefer NO refinement
    - If SQL has minor optimization opportunities but works correctly, prefer NO refinement
    
    Format your response as:
    NEEDS_REFINEMENT: [YES or NO]
    FEEDBACK: [Brief feedback if major issues exist, otherwise "No refinement needed"]
    """
    
    reflection_response = LLM.invoke(
        [
            SystemMessage(content="You are a pragmatic AI output reviewer. Only flag major issues that truly impact usability."),
            HumanMessage(content=reflection_prompt),
        ]
    )
    
    # Extract response
    if isinstance(reflection_response, str):
        response_text = reflection_response
    else:
        response_text = reflection_response.content
    
    # Default to no refinement
    needs_refinement = False
    feedback = "No refinement needed"
    
    # Only proceed if explicitly marked YES
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
    print(f"Reflection analysis - Feedback: {feedback}")
    return {
        "reflection_feedback": feedback,
        "needs_refinement": needs_refinement
    }


def refined_node(state: AgentState):
    """
    Refine the output based on user input, reflection feedback, and search results.
    Uses Tavily search to find relevant context for improving responses.
    """
    global LLM
    LLM = open_ai_model()
    
    output = state.get("output", None)
    needs_refinement = state.get("needs_refinement", False)
    
    # Skip refinement if not needed
    if not needs_refinement or not output:
        return {"output": output}
    
    feedback = state.get("reflection_feedback", "")
    user_input = state.get("input", "")
    
    original_response = (
        output.content if hasattr(output, "content") else 
        (output if isinstance(output, str) else "")
    )

    # Summarize original response and identify flaws
    summarization_prompt = f"""
    ORIGINAL USER QUERY: {user_input}
    ORIGINAL RESPONSE: {original_response}
    IMPROVEMENT NEEDED: {feedback}

    Provide a concise summary of the original response and identify specific flaws 
    that need to be addressed based on the feedback.
    """
    
    summary_result = LLM.invoke(
        [
            SystemMessage(content="You are an expert at summarizing content and identifying areas for improvement."),
            HumanMessage(content=summarization_prompt),
        ]
    )
    
    summary = summary_result.content if hasattr(summary_result, "content") else str(summary_result)
    
    
    # Create search query based on user input and identified issues
    search_results = Search_agent(user_input)
    
    # Format search results
    formatted_results = "\n"
    for i, result in enumerate(search_results):
        formatted_results += f"Source {i+1}: {result.get('title', 'No title')}\n"
        formatted_results += f"URL: {result.get('url', 'No URL')}\n"
        formatted_results += f"Content: {result.get('content', 'No content')}\n\n"
    
    # Step 3: Refine the response using the search results
    refinement_prompt = f"""
    ORIGINAL USER QUERY: {user_input}
    ORIGINAL RESPONSE: {original_response}
    IMPROVEMENT NEEDED: {feedback}
    SUMMARY OF FLAWS: {summary}
    
    {formatted_results}

    INSTRUCTIONS:
    1. Use the search results to address the specific issues mentioned in the feedback
    2. Maintain the overall structure and purpose of the original response
    3. Update factual information based on the search results
    4. Don't add unrelated content or assumptions
    5. Provide the complete refined response
    """
    
    refined_response = LLM.invoke(
        [
            SystemMessage(content="You are a skilled editor who refines content using reliable information from search results."),
            HumanMessage(content=refinement_prompt),
        ]
    )
    
    refined_output = AIMessage(content=refined_response.content if hasattr(refined_response, "content") else str(refined_response))
    print(f"Refined output: {refined_output}")
    return {"output": refined_output}

def should_refine(state: AgentState):
    """
    Determine if the output needs reflection based on the source node.
    Only amazon_policy, sale_expert, and sql_agent should go through reflection.
    """
    decision = state.get("decision", None)
    
    # Only these specific nodes should go through reflection
    if decision in ["amazon_policy", "sale_expert", "sql_agent"]:
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

