import os
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import shutil
import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
import uuid
from .state import AgentState 
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URL = os.getenv("MONGO_URL")
def mongo_db():
    return MongoClient(MONGO_URL, server_api=ServerApi('1'))
def get_mongo_connection():
    client = mongo_db()
    db = client["AI_MEMORY"]
    return db["chat_history"]

def create_agent_memory(user_id):
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        chat_history_collection = get_mongo_connection()

        semantic_db = MongoDBAtlasVectorSearch(
            collection=chat_history_collection,
            embedding=embedding,
        )
        
        # Lưu vector search vào bộ nhớ MongoDB
        chat_history_collection.update_one(
            {"user_id": user_id},
            {"$setOnInsert": {"user_id": user_id, "history": []}},
            upsert=True
        )

        return semantic_db
    except Exception as e:
        print(f"Error initializing Semantic Memory for user {user_id}: {str(e)}")
        return None

def get_agent_memory(state: AgentState):
    """Create or get memory for a user with proper configuration"""
    try:
        user_id = state.get("user_id", "default_user")
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        
        # Get MongoDB collection
        chat_history_collection = get_mongo_connection()
        
        # Ensure user document exists
        chat_history_collection.update_one(
            {"user_id": user_id},
            {"$setOnInsert": {
                "user_id": user_id, 
                "history": [], 
                "created_at": datetime.datetime.now()
            }},
            upsert=True
        )
        
        # Setup vector search with proper configuration
        semantic_db = MongoDBAtlasVectorSearch(
            collection=chat_history_collection,
            embedding=embedding,
            relevance_score_fn="cosine"

        )
        
        return semantic_db
        
    except Exception as e:
        print(f"Error initializing memory for user {user_id}: {str(e)}")
        return None

def save_conversation_to_memory(user_input, response, user_id):
    """Save a conversation to both regular MongoDB and vector store"""
    try:
        chat_history_collection = get_mongo_connection()
        message_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        # Format the text for vector search
        text_content = f"Query: {user_input} → Response: {response}"
        
        # 1. First save to regular MongoDB history array
        chat_history_collection.update_one(
            {"user_id": user_id},
            {"$push": {"history": {
                "message_id": message_id,
                "query": user_input, 
                "response": response, 
                "created_at": timestamp
            }}},
            upsert=True
        )
        
        try:
            memory = create_agent_memory(user_id)
            if memory:
                memory.add_texts(
                    texts=[text_content],
                    metadatas=[{
                        "message_id": message_id,
                        "query": user_input, 
                        "response": response, 
                        "user_id": user_id,
                        "created_at": timestamp.isoformat()
                    }],
                    ids=[message_id]
                )
                print(f"Conversation saved to vector store for user {user_id}")
            else:
                print(f"Vector store unavailable for user {user_id}")
        except Exception as e:
            print(f"Vector store error: {str(e)}")
            
        return True
            
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")
        return False

def get_user_history(user_id, limit=5):
    """Get user history directly from MongoDB (fallback method)"""
    try:
        chat_history_collection = get_mongo_connection()
        user_doc = chat_history_collection.find_one({"user_id": user_id})
        
        if not user_doc or "history" not in user_doc:
            return []
            
        history = user_doc["history"]
        history.reverse() 
        return history[:limit]
        
    except Exception as e:
        print(f"Error retrieving history: {str(e)}")
        return []
        
def initialize_checkpointer(db_path):
    """Ensures the checkpointer database is not corrupted by recreating it if necessary."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except sqlite3.DatabaseError:
        shutil.rmtree(db_path, ignore_errors=True)  
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
