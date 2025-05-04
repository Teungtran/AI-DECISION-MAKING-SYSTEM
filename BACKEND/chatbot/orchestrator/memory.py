import os
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import shutil
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

def get_agent_memory(user_id):
    chat_history_collection = get_mongo_connection()
    memory_data = chat_history_collection.find_one({"user_id": user_id})

    if not memory_data:
        print(f"No memory found for user {user_id}, initializing new memory.")
        return create_agent_memory(user_id)

    # Trả về một đối tượng MongoDBAtlasVectorSearch
    return create_agent_memory(user_id)

def save_conversation_to_memory(user_input, response, user_id):
    memory = get_agent_memory(user_id)
    if memory is None:
        print(f"Failed to initialize memory for user {user_id}")
        return

    chat_history_collection = get_mongo_connection()
    message_id = str(uuid.uuid4())

    try:
        chat_history_collection.update_one(
            {"user_id": user_id},
            {"$push": {"history": {"query": user_input, "response": response}}},
            upsert=True
        )

        memory.add_texts(
            texts=[f"Query: {user_input} → Response: {response}"],
            metadatas=[{"query": user_input, "response": response, "user_id": user_id}],
            ids=[message_id]  
        )

        print(f"Successfully saved conversation for user {user_id}.")
    except Exception as e:
        print(f"Error saving for user {user_id}: {str(e)}")
        
        
def initialize_checkpointer(db_path):
    """Ensures the checkpointer database is not corrupted by recreating it if necessary."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except sqlite3.DatabaseError:
        shutil.rmtree(db_path, ignore_errors=True)  
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)