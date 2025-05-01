import os
import warnings
warnings.filterwarnings('ignore')
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_chroma import Chroma
import chromadb
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def Semantic_memory():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )
    
    semantic_db = Chroma(
    client=chromadb.PersistentClient(path=r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\chatbot\database\orchestrator_memory"),
    collection_name='langraph_memory',
    embedding_function=embedding,
)
    return semantic_db

def initialize_checkpointer(db_path):
    """Ensures the checkpointer database is not corrupted by recreating it if necessary."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except sqlite3.DatabaseError:
        shutil.rmtree(db_path, ignore_errors=True)  
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)