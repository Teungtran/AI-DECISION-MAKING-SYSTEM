
import warnings
warnings.filterwarnings('ignore')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from BACKEND.chatbot.prompt_template import RAG_PROMPT
from langchain.retrievers import EnsembleRetriever
from langchain_openai import AzureChatOpenAI
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_AMAZON = os.getenv("QDRANT_AMAZON")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
# Global vectorstore instance
VECTORSTORE = None

def setup_vectorstore():
    """
    Connects to an existing Qdrant vectorstore collection using GoogleGenerativeAIEmbeddings.

    Returns:
        Qdrant: An instance of the connected vectorstore.
    """
    global VECTORSTORE

    if VECTORSTORE is not None:
        return VECTORSTORE

    genai_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GOOGLE_API_KEY
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # Connect to the collection
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_AMAZON,
        embeddings=genai_embedding
    )

    VECTORSTORE = vectorstore
    return VECTORSTORE

#_______________________rag_________________________________________
def RAG_BOT(vectorstore,question):
    genllm = AzureChatOpenAI(openai_api_key=OPENAI_API_KEY,
                            openai_api_version="2024-08-01-preview",
                            model ="gpt-4o-mini",
                            openai_api_base=AZURE_ENDPOINT, temperature=0)    
    prompt = ChatPromptTemplate.from_messages(
    [("system", RAG_PROMPT),("human", "{input}")])
    
    retriever = vectorstore.as_retriever(search_type="mmr",  search_kwargs={"k": 3,"fetch_k": 5,"lambda_mult": 0.5})
    keyword_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever, keyword_retriever],
        weights=[0.5, 0.5] 
    )
    question_answer_chain = create_stuff_documents_chain(
            genllm,
            prompt,
            document_variable_name="context",
            document_separator="\n\n"
    )    
    rag_qa  = create_retrieval_chain(ensemble_retriever, question_answer_chain)
    rag_response = rag_qa.invoke({"input": question, "metadata": {"requires_reasoning": True}}
)
    return rag_response["answer"]
#_______________QA___________________________________________________
def RAG_QA(user_input):
    vectorstore = setup_vectorstore()
    response = RAG_BOT(vectorstore, user_input)
    return response
