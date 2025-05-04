import warnings
warnings.filterwarnings('ignore')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from BACKEND.chatbot.prompt_template import EXPERT_PROMPT
from langchain.retrievers import EnsembleRetriever
from langchain_openai import AzureChatOpenAI
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_EXPERT = os.getenv("QDRANT_EXPERT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
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

    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_EXPERT,
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
    [("system", EXPERT_PROMPT),("human", "{input}")])
    
    retriever = vectorstore.as_retriever(search_type="mmr",  search_kwargs={"k": 5,"fetch_k": 10,"lambda_mult": 0.7})
    keyword_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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
def AI_Expert(user_input):
    vectorstore = setup_vectorstore()
    response = RAG_BOT(vectorstore, user_input)
    return response
