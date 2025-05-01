# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import chromadb 
from ..prompt_template import EXPERT_PROMPT
from chromadb.errors import NotFoundError

from langchain.retrievers import EnsembleRetriever
from langchain_openai import AzureChatOpenAI
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
#_______________________Process document(chunking, embedding, vector store)________________
local_path = [r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\marketing&sales\45101-111548-2-PB.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\marketing&sales\A_winning_sales_strategy-1.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\marketing&sales\FullPaper.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\marketing&sales\jtaer-16-00164.pdf",]

# Define Chroma constants
EXPERT_CHROMA_PATH = os.getenv("EXPERT_CHROMA_PATH")
EXPERT_COLLECTION_NAME = os.getenv("EXPERT_COLLECTION_NAME")
def process_document():
    genai_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    
    # Connect to Chroma persistent client
    client = chromadb.PersistentClient(path=EXPERT_CHROMA_PATH)

    try:
        collection = client.get_collection(EXPERT_COLLECTION_NAME)
        if collection.count() > 0:
            print(f"Loaded existing collection with {collection.count()} documents.")
            return Chroma(
                client=client,
                EXPERT_COLLECTION_NAME=EXPERT_COLLECTION_NAME,
                embedding_function=genai_embedding
            )
    except NotFoundError:
        pass  # Collection doesn't exist; we'll create it below

    # If not found or empty, process and store documents
    all_documents = []
    for path in local_path:
        loader = PyPDFLoader(file_path=path)  
        doc = loader.load()
        all_documents.extend(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(all_documents)

    documents = [
        Document(page_content=chunk.page_content, metadata={**chunk.metadata, 'chunk_id': str(i),
                                                            'page': chunk.metadata['page']}) 
        for i, chunk in enumerate(chunks)
    ]
    ids = [str(uuid4()) for _ in range(len(documents))]

    # Store in vector DB
    vectorstore = Chroma(
        client=client,
        EXPERT_COLLECTION_NAME=EXPERT_COLLECTION_NAME,
        embedding_function=genai_embedding
    )
    vectorstore.add_documents(documents=documents, ids=ids)

    print(f"Created new collection with {len(documents)} documents.")
    return vectorstore
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
    vectorstore = process_document()
    response = RAG_BOT(vectorstore, user_input)
    return response
