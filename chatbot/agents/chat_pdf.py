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
from ..prompt_template import RAG_PROMPT
from chromadb.errors import NotFoundError
from langchain.retrievers import EnsembleRetriever
from langchain_openai import AzureChatOpenAI
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
#_______________________Process document(chunking, embedding, vector store)________________
local_path = [r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\amazon_policy\pot_1047_customer_service_policy.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\amazon_policy\Selling_Policies_and_Seller_Code_of_Conduct_SG_new_version_clean_PDF.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\amazon_policy\Beginners-Guide-to-Selling-on-Amazon.pdf",
            r"C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\Data\amazon_policy\Conduct-and-Performance-Policy 9.21.pdf"]

AMAZON_CHROMA_PATH = os.getenv("AMAZON_CHROMA_PATH")
AMAZON_COLLECTION_NAME = os.getenv("AMAZON_COLLECTION_NAME")
def process_document():
    genai_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    
    client = chromadb.PersistentClient(path=AMAZON_CHROMA_PATH)

    try:
        collection = client.get_collection(AMAZON_COLLECTION_NAME)
        if collection.count() > 0:
            print(f"Loaded existing collection with {collection.count()} documents.")
            return Chroma(
                client=client,
                AMAZON_COLLECTION_NAME=AMAZON_COLLECTION_NAME,
                embedding_function=genai_embedding
            )
    except NotFoundError:
        pass 

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
        AMAZON_COLLECTION_NAME=AMAZON_COLLECTION_NAME,
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
    [("system", RAG_PROMPT),("human", "{input}")])
    
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
def RAG_QA(user_input):
    vectorstore = process_document()
    response = RAG_BOT(vectorstore, user_input)
    return response
