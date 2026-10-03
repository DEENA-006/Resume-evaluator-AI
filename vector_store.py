import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


def get_embedding_model():
    """Local embedding model with fallback to Google embeddings for cloud resilience."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = os.getenv("GOOGLE_API_KEY")
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        except Exception:
            raise e


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Document]:
    """Splits documents into coherent chunks without splitting words."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def build_faiss_vectorstore(chunks: List[Document]) -> FAISS:
    """Generates embeddings for chunks and indexes them into an in-memory FAISS store."""
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_relevant_chunks(vectorstore: FAISS, query: str, top_k: int = 4) -> List[Document]:
    """Performs semantic search to find top_k chunks most relevant to the Job Description."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)