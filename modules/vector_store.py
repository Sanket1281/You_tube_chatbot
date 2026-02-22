from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(chunks):
    """Embeds document chunks and creates a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store