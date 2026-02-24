import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Create a master directory name to hold all our saved databases
INDEX_DIR = "faiss_indexes"


def check_index_exists(video_id: str) -> bool:
    """Checks if a vector database for this video already exists on disk."""
    folder_path = os.path.join(INDEX_DIR, video_id)
    return os.path.exists(folder_path)


def get_vector_store(video_id: str, chunks=None):
    """Loads an existing index, or builds and saves a new one."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    folder_path = os.path.join(INDEX_DIR, video_id)

    # SCENARIO A: The database already exists on our hard drive!
    if check_index_exists(video_id):
        # Note: LangChain requires allow_dangerous_deserialization=True to load local files
        vector_store = FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store

    # SCENARIO B: We are processing this video for the very first time.
    if chunks is None:
        raise ValueError("No chunks provided to build a new vector store.")

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save it locally so we never have to embed this video again!
    vector_store.save_local(folder_path)

    return vector_store