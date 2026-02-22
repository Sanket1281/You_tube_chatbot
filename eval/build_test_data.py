import sys
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Tell Python to look in the parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.ingestion import get_video_chunks
from modules.vector_store import create_vector_store
from modules.generation import get_rag_chain

# 1. THE HUMAN PART: You define the questions and perfect answers
questions = [
    "What car is being discussed in this video?",
    "Does the Mustang GTD have a hybrid engine?"
]
ground_truths = [
    "The car being discussed is the new 2025 Mustang GTD",
    "NO, the Mustang has a supercharged 5.2 liter V8 engine pumping out 800 HP."
]

# 2. Setup your bot
video_id = "e80GvN2OcTM"  # Use the video ID these questions belong to
chunks = get_video_chunks(video_id)
vector_store = create_vector_store(chunks)
rag_chain = get_rag_chain(vector_store)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3. THE CHATBOT PART: Let the bot generate the contexts and answers
contexts_list = []
answers_list = []

for q in questions:
    print(f"Asking bot: {q}")

    # Get the answer from your Gemini chain
    bot_answer = rag_chain.invoke(q)
    answers_list.append(bot_answer)

    # Get the contexts from your FAISS retriever
    retrieved_docs = retriever.invoke(q)
    doc_texts = [doc.page_content for doc in retrieved_docs]
    contexts_list.append(doc_texts)

# 4. Combine it all into the final dictionary!
data = {
    "question": questions,
    "ground_truth": ground_truths,
    "contexts": contexts_list,
    "answer": answers_list
}

print("\n--- Your final Ragas Data ---")
print(data)