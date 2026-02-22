import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import TranscriptsDisabled

# Import your new modules
from modules.ingestion import get_video_chunks
from modules.vector_store import create_vector_store
from modules.generation import get_rag_chain

# Load environment variables
load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"

st.title("📺 YouTube Video Chatbot")
st.markdown("Chat with the transcript of any YouTube video.")

# --- SIDEBAR: Video Processing ---
with st.sidebar:
    st.header("1. Load Video")
    video_id = st.text_input("Enter YouTube Video ID (e.g., iUh-8yjycHU):")
    process_button = st.button("Process Video")

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- INDEXING LOGIC ---
if process_button and video_id:
    with st.spinner("Fetching transcript and building index..."):
        try:
            # 1. Ingest Data
            chunks = get_video_chunks(video_id)

            # 2. Build Vector Store
            st.session_state.vector_store = create_vector_store(chunks)

            st.success("Video processed successfully! You can now ask questions.")
        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- CHAT INTERFACE ---
if st.session_state.vector_store:
    st.header("2. Ask Questions")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_question := st.chat_input("What would you like to know about this video?"):
        # Display user message
        st.chat_message("user").markdown(user_question)
        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Generate and display response using the generation module
        with st.spinner("Thinking..."):
            # 3. Get the configured chain
            rag_chain = get_rag_chain(st.session_state.vector_store)

            # 4. Invoke it
            response = rag_chain.invoke(user_question)

            with st.chat_message("assistant"):
                st.markdown(response)

            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})