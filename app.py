import streamlit as st
import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import TranscriptsDisabled
from langchain_core.messages import HumanMessage, AIMessage

# Import your modules
from modules.ingestion import get_video_chunks
from modules.vector_store import get_vector_store, check_index_exists
from modules.generation import get_agent

# Load environment variables
load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")
st.title("📺 YouTube Video Chatbot")

# --- DATABASE LOGIC: Saving & Loading Chats ---
CHAT_DB_FILE = "chat_history.json"

def load_chat_db():
    """Reads the saved chats from the hard drive."""
    if os.path.exists(CHAT_DB_FILE):
        with open(CHAT_DB_FILE, "r") as f:
            return json.load(f)
    return {} # Return an empty dictionary if the file doesn't exist yet

def save_chat_db(chats_dict):
    """Writes the current chats to the hard drive."""
    with open(CHAT_DB_FILE, "w") as f:
        json.dump(chats_dict, f)

# --- SESSION STATE: The Dictionary of Chats ---
if "chats" not in st.session_state:
    # Instead of an empty {}, load from the JSON file!
    st.session_state.chats = load_chat_db()
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR: Session Management ---
with st.sidebar:
    st.header("1. Start a New Chat")
    new_video_id = st.text_input("Enter YouTube Video ID:")

    if st.button("Process & Start Chat"):
        if new_video_id:
            st.session_state.current_video = new_video_id
            if new_video_id not in st.session_state.chats:
                st.session_state.chats[new_video_id] = []

            with st.spinner("Preparing video database..."):
                try:
                    if check_index_exists(new_video_id):
                        st.session_state.vector_store = get_vector_store(new_video_id)
                        st.success("Loaded instantly from local storage!")
                    else:
                        chunks = get_video_chunks(new_video_id)
                        st.session_state.vector_store = get_vector_store(new_video_id, chunks=chunks)
                        st.success("Video processed and saved to local storage!")
                except TranscriptsDisabled:
                    st.error("No captions available for this video.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.divider()

    st.header("2. Previous Chats")
    if st.session_state.chats:
        video_options = list(st.session_state.chats.keys())
        current_index = 0
        if st.session_state.current_video in video_options:
            current_index = video_options.index(st.session_state.current_video)

        selected_video = st.selectbox("Select a video to resume:", options=video_options, index=current_index)

        if st.button("Resume Chat"):
            st.session_state.current_video = selected_video
            with st.spinner("Loading previous session..."):
                st.session_state.vector_store = get_vector_store(selected_video)
            st.success(f"Switched back to {selected_video}")

# --- CHAT INTERFACE ---
if st.session_state.current_video and st.session_state.vector_store:

    # --- The Active Chat UI ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header(f"Chatting about: {st.session_state.current_video}")
    with col2:
        youtube_url = f"https://www.youtube.com/watch?v={st.session_state.current_video}"
        st.video(youtube_url)

    st.divider()

    # Display the chat history with CUSTOM AVATARS
    current_history = st.session_state.chats[st.session_state.current_video]
    for message in current_history:
        # Define avatars: '🧑‍💻' for user, '🤖' for the AI assistant
        avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    # Handle user input
    if user_question := st.chat_input("Ask a question about this video..."):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_question)
        current_history.append({"role": "user", "content": user_question})

        formatted_history = []
        for msg in current_history[:-1]:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            else:
                formatted_history.append(AIMessage(content=msg["content"]))

        with st.spinner("Agent is thinking..."):
            agent = get_agent(st.session_state.vector_store)
            response = agent(user_question, formatted_history)

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

            # Update the current history
            current_history.append({"role": "assistant", "content": response})

            # Save it back to Streamlit's memory
            st.session_state.chats[st.session_state.current_video] = current_history

            # THE NEW LINE: Save the entire master dictionary to the hard drive!
            save_chat_db(st.session_state.chats)

else:
    # --- The "Empty State" Hero Dashboard ---
    # This shows up ONLY when no video is selected, making the app look professional immediately.
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds some breathing room

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👋 Welcome to your AI Video Assistant!")
        st.markdown("""
        ### How to use this tool:
        1. **Grab a Link:** Copy any YouTube Video ID (the letters/numbers after `v=` in the URL).
        2. **Process:** Paste it into the sidebar and click 'Process & Start Chat'.
        3. **Ask Anything:** The AI will read the transcript, summarize the video, or even search the web if the video doesn't have the answer!

        *Start by pasting a video ID in the sidebar to the left!*
        """)