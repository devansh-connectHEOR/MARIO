from scripts.RAG import RAG
from scripts.utilities import create_sqldb, list_sessions, add_session, rename_session
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

import streamlit as st
import uuid
from itertools import chain
import os

load_dotenv()

# --- Session State Initialization ---
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'cwd' not in st.session_state:
    st.session_state.cwd = Path.cwd() / "rag_working_dir"

if 'chat_db' not in st.session_state:
    # Start with the initial thread
    st.session_state.chat_db = create_sqldb("App_1_session.db", st.session_state.cwd)
    add_session(st.session_state.chat_db, st.session_state.thread_id)

if "app" not in st.session_state:
    rag = RAG(None, st.session_state.cwd, checkpointer=SqliteSaver(st.session_state.chat_db))
    st.session_state.app = rag

if 'messages' not in st.session_state:
    st.session_state.messages = []
# --- Sidebar Framework ---
with st.sidebar:
    st.title("Chat Management")
    
    # 1. Create a New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        # Add the new chat to the top of the list
        st.toast(add_session(st.session_state.chat_db, st.session_state.thread_id))
        # Fetch history for the new thread (which should be empty)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 2. Rename the Current Chat
    # Find the title of the current thread_id
    current_title = next(
        (title for sid, title in list_sessions(st.session_state.chat_db) if sid == st.session_state.thread_id), 
        "New Chat"
    )
    
    new_title = st.text_input("Rename Current Chat", value=current_title)
    if st.button("Save Name", use_container_width=True):
        if new_title != current_title:
            # Rebuild the list of tuples with the updated name for the active thread
            st.toast(rename_session(st.session_state.chat_db, st.session_state.thread_id, new_title))
            st.rerun()

    st.divider()

    # 3. Select a Chat Session
    st.subheader("Previous Chats")
    for sid, title in list_sessions(st.session_state.chat_db):
        is_active = (sid == st.session_state.thread_id)
        # Visually indicate which chat is currently active
        button_label = f"🟢 {title}" if is_active else f"💬 {title}"
        
        # Use the session_id as the unique key for the button
        if st.button(button_label, key=sid, use_container_width=True):
            if not is_active:
                # Switch thread ID and load its history
                st.session_state.thread_id = sid
                st.session_state.messages = st.session_state.app.get_msg_history(thread_id=sid)
                st.rerun()


# --- Main UI Framework ---
st.title('MARIO')

st.caption(f"Conversation ID: `{st.session_state.thread_id}`")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask a question")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the agent
    agent_output = st.session_state.app.analyze_stream(
        user_input, thread_id=st.session_state.thread_id)
    
    first_token = ""
    with st.spinner("Thinking"):
        try:
            first_token = next(agent_output)
        except StopIteration:
            st.warning("The agent did not return any response.")

    output_stream = chain([first_token], agent_output)

    # Display agent response and add to history
    final_response_content = ""
    with st.chat_message("assistant"):
        final_response_content = st.write_stream(output_stream)
        
    # Update messages from the agent's memory backend
    st.session_state.messages = st.session_state.app.get_msg_history(thread_id=st.session_state.thread_id)