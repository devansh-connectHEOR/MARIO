from scripts.RAG import RAG
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import uuid
from itertools import chain

import os
from dotenv import load_dotenv
load_dotenv()

# --- Session State Initialization ---
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Store conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Agent Setup ---
if "app" not in st.session_state:
    cwd = Path(r"C:\Users\Public\Documents\MARIO\rag_working_dir")
    rag = RAG(None, cwd)

    st.session_state.app = rag

# --- Streamlit Framework ---
st.title('MARIO')

st.caption(f"Conversation ID: `{st.session_state.thread_id}`")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Handle user input
user_input = st.chat_input(
    "Ask a question")

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
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response_content})
