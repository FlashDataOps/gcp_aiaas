import asyncio
import streamlit as st
import langchain_utils as lu
import pandas as pd
import json
from streamlit_navigation_bar import st_navbar
import time
import os
import aux_functions as af
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import traceback
from gtts import gTTS
import base64
import datetime
import pyperclip

def add_copy_button(text, key_suffix=""):
    """Adds a copy button for the full text with a unique key."""
    col1, col2 = st.columns([9, 1])
    
    with col1:
        st.empty()  # Remove this line to prevent duplicate text
    
    with col2:
        if st.button("📋", key=f"copy_full_text_{key_suffix}"):
            pyperclip.copy(text)
            st.toast('Text copied to clipboard!', icon='📋')

def render_or_update_model_info(model_name):
    """Renders or updates the model information on the webpage."""
    with open("./design/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

def reset_chat_history():
    """Resets the chat history by clearing the 'messages' list in the session state."""
    if "messages" in st.session_state:
        st.session_state.messages = []

model_options = ["llama-3.1-70b-versatile","llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]
max_tokens = {
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "gemma-7b-it": 8192,
    "gemini-1.5-flash-002": 128000,
    "gemini-1.5-pro-002": 128000,
    "llama-3.1-70b-versatile":8_000
}

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = model_options[1]
    st.session_state.temperature = 0
    st.session_state.max_tokens = max_tokens[st.session_state.model]
    st.session_state.input_audio = 1

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.initial_message_displayed = False
    st.session_state.sql_messages = []
    st.session_state.show_success_audio = False

# Sidebar configuration
with st.sidebar:
    st.title("Model Configuration")    
  
    # List files in the 'db' folder
    carpeta_db = 'db' 
    try:
        dbs = os.listdir(carpeta_db)
        archivos_db = [f for f in dbs if os.path.isfile(os.path.join(carpeta_db, f))]
    except FileNotFoundError:
        archivos_db = []
        st.error(f"The folder '{carpeta_db}' does not exist.")
    
    if archivos_db:
        archivo_db_seleccionado = st.selectbox("Select a database:", archivos_db)
        af.db_connection.db_name = archivo_db_seleccionado
        
    # Select model
    st.session_state.model = st.selectbox(
        "Choose a model:",
        model_options,
        index=1
    )

    # Select temperature
    st.session_state.temperature = st.slider('Select a level of creativity:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    st.session_state.max_tokens = 8000

    clear_chat_column, record_audio_column = st.columns([1, 1])
    # Reset chat history button
    if st.button(":broom: Clear chat", use_container_width=True):
        reset_chat_history()
  
# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Use st.write instead of st.markdown to avoid duplicate display
            st.write(message["content"])
            add_copy_button(message["content"], key_suffix=str(idx))
        else:
            st.markdown(message["content"])            
        
        if "figure_p" in message["aux"].keys():
            for figure in message["aux"]["figure_p"]:
                st.plotly_chart(figure)
                      
# Accept user input
prompt = st.chat_input("How can I help you?", key="user_input")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Generate response
        response = lu.invoke_chain(
            question=prompt,
            messages=st.session_state.messages[1:],
            sql_messages=st.session_state.sql_messages[1:],
            model_name=st.session_state.model,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
        
        # Stream the response with a single copy button
        full_response = st.empty()
        streamed_response = ""
        for chunk in response:
            streamed_response += chunk
            full_response.markdown(streamed_response)
        
        # Clear the empty placeholder and display the full response
        full_response.markdown(streamed_response)
        
        # Add copy button to the final response
        add_copy_button(streamed_response, key_suffix=str(len(st.session_state.messages)))
        
        # Prepare auxiliary data
        aux_v2 = lu.invoke_chain.aux
                
        # Handle figures
        if "figure_p" in aux_v2.keys():
            with st.spinner("Generating plots ..."):
                for figure in aux_v2["figure_p"]:
                    st.plotly_chart(figure)
        
        # Update session state
        st.session_state.messages.extend([
            {"role": "user", "content": prompt, "aux": {}},
            {
                "role": "assistant", 
                "content": streamed_response, 
                "aux": aux_v2
            }
        ])