import streamlit as st
import langchain_foundations as lu
import pandas as pd
import json
from streamlit_navigation_bar import st_navbar
import time
import os
import aux_functions as af
from streamlit_pills import pills
def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/foundations/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/foundations/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

# Reset chat history
def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
    if "messages" in st.session_state:
        st.session_state.messages_foundations = []

model_options = ["llama3-70b-8192"]
max_tokens = {
    "llama3-70b-8192": 8192,
#    "llama3-8b-8192": 8192,
#    "mixtral-8x7b-32768": 32768,
#   "gemma-7b-it": 8192,
#    "gemini-1.5-flash-002": 128000,
#    "gemini-1.5-pro-002": 128000
}

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = 8192

# Initialize chat history
if "messages_foundations" not in st.session_state:
    st.session_state.messages_foundations = []
    st.session_state.sql_messages = []
    st.session_state.prompt = None
    
with st.sidebar:
    st.title("Configuración de modelo")

    # Select model
    st.session_state.model = st.selectbox(
        "Elige un modelo:",
        model_options,
        index=0
    )

    # Select temperature
    st.session_state.temperature = st.slider('Selecciona una temperatura:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]

    st.session_state.max_tokens = st.number_input('Seleccione un máximo de tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)

    # Reset chat history button
    if st.button("Vaciar Chat"):
        reset_chat_history()
    
# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for message in st.session_state.messages_foundations:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "figure" in message["aux"].keys() and len(message["aux"]["figure"]) > 0:
            st.plotly_chart(message["aux"]["figure"][0])
        st.text("")

# Accept user input
st.session_state.prompt = st.chat_input("¿En qué puedo ayudarte?")


if st.session_state.prompt:
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(st.session_state.prompt)

    with st.chat_message("assistant"):

        response = lu.invoke_chain(
            question=st.session_state.prompt,
            messages=st.session_state.messages_foundations,
            sql_messages = st.session_state.sql_messages,
            model_name=model_options[model_options.index(st.session_state.model)],
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
        )
        st.write_stream(response)
        if "figure" in lu.invoke_chain.aux.keys() and len(lu.invoke_chain.aux["figure"]) > 0:
            st.plotly_chart(lu.invoke_chain.aux["figure"][0])
        if hasattr(lu.invoke_chain, 'recursos'):
            for recurso in lu.invoke_chain.recursos:
                st.button(recurso)

    # Add user message to chat history
    st.session_state.messages_foundations.append({"role": "user", "content": st.session_state.prompt, "aux": {}})
    st.session_state.messages_foundations.append({"role": "assistant", "content": lu.invoke_chain.response, "aux": lu.invoke_chain.aux})
    