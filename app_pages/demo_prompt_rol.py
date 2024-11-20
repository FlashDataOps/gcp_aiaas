import streamlit as st
import langchain_demo_agent as lu
import pandas as pd
import json
from streamlit_navigation_bar import st_navbar
import time
import os
import aux_functions as af
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/promptrol/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/promptrol/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)


# Reset chat history
def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
    if "messages_foundations" in st.session_state:
        st.session_state.messages_foundations = []

# Función genérica para manejar clics en los botones
def toggle_button(button_key):
    # Desactivar todos los botones antes de activar el seleccionado
    for key in st.session_state.button_states:
        st.session_state.button_states[key] = False
    st.session_state.button_states[button_key] = True

model_options = ["llama3-70b-8192","gemini-1.5-flash-002"]
max_tokens = {
    "llama3-70b-8192": 8192,
#    "llama3-8b-8192": 8192,
#    "mixtral-8x7b-32768": 32768,
#   "gemma-7b-it": 8192,
   "gemini-1.5-flash-002": 128000,
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
    
# Inicialización de los estados de los botones
if 'button_states' not in st.session_state:
    st.session_state.button_states = {
        'button1': False,
        'button2': False,
        'button3': False,
        'button4': False,
        'button5': False,
    }


with st.sidebar:
    st.title("Configuración de modelo")

    # Select model
    st.session_state.model = st.selectbox(
        "Elige un modelo:",
        model_options,
        index=0
    )

    # # Select temperature
    # st.session_state.temperature = st.slider('Selecciona una temperatura:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # # Select max tokens
    # if st.session_state.max_tokens > max_tokens[st.session_state.model]:
    #     max_value = max_tokens[st.session_state.model]

    # st.session_state.max_tokens = st.number_input('Seleccione un máximo de tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)

    # # Reset chat history button
    if st.button(":broom: Vaciar chat", use_container_width=True):
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

# Crear botones con texto y on_click genérico
buttons = [
    ("Eres una especialista de la Agencia Tributaria de Canarias 👤", 'button1'),
    ("Eres una persona que le gustan mucho los emojis  🙂", 'button2'),
    ("Eres una persona que le gusta hablar mucho 🗣️", 'button3'),
    ("Eres una persona que le gusta hablar poco 🤫", 'button4'),
    ("Eres una persona que le gusta mostrar la información con tablas 📋", 'button5')
]

# Mostrar los botones y asociar la función de clic a cada uno
for label, key in buttons:
    st.button(label, key=key, on_click=toggle_button, args=(key,), use_container_width=True)

# Asignar el valor de prompt_selection según el estado de los botones
prompt_selection = next(
    (label for label, key in buttons if st.session_state.button_states[key]), 
       "Selecciona una opción"
) 

# Mostrar el texto del mensaje en el chat
st.write(f"**El prompt seleccionado es:** {prompt_selection}")

# Accept user input
st.session_state.prompt = st.chat_input("¿En qué puedo ayudarte?")


main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {prompt_selection}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

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
            main_prompt= main_prompt
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
    