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
import azure.cognitiveservices.speech as speechsdk
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import traceback
from gtts import gTTS
import base64
import datetime

try:
    list_input_audio = [input for input in enumerate(sr.Microphone.list_microphone_names())]
except:
    traceback.print_exc()
    list_input_audio = [(0, "No Mic Detected")]

default_mic_name = "Microphone Array (IntelÂ® Smart "
default_mic_index = next((index for index, name in list_input_audio if default_mic_name in name), 0)
default_mic_name_selected = list_input_audio[default_mic_index][1]

if "input_audio" not in st.session_state:
   st.session_state.input_audio = default_mic_name_selected
    

def update_chat_input(new_input):
    js = f"""
    <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
            var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeInputValueSetter.call(chatInput, "{new_input}");
            var event = new Event('input', {{ bubbles: true}});
            chatInput.dispatchEvent(event);
        }}
        insertText({len(st.session_state.messages)});
    </script>
    """
    st.components.v1.html(js)

# Función para reconocimiento de voz a texto
async def speech_to_text():
    recognizer = sr.Recognizer()
    list_input_audio_names = [name for _, name in list_input_audio]
    
    try:
        pass
        #device_index = list_input_audio_names.index(st.session_state.input_audio)
    except ValueError:
        device_index = 0 # por defecto si no se ecuentra el micro de PwC
        
    with sr.Microphone(device_index=device_index) as source:
       audio = recognizer.listen(source)
       
       try:
           text = recognizer.recognize_google(audio, language='es-ES')
           update_chat_input(text)
           #st.success(f"Texto reconocido: {text}")
           st.session_state.user_input = text  # Actualizar el valor del input con el texto transcrito
       except sr.UnknownValueError:
           traceback.print_exc()
           #st.error("No se pudo entender el audio.")
       except sr.RequestError:
           traceback.print_exc()
           #st.error("Error al intentar usar el servicio de Google Speech Recognition.")

def text_to_speech(input_text):
    try:
        input_text = af.format_text_for_audio(input_text)
        tts = gTTS(input_text, lang='es')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)  # Reiniciar el puntero al inicio del archivo para reproducirlo
        
        #return mp3_fp
        # Reproducir solo si el archivo se generó exitosamente
        #st.audio(mp3_fp, format='audio/mp3', autoplay=True)
        return mp3_fp
    except Exception as e:
        st.error(f"Error al generar el audio: {e}")
        


def update_playback_rate(mp3_file, rate, autoplay=""):
    audio_data = mp3_file.read()
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    # Crear HTML para el reproductor con JavaScript para la velocidad
    html_code = f""" <div data-stale="false" width: 100% class="element-container st-emotion-cache-a1dagx e1f1d6gn4" data-test="element-container"> 
    <audio id="audio" controls {autoplay} style="width: 100%;"> 
    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio> 
    <script> document.getElementById("audio").playbackRate = {rate}; </script> </div> """

    # Incrustar el reproductor y el botón de descarga
    st.components.v1.html(html_code, height=100)
    
    
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
    
    
def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

# Reset chat history
def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
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

# Continue with sidebar settings...
with st.sidebar:
    st.title("Model Configuration")
    
    audio_toggle = st.toggle("Responses with audio", value=True)
    
    # Select mic input
    st.session_state.input_audio = st.selectbox(
       "Choose an audio input:",
       [elem[1] for elem in list_input_audio],
       index=default_mic_index,
    )
    
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
    st.session_state.temperature = st.slider('Select a temperature:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]

    st.session_state.max_tokens = st.number_input('Select max tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)
    
    clear_chat_column, record_audio_column= st.columns([1, 1])
    # Reset chat history button
    if st.button(":broom: Clear chat", use_container_width=True):
        reset_chat_history()

    if st.button("🎙️ Record", use_container_width=True):
        st.session_state.show_success_audio = True
        with st.spinner("Listening... 👂"):
            result = asyncio.run(speech_to_text())
  
# Render or update model information
render_or_update_model_info(st.session_state.model)

if st.session_state.initial_message_displayed == False:
    model_name = st.session_state.model
    temperature = st.session_state.temperature
    max_tokens_value = st.session_state.max_tokens
    with st.spinner("Generando reporte diario..."):
        # Generate initial message with content and auxiliary data
        initial_message_content = lu.summary_of_the_date_generation(
            model_name, temperature, max_tokens_value
        )
        st.write_stream(initial_message_content)
        
        # Generate audio for the initial message
        mp3_file = text_to_speech(lu.summary_of_the_date_generation.initial_message)
        lu.summary_of_the_date_generation.aux["aux"]["audio"] = mp3_file
        
        # Update the initial message in the session state
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": lu.summary_of_the_date_generation.initial_message,
                "aux": {
                    "figure": lu.summary_of_the_date_generation.aux["aux"]['figure'],
                    "audio": lu.summary_of_the_date_generation.aux["aux"]['audio']
                }
            }
        ]

# Display chat messages from history on app rerun
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if st.session_state.initial_message_displayed:
            st.markdown(message["content"])            
        
        if "audio" in message["aux"].keys():
            if not st.session_state.initial_message_displayed:
                update_playback_rate(mp3_file=message["aux"]['audio'], rate=1.55, autoplay='autoplay')
                
                st.session_state.initial_message_displayed = True
                # st.session_state.messages = []
            else:
                update_playback_rate(mp3_file=message["aux"]['audio'], rate=1.55)
                message["aux"]['audio'].seek(0)
        
        if "figure" in message["aux"].keys():
            for figure in message["aux"]["figure"]:
                st.plotly_chart(figure['figure'])
        
        if "figure_p" in message["aux"].keys():
            for figure in message["aux"]["figure_p"]:
                st.plotly_chart(figure)
                      

# Accept user input
prompt = st.chat_input("¿Cómo puedo ayudarte?", key="user_input")

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
        
        # Stream the response
        full_response = st.write_stream(response)
        
        # Prepare auxiliary data
        aux_v2 = lu.invoke_chain.aux
        
        # Generate audio if toggle is on
        if audio_toggle:
            with st.spinner("Generando audio ..."):
                mp3_file = text_to_speech(full_response)
                update_playback_rate(mp3_file=mp3_file, rate=1.55, autoplay="autoplay")
                mp3_file.seek(0)
                aux_v2["audio"] = mp3_file
                # st.audio(mp3_file, format="audio/mp3", autoplay=F)
                
        # Handle figures
        if "figure_p" in aux_v2.keys():
            with st.spinner("Generando gráficos ..."):
                for figure in aux_v2["figure_p"]:
                    st.plotly_chart(figure)
        
        # Update session state
        st.session_state.messages.extend([
            {"role": "user", "content": prompt, "aux": {}},
            {
                "role": "assistant", 
                "content": full_response, 
                "aux": aux_v2
            }
        ])