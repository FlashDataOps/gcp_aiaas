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

PDF_FOLDER = "pdfs"

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
        device_index = list_input_audio_names.index(st.session_state.input_audio)
    except ValueError:
        device_index = 0

    with sr.Microphone(device_index=device_index) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='en-UK')
            st.session_state.recognized_text = text  # Store recognized text in session state
            st.session_state.show_send_text_button = True  # Show send button
            st.success(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Unable to understand the audio.")
        except sr.RequestError:
            st.error("Error with Google Speech Recognition service.")
        return None

def text_to_speech(input_text):
    try:
        input_text = af.format_text_for_audio(input_text)
        tts = gTTS(input_text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)  # Reiniciar el puntero al inicio del archivo para reproducirlo
        return mp3_fp
    except Exception as e:
        st.error(f"Error generating audio: {e}")

def update_playback_rate(mp3_file, rate, autoplay=""):
    audio_data = mp3_file.read()
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    html_code = f""" 
    <div data-stale="false" width: 100% class="element-container st-emotion-cache-a1dagx e1f1d6gn4" data-test="element-container"> 
    <audio id="audio" controls {autoplay} style="width: 100%;"> 
    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio> 
    <script> document.getElementById("audio").playbackRate = {rate}; </script> 
    </div> """
    st.components.v1.html(html_code, height=100)

def render_or_update_model_info(model_name):
    with open("./design/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

def reset_chat_history():
    if "messages" in st.session_state:
        st.session_state.messages = []

# Initialize session state variables
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

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = max_tokens[st.session_state.model]
    st.session_state.input_audio = 1
    st.session_state.last_prompt = None
    

if "messages" not in st.session_state:
    st.session_state.recognized_text = ""
    st.session_state.messages = []
    st.session_state.sql_messages = []
    st.session_state.show_success_audio = False
    st.session_state.show_send_text_button = False
    st.session_state.is_recording = False

# Sidebar configuration
with st.sidebar:
    st.image("Logo-pwc.png", width=60)
    st.sidebar.header("Model Configuration")
    
    audio_toggle = st.toggle("Responses with audio", value=True)
    
    # List files in the 'db' folder
    carpeta_db = 'db' 
    try:
        dbs = os.listdir(carpeta_db)
        archivos_db = [f for f in dbs if os.path.isfile(os.path.join(carpeta_db, f))]
    except FileNotFoundError:
        archivos_db = []
        st.error(f"The folder '{carpeta_db}' does not exist.")
    
    af.db_connection.db_name = archivos_db[0]
        
    st.session_state.model = model_options[0]

    # Select temperature
    # st.session_state.temperature = st.slider('Select the level of creativity:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    clear_chat_column, record_audio_column = st.columns([1, 1])
    
    if st.button(":broom: Clear chat", use_container_width=True):
        reset_chat_history()
        
    # st.sidebar.header("Uploaded PDFs")
    # uploaded_pdfs = [file for file in os.listdir(PDF_FOLDER) if file.endswith(".pdf")] #! TODO METER LOS DEL CSV

    # if uploaded_pdfs:
    #     for pdf_file in uploaded_pdfs:
    #         pdf_name = pdf_file[:-4]
    #         st.sidebar.markdown(f"📄 **{pdf_name}**")
    # else:
    #     st.sidebar.write("No PDFs uploaded yet.")
        
  
# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])            
        
        if "audio" in message["aux"].keys():
            update_playback_rate(mp3_file=message["aux"]['audio'], rate=1.35)
            message["aux"]['audio'].seek(0)
        
        if "figure_p" in message["aux"].keys():
            for figure in message["aux"]["figure_p"]:
                st.plotly_chart(figure)

# Get the prompt
prompt = st.chat_input("How can I help you?")

# Process the prompt
if prompt:
    # Store the prompt in session state
    st.session_state.last_prompt = prompt

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
            with st.spinner("Generating audio ..."):
                mp3_file = text_to_speech(full_response)
                update_playback_rate(mp3_file=mp3_file, rate=1.35, autoplay="autoplay")
                mp3_file.seek(0)
                aux_v2["audio"] = mp3_file
                
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
                "content": full_response, 
                "aux": aux_v2
            }
        ])

# Ensure the chat input remains available
if st.session_state.last_prompt:
    # Clear the last_prompt after processing
    st.session_state.last_prompt = None