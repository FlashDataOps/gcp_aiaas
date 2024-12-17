import streamlit as st
import langchain_utils as lu
import aux_functions as af
from dotenv import load_dotenv
load_dotenv()
       
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

model_options = ["gpt-4o-mini", "gpt-4o"]
max_tokens = {
    "gpt-4o-mini": 8192,
    "gpt-4o": 8192,
    "o1-preview": 32768,
    "o1-mini": 32768,
}

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = max_tokens[st.session_state.model]

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.sql_messages = []

# Continue with sidebar settings...
with st.sidebar:
    st.title("Model Configuration")
      
    archivo_db_seleccionado = "nestle_db.db"
    af.db_connection.db_name = archivo_db_seleccionado
    
    # Select model
    st.session_state.model = st.selectbox(
        "Choose a model:",
        model_options,
        index=0
    )

    # Select temperature
    st.session_state.temperature = st.slider('Select a temperature:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]
    
    clear_chat_column, record_audio_column= st.columns([1, 1])
    # Reset chat history button
    if st.button(":broom: Clear chat", use_container_width=True):
        reset_chat_history()
  
# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])            
                
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