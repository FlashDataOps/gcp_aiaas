import streamlit as st
import pandas as pd
import numpy as np
import re
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

left_co, cent_co,last_co = st.columns([1,5,1])
with cent_co:
    image_url = "images/botimagen.png"
    st.image(image_url, use_column_width=True)

# Título y subtítulo centrados
st.markdown("""
    <div style='text-align: center;'>
        <h1>👓 Extractor de campos</h1>
        <h3>Sube un archivo de cliente del que extraer los campos</h3>
    </div>
    """, unsafe_allow_html=True)

# Configuración en la barra lateral
with st.sidebar:
    st.title("Menú de configuración")

    files = st.file_uploader("Suba aquí sus documentos")
    ramos = ['Automóviles', 'Vida']
    dropdown = st.selectbox("Elija el ramo que quiere investigar", ramos)
    
    if files is not None:
        # Leer el contenido del archivo
        files = files.read().decode("utf-8")

# Inicializar historial de chat y memoria
if "messages" not in st.session_state:
    st.session_state.messages = []

if "dfs" not in st.session_state:  # Diccionario para guardar los DataFrames
    st.session_state.dfs = {}

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5)

# for message in st.session_state.messages:
#     avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
#     with st.chat_message(message["role"], avatar=avatar):
#         st.markdown(message["content"])
#         if message["role"] == "assistant":
#             if st.session_state.df is not st.session_state.df.empty:
#                 st.dataframe(st.session_state.df, hide_index = True, use_container_width=True)

for i, message in enumerate(st.session_state.messages):
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i in st.session_state.dfs:
            df = st.session_state.dfs[i]
            if df is not df.empty:  
                st.dataframe(df, hide_index=True, use_container_width=True)

# Procesar la pregunta del usuario
if question := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": question})

    prompt = ChatPromptTemplate.from_messages([
        ('system', """
        Eres un asistente especialista en la extracción de campos relevantes de archivos.
        Debes comenzar la respuesta con un párrafo introductorio en el que destaques algún punto de información clave del cliente como su nombre o el seguro que quieren contratar.
        Debes extraer siempre todos los campos que haya en el archivo.
        No muestres los campos en el párrafo inicial.
        La información extraída solo debe mostrarse en la tabla del código de python.
        Debes dar una respuesta continua, evita hacer secciones.
        Antes de la tabla di: Esta es la información extraída del cliente:
        Siempre tienes que crear un código python que pueda ser ejecutado directamente.
        Vas a recibir información de un cliente y vas a extraer campos de información relevante.
        Después del párrafo inicial, debes generar un código en Python para mostrar los campos extraídos. El código de Python debe
        crear un único dataframe de Pandas usando Streamlit y lo muestre en un st.dataframe().
        La tabla debe tener un formato de dos columnas y tantas filas como campos haya.
        En st.dataframe() mete el parámetro: hide_index=True
        En st.dataframe() mete el parámetro: use_container_width=True
        
        Información que vas a utilizar para responder: {files}
        """),
        ("placeholder", "{chat_history}"),
        ('user', "{question}")
    ])

    # Usar el modelo para generar la respuesta
    llm = ChatGroq(model="llama3-70b-8192")
    chain = (prompt | llm | StrOutputParser())

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(question)

        # Obtener la respuesta
        response = chain.invoke({"question": question, "files":files, "chat_history": st.session_state.memory.chat_memory.messages})

    # Mostrar la respuesta y procesarla en una tabla
    with st.chat_message("assistant", avatar="🤖"):
    # Usamos una expresión regular para extraer el bloque de código Python
        text_response = response.split("```")[0].strip()
        text_response
        
        try:
            # Primer intento: buscar cualquier bloque entre triple backticks (``` ... ```)
            code_match = re.search(r"```(.*?)```", response, re.DOTALL)

        # Verificar si el bloque encontrado comienza con "python"
            if code_match.group(1).strip().startswith("python"):
                # Eliminar "python" al inicio si está presente
                python_code = code_match.group(1).strip()[len("python"):].strip()
                
                local_scope = {}
                exec(python_code, {}, local_scope)
            else:
                # Si no inicia con "python", simplemente usar el contenido extraído
                python_code = code_match.group(1).strip()

                local_scope = {}
                exec(python_code, {}, local_scope)

        except Exception as e:
            st.error(f"Hubo un error al procesar los datos: {e}")
    

    # if "df" in local_scope:
    #  st.session_state.df = local_scope["df"]
    
    if "df" in local_scope:
        df = local_scope["df"]
        st.session_state.dfs[len(st.session_state.messages)] = df  # Asociar al mensaje actual
    else:
        df = None

    # Agregar la tabla en el historial de chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": text_response
    })

    st.session_state.memory.save_context(
        {"Human": question},  # Inputs
        {"AI": text_response}  # Output
    )