import streamlit as st
import langchain_demo_agent as lu
import pandas as pd
import json
from streamlit_navigation_bar import st_navbar
import time
import os
import aux_functions as af
def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/agente/styles.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/agente/content.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)

# Reset chat history
def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
    if "messages_agent" in st.session_state:
        st.session_state.messages_agent = []
        st.session_state.sql_messages = []

model_options = ["llama3-70b-8192","gemini-1.5-flash-002", "gemini-1.5-pro-002"]
max_tokens = {
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "gemma-7b-it": 8192,
    "gemini-1.5-flash-002": 128000,
    "gemini-1.5-pro-002": 128000
}

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = 8192

# Initialize chat history
if "messages_agent" not in st.session_state:
    st.session_state.messages_agent = []
    st.session_state.sql_messages = []

with st.sidebar:
    st.title("Configuración de modelo")

    # Listar los archivos en la carpeta db
    carpeta_db = 'db' 
    try:
        dbs = os.listdir(carpeta_db)
        # Filtrar para mostrar solo archivos, no carpetas
        archivos_db = [f for f in dbs if os.path.isfile(os.path.join(carpeta_db, f))]
    except FileNotFoundError:
        archivos_db = []
        st.error(f"La carpeta '{carpeta_db}' no existe.")
    
    if archivos_db:
        archivo_db_seleccionado = st.selectbox("Selecciona una base de datos:", archivos_db)
        af.db_connection.db_name = archivo_db_seleccionado
    
    ERROR_PROVOCADO = st.toggle("Provocar error en la consulta")
    SLEEP_TIME = st.slider('Selecciona un tiempo (seg) por iteración', min_value=0, max_value=5, step=1)

    
    # Select model
    st.session_state.model = st.selectbox(
        "Elige un modelo:",
        model_options,
        index=1
    )

    # Select temperature
    st.session_state.temperature = st.slider('Selecciona una temperatura:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Select max tokens
    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]

    st.session_state.max_tokens = st.number_input('Seleccione un máximo de tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)

    # Reset chat history button
    if st.button(":broom: Vaciar Chat", use_container_width=True):
        reset_chat_history()
    
# Render or update model information
render_or_update_model_info(st.session_state.model)

# Display chat messages from history on app rerun
for message in st.session_state.messages_agent:
    with st.chat_message(message["role"], avatar = "https://www.hola.com/horizon/square/4fe315b0171d-bond-t.jpg" if message["role"] == "assistant" else "https://cdn-icons-png.flaticon.com/512/9750/9750857.png"
):
        st.markdown(message["content"])
        if "figure" in message["aux"].keys() and len(message["aux"]["figure"]) > 0:
            st.plotly_chart(message["aux"]["figure"][0])
        st.text("")

# Accept user input
prompt = st.chat_input("¿En qué puedo ayudarte?")
step = 1
aux = {}
if prompt:
    # Display user message in chat message container
    with st.chat_message("user", avatar="https://cdn-icons-png.flaticon.com/512/9750/9750857.png"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar= "https://www.hola.com/horizon/square/4fe315b0171d-bond-t.jpg"):
        st.write("**---------------------- INCICIO CADENA DE PENSAMIENTO ---------------------------**")
        st.write(f"**Paso {step}: Identificando intención... 🔍**")
        step += 1
        with st.spinner("Pensando..."):
            intent = lu.invoke_intent(
                question=prompt,
                messages=st.session_state.messages_agent,
                sql_messages = st.session_state.sql_messages,
                model_name=model_options[model_options.index(st.session_state.model)],
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
                )
        
            time.sleep(SLEEP_TIME)
        st.write(f"- **Intención detectada** -> {intent}")
            
        if "consulta" in intent:
            
            st.write(f"**Paso {step}: Generando consulta SQL... 📊**")
            step += 1
            with st.spinner("Pensando mucho..."):
                sql_query = lu.invoke_create_sql(
                question=prompt,
                messages=st.session_state.messages_agent,
                sql_messages = st.session_state.sql_messages,
                model_name=model_options[model_options.index(st.session_state.model)],
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
                )
                #sql_query = "SELCT COUNT(*) FROM COCHES"
            
                time.sleep(SLEEP_TIME)
            #st.write(f"- **Query SQL** -> {sql_query}")
            if ERROR_PROVOCADO:
                sql_query = sql_query.replace("SELECT", "SELCT")
            sql_query = sql_query.replace('```sql', "").replace("sql", "")

            st.code(sql_query, language='sql')

            
            st.write(f"**Paso {step}: Ejecutando consulta SQL... ▶️**")
            step += 1
            with st.spinner("Ejecutando consulta SQL..."):
                result_query = lu.run_query(sql_query)
                
                time.sleep(SLEEP_TIME)
                st.write(f"- **Resultado** -> {result_query}")
            
            st.write(f"**Paso {step}: Verificando consulta SQL... 🛠️**")
            with st.spinner("Verificando consulta SQL..."):
                step += 1
                
                for tries in range(3):
                    time.sleep(SLEEP_TIME)
                    if "Error" in result_query:
                        st.write(fr"- **Intento {tries+1} -> Arreglando consulta**")
                        sql_query = lu.invoke_fix_sql(
                        question=prompt,
                        messages=st.session_state.messages_agent,
                        sql_messages = st.session_state.sql_messages,
                        model_name=model_options[model_options.index(st.session_state.model)],
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens
                        )
                        sql_query = sql_query.replace('```sql', "").replace("sql", "")
                        result_query = lu.run_query(sql_query)
                        st.write(f"- **Resultado** -> {result_query}")
                    else:
                        if tries > 0:
                            st.write("- **Consulta SQL corregida y ejecutada correctamente ✅**")
                            sql_query = sql_query.replace('```sql', "").replace("sql", "")
                            st.code(sql_query, language='sql')
                            st.write(f"- **Resultado** -> {result_query}")
                            break
                        else:
                            st.write("- **La consulta no tiene errores ✅**")
                            #st.write(f"- **Resultado** -> {result_query}")
                            break
            
            st.write(f"**Paso {step}: ¿Se debería generar un gráfico?...❓**")
            step += 1
            gen_plot = False
            with st.spinner("Comprobando resultados de la base de datos..."):
                time.sleep(SLEEP_TIME)
                print(eval(result_query), type(eval(result_query)))
                if len(eval(result_query)) > 1:
                    gen_plot = True
                    st.write("- **Se generará un gráfico ✅**")
                else:
                    st.write("- **No se generará un gráfico ❌**")
                

            
            
            st.write("**---------------------- FIN CADENA DE PENSAMIENTO ---------------------------**")
            response = lu.invoke_gen_response(
            question=prompt,
            messages=st.session_state.messages_agent,
            sql_messages = st.session_state.sql_messages,
            model_name=model_options[model_options.index(st.session_state.model)],
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            sql_result = result_query
            )
            st.write_stream(response)
            
            if gen_plot:
                with st.spinner("Generando gráfico... 📊"):
                    try:
                        plotly_code = lu.invoke_gen_plot_agent_007(
                        question=prompt,
                        messages=st.session_state.messages_agent,
                        sql_messages = st.session_state.sql_messages,
                        model_name=model_options[model_options.index(st.session_state.model)],
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        sql_result = result_query,
                        sql_query = sql_query
                        )
                        plotly_code = plotly_code.replace("```python", "").replace('```', "").replace("fig.show()", "")
                        #st.code(plotly_code, language="python", line_numbers=True)
                        exec(plotly_code)
                        fig = eval("fig")
                        st.plotly_chart(fig)
                        aux["figure"] = [fig]
                    except:
                        st.write("Lo siento no he podido generar el gráfico 😔")
        else:
            st.write("**---------------------- FIN CADENA DE PENSAMIENTO ---------------------------**")
            response = lu.invoke_general_agent_007(
            question=prompt,
            messages=st.session_state.messages_agent,
            sql_messages = st.session_state.sql_messages,
            model_name=model_options[model_options.index(st.session_state.model)],
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
            st.write_stream(response)
        if "figure" in lu.invoke_chain.aux.keys() and len(lu.invoke_chain.aux["figure"]) > 0:
            st.plotly_chart(lu.invoke_chain.aux["figure"][0])
        if hasattr(lu.invoke_chain, 'recursos'):
            for recurso in lu.invoke_chain.recursos:
                st.button(recurso)

    # Add user message to chat history
    st.session_state.messages_agent.append({"role": "user", "content": prompt, "aux": {}})
    if "consulta" in intent:
        st.session_state.messages_agent.append({"role": "assistant", "content": lu.invoke_gen_response.response, "aux": aux})
    else:
        st.session_state.messages_agent.append({"role": "assistant", "content": lu.invoke_general_agent_007.response, "aux": aux})
