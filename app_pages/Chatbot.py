import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import plotly.graph_objects as go
import plotly.express as px  # Import adicional para gráficos más sofisticados

# Cargar las variables del archivo .env
load_dotenv()

# image_url = "images/botimagen.png"
# st.image(image_url, use_column_width=False)
# Título y subtítulo centrados
left_co, cent_co,last_co = st.columns([1,5,1])
with cent_co:
    image_url = "images/botimagen.png"
    st.image(image_url, use_column_width=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1>🤖 ChatBot de consulta de BBDD</h1>
        <h3>Haga consultas a la base de datos de la compañía en lenguaje natural</h3>
    </div>
    """, unsafe_allow_html=True)

def reset_chat_history():
    """
    Resets the chat history by clearing the 'messages' list in the session state.
    """
    if "messages" in st.session_state:
        st.session_state.messages = []

with st.sidebar:
    if st.button(":broom: Clear chat", use_container_width=True):
        reset_chat_history()

db_conn = sqlite3.connect("db/database.db")
db = SQLDatabase.from_uri("sqlite:///database.db")

# Configurar la memoria de conversación si no está inicializada
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    st.session_state.memory_sql = ConversationBufferWindowMemory(k=5, return_messages=True)

# Configurar el modelo LLM con LLaMA3-70b-8192
llm = ChatGroq(model="LLaMA3-70b-8192")  # Configurado para usar el modelo de Meta

# Inicializar el chat si no está inicializado
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial de chat al volver a ejecutar la aplicación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")
        if "dataframe" in message:
            st.write("Resultados de la consulta:")
            st.dataframe(message["dataframe"])
        if "figure" in message:
            fig = go.Figure(message['figure'])
            st.plotly_chart(fig)


inicio = "SELECT * FROM Sheet1"
contenido = pd.read_sql_query(inicio, db_conn)
# contenido
tablas_y_columnas = contenido.columns

# Input de preguntas
if question := st.chat_input("Escribe tu pregunta aquí..."):
    # Registrar y mostrar el mensaje del usuario inmediatamente
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(f"{question}")

    # Crear el prompt con las tablas y columnas disponibles como texto estático
    # tablas_y_columnas = "\n".join([f"Tabla '{tabla}': {', '.join(columnas)}" for tabla, columnas in columnas_disponibles.items()])
    empresas_previas = ', '.join(st.session_state.get('previous_companies', []))
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ('system', """
        Eres un asistente especializado en análisis de datos SQL. Las columnas disponibles en las tablas son las siguientes:
        {tablas_y_columnas}
        
        Cuando generes una consulta SQL, asegúrate de devolver solo la consulta SQL en texto plano, sin ningún texto adicional o explicación.
        Debes proporcionar una consulta en lenguaje natural para que los usuarios entiendan los resultados de la consulta SQL sin tener que leer la tabla.
        Todas las empresas deben tener una prima distinta.
         
        La tabla a la que tienes que dirigir las consultas se llama "Sheet1"    

        1. Para preguntas sobre "top clientes" en una industria específica, la consulta debe devolver la siguiente información en formato tabla:
            - Nombre
            - Industria
            - CNAE 4
            - Comunidad Autónoma
            - Provincia
            - Prima neta venta total.
            Ten en cuenta que un Nombre solo puede salir una vez, si sale más de dos veces debería estar sumado los importes
        
        2. Para preguntas sobre qué ramos se han VENDIDO, la consulta debe devolver:
            - Nombre
            - Ramo Vendido
            - Prima neta venta total (solo ventas, no estimaciones).
            Ten en cuenta que solo deben ser ramos de la columna Ramo Vendido, debes tener en cuenta {empresas_previas}. Elimina filas duplicadas
            
        3. Para preguntas sobre qué ramos se PROPONDRÍAN, excluye las ventas y devuelve:
            - Nombre
            - Ramo Propuesto
            - Propensión
            - Prima neta estimada de Xsellling
            Ten en cuenta que solo deben ser ramos propuestos, debes tener en cuenta {empresas_previas}. Elimina filas duplicadas
        
        4. Para preguntas sobre "en qué industria tengo una mayor oportunidad de mercado", devuelve:
            - Industria
            - Sumatorio de prima neta de xselling por industria en euros
            Debes tener en cuenta todas las propensiones y sacar todas las industrias de mayor a menor. Elimina filas duplicadas

        5. Para preguntas sobre "en qué ramo tengo una mayor oportunidad de mercado", devuelve:
            - Ramo
            - Sumatorio de prima neta de xselling por ramo en euros
            Debes tener en cuenta todas las propensiones y sacar todos los ramos de mayor a menor. Elimina filas duplicadas

        6. Para preguntas sobre "en qué provincia tengo una mayor oportunidad de mercado", devuelve:
            - Comunidad Autónoma
            - Sumatorio de prima neta de xselling por provincia autonoma en euros
            Debes tener en cuenta todas las propensiones y sacar todas las provincias de mayor a menor. Elimina filas duplicadas

        7. Cuando se pida información detallada de un cliente debes devolver la información completa. 
            Debes tener en cuenta que los datos de las empresas en la tabla se encuentran como Empresa X, siendo X un número. Elimina filas duplicadas
        
        8. Para preguntas sobre "Soy un comercial de una provincia que trabajo en una industria concreta, ¿a qué clientes debería ir?", devuelve:
            - Nombre
            - Ramo Propuesto
            - Industria
            - CNAE 4
            - Propensión
            - Prima neta estimada de cross-selling.
        Debes filtrar por provincia y por industria previamente. Elimina filas duplicadas
        
        Además, debes proporcionar un resumen breve del contenido de la tabla en lenguaje natural para los usuarios, destacando los puntos más relevantes.
        """),
        ('user', "{question}")
    ])

    # Ejecutar el prompt y conectar con el LLM
    chain = (prompt | llm | StrOutputParser())

    try:
        # Ejecutar la consulta y pasar el historial de conversación
        response = chain.invoke({
            "question": question,
            "empresas_previas": empresas_previas,
            "tablas_y_columnas": tablas_y_columnas,
            "chat_history": st.session_state.memory.chat_memory.messages
        })
        # Limpiar la respuesta para extraer solo la consulta SQL
        if "SELECT" in response:
            start_index = response.find("SELECT")
            query = response[start_index:].split(";")[0] + ";"

            # Agregar un párrafo introductorio
            intro_text = f"Para responder a tu pregunta '{question}', he generado los siguientes resultados basados en los datos proporcionados:"

            # Mostrar la consulta SQL generada
            try:
                # Ejecutar la consulta SQL
                result_df = pd.read_sql_query(query, db_conn)

                # Verificar si el DataFrame tiene datos
                if not result_df.empty:
                    # Generar un resumen en lenguaje natural
                    summary = f"Se encontraron {len(result_df)} registros. Los resultados incluyen las siguientes columnas: {', '.join(result_df.columns)}."
                    
                    # Crear un segundo prompt para explicar el contenido de la tabla
                    explanation_prompt = ChatPromptTemplate.from_messages([
                        ('system', "Eres un asistente especializado en análisis de datos con experiencia en ventas de seguros. Proporciona explicaciones claras y concisas que ayuden a los agentes de ventas a entender y utilizar la información."),
                        ('user', f"""
                        He generado una tabla con los siguientes datos:
                        
                        {result_df.to_string(index=False)}
                        
                        Por favor, proporciona una explicación breve (aproximadamente dos párrafos) que resalte los hallazgos más importantes y cómo los agentes de ventas pueden beneficiarse de esta información.
                        """)    
                    ])

                    # Ejecutar el segundo prompt con el LLM para obtener la explicación
                    explanation_chain = (explanation_prompt | llm | StrOutputParser())
                    explanation_response = explanation_chain.invoke({
                        "question": question,
                        "chat_history": st.session_state.memory.chat_memory.messages
                    })

                    # Mostrar la explicación generada por el LLM
                    assistant_message = {
                        "role": "assistant",
                        "content": f"{intro_text}\n\n{summary}\n\nExplicación: {explanation_response}",
                        "sql": query,
                        "dataframe": result_df
                    }
                    
                    prompt_custom_chart = ChatPromptTemplate.from_messages(
                    [
                        MessagesPlaceholder(variable_name="chat_history"),
                        (
                            "system",
                            """
                            Responde únicamente con código Python.
                            Debes utilizar los siguientes datos para escribir el código en Plotly en Python que represente la respuesta realizada con la siguiente query:
                            - Query SQL: {query}
                            - Respuesta: {response}
                            
                            SOLO DEBES INCLUIR CÓDIGO PYTHON EN TU RESPUESTA. NO INCLUYAS LENGUAJE NATURAL EN TU RESPUESTA.
                            HAZ EL GRÁFICO LO MÁS BONITO Y PROFESIONAL POSIBLE. UTILIZA EL EJE X COMO VARIABLE PARA ASIGNAR LOS COLORES EN EL GRÁFICO, DE MODO QUE LA LEYENDA CORRESPONDA AL EJE X. AGREGA TÍTULOS, ETIQUETAS DE EJE, LEYENDAS Y CUALQUIER OTRO ELEMENTO QUE MEJORE LA COMPRENSIÓN DEL GRÁFICO. ASEGÚRATE DE QUE EL GRÁFICO SEA CLARO Y FÁCIL DE ENTENDER PARA AGENTES DE VENTAS DE UN BROKER DE SEGUROS.
                            NUNCA DEBES GENERAR GRÁFICOS PARA LAS PREGUNTAS QUE VIENEN INDICADAS CON: 'No debes mostrar ningún gráfico'.
                            ES OBLIGATORIO QUE LA INFORMACIÓN CONCUERDE CON EL {dataframe}.
                                1. Para preguntas sobre "top clientes" en una industria específica, la consulta debe devolver la siguiente información en formato tabla:
                                - El eje X será la Empresa
                                - El eje Y será la Prima Neta Venta Total
                                Ten en cuenta que una empresa solo puede salir una vez.
                            
                            2. Para preguntas sobre qué ramos se han VENDIDO o qué ramos les hemos ofrecido a estas empresas, la consulta debe devolver:
                                No debes mostrar ningún gráfico.
                                
                            3. Para preguntas sobre qué ramos se PROPONDRÍAN, excluye las ventas y devuelve:
                                - El eje X será la Empresa-Ramo
                                - El eje Y será la Propensión
                                Ten en cuenta que no debes mostrar probabilidades del 100%.
                            
                            4. Para preguntas sobre "en qué industria tengo una mayor oportunidad de mercado", devuelve:
                                - Industria
                                - Sumatorio de Prima Neta de Xselling por Industria
                                Debes tener en cuenta todas las probabilidades y sacar todas las industrias de mayor a menor.
                                Debes mostrar las 5 primeras columnas.

                            5. Para preguntas sobre "en qué ramo tengo una mayor oportunidad de mercado", devuelve:
                                - El eje X será el Ramo
                                - El eje Y será Sumatorio de Prima Neta de Xselling por Ramo
                                Debes tener en cuenta todas las probabilidades y sacar todos los ramos de mayor a menor.
                                Debes mostrar las 5 primeras columnas.

                            6. Para preguntas sobre "en qué comunidad autónoma tengo una mayor oportunidad de mercado", devuelve:
                                - El eje X será la Comunidad Autónoma
                                - El eje Y será Sumatorio de Prima Neta de Xselling por Comunidad Autónoma
                                Debes tener en cuenta todas las probabilidades y sacar todas las comunidades autónomas de mayor a menor.
                                Debes mostrar las 5 primeras columnas.

                            7. Cuando se pida información detallada de un cliente debes devolver la información completa. 
                                No debes mostrar nada.
                                
                            8. Para preguntas sobre "Soy un comercial de una comunidad que trabajo en una industria concreta, ¿a qué clientes debería ir?", devuelve:
                                No debes mostrar nada.
                        
                            EL CÓDIGO DEBE DEFINIR UNA VARIABLE LLAMADA 'fig' QUE CONTENGA LA FIGURA DE PLOTLY.
                            
                            ASEGÚRATE DE QUE LA RESPUESTA TIENE ÚNICAMENTE CÓDIGO PYTHON.
                            
                            RESPONDE EN ESPAÑOL.
                            """,
                        ),
                        ("user", "{input}"),
                    ]
                    )
                    plot_chain = (prompt_custom_chart | llm | StrOutputParser())
                    len_df = len(result_df)
                    plot_code = {}
                    if len(result_df) > 2:
                        try:
                            plot_code = plot_chain.invoke({
                                "query": query,
                                "response": explanation_response,
                                "input": question,
                                "chat_history": st.session_state.memory.chat_memory.messages,
                                "empresas_previas": empresas_previas,
                                "len_df": len_df,
                                "empresas": result_df['Empresa'].unique() if 'Empresa' in result_df.columns else result_df['Nombre'].unique(),
                                "dataframe": result_df
                            })
                            try:
                                plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")# Mostrar el código generado
                                local_vars = {'df': result_df}  # Asignar result_df a df
                                exec(plot_code, globals(), local_vars)
                                fig = local_vars.get('fig')
                            except Exception as e:
                                print(f"Error al ejecutar el código del gráfico: {e}")
                                fig = None
                        except Exception as e:
                            print(f"Error al generar el código del gráfico: {e}")
                            plot_code = {}
                            fig = None
                    else:
                        fig = None
                        
                    # Guardar la figura como un diccionario en el mensaje del asistente
                    if fig is not None:
                        assistant_message['figure'] = fig.to_dict()

                    st.session_state.messages.append(assistant_message)
                    with st.chat_message("assistant"):
                        st.markdown(f"{intro_text}")
                        st.dataframe(result_df)
                        st.markdown(f"**Explicación:** {explanation_response}")
                        try:
                            if fig is not None:
                                if len(result_df) > 1:
                                    st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Error al mostrar el gráfico: {e}")
                    

                    # Guardar la conversación en la memoria

                    st.session_state.memory.save_context(
                        {"Human": question},
                        {"AI": explanation_response + "\n\n" + query}
                    )
                else:
                    st.warning("La consulta se ejecutó correctamente, pero no devolvió resultados.")
            except Exception as db_error:
                st.error(f"Error ejecutando la consulta: {db_error}")
        else:
            st.error("La respuesta no contiene una consulta SQL válida.")
    except Exception as e:
        st.error(f"Error procesando la pregunta: {e}")
