import json
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import os
from functools import lru_cache
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI

from sympy import im
import aux_functions as af
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback

load_dotenv()

def get_schema(_):
    db = af.db_connection.get_db()
    schema = db.get_table_info()
    return schema

def run_query(query):
    db = af.db_connection.get_db()
    return db.run(query)

def clean_query(query):
    return query.replace("```sql", "").replace("```", "").replace("[SQL:", "").replace("]", "").strip()

@lru_cache(maxsize=None)
def get_model(model_name, temperature, max_tokens):
    """
    Returns a language model based on the specified model name, temperature, and max tokens.

    Args:
        model_name (str): The name of the language model.
        temperature (float): The temperature parameter for generating responses.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        ChatGroq: The language model object based on the specified parameters.
    """
    print(f"Parámetros de modelo {model_name, temperature, max_tokens}")
    llm = {
        "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
        "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True, temperature=temperature),
        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True, temperature=temperature),
        "llama-3.1-70b-versatile": ChatGroq(temperature=temperature,model_name="llama-3.1-70b-versatile", max_tokens=max_tokens),
    }
    return llm[model_name]

# First we need a prompt that we can pass into an LLM to generate this search query
prompt_create_sql = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        (
            "system",
            """Basándote en la tabla esquema:
            {schema}
            
            Escribe una consulta SQL para SQLLite que pueda responder a la pregunta del usuario.            
            No incluyas introducción, ni introduzcas en una lista la resupuesta, solo incluye la consulta SQL.
            
            """,
        ),
        ("user", "{input}")
    ]
)

prompt_create_sql_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Basándote en los siguientes datos, responde en lenguaje natural. 
            Debes utilizar los siguientes datos para responder a la pregunta del usuario:
            {schema}
            
            - Pregunta: {input}
            - Query SQL: {query}
            - Respuesta: {response}
            
            
            Utiliza formato markdown y emplea elementos visuales para hacer la respuesta más atractiva, como tablas en caso necesario. 
            Usa siempre inglés como idioma de texto.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

prompt_custom_chart = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Responde únicamente con código python
            Debes utilizar los siguientes datos para escribir el código en plotly en python que represente la respuesta realizada con la siguiente query:
            - Query SQL: {query}
            - Respuesta: {response}
            
            SOLO DEBES INCLUIR CÓDIGO PYTHON EN TU RESPUESTA. NO INCLUYAS LENGUAJE NATURAL INTRODUCIENDO TU RESPUESTA.
            HAZ EL GRÁFICO BONITO Y VISUAL. QUIERO QUE ESTÉ PREPARADO PARA SER MOSTRADO ANTE UN CLIENTE MUY IMPORTANTE.
            
            ASEGURATE DE QUE LA RESPUESTA TIENE ÚNICAMENTE CÓDIGO PYTHON
            
            RESPONDE EN ESPAÑOL
            """,
        ),
        ("user", "{input}"),
    ]
)


prompt_intent = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu tarea es decidir cuál es la intención del usuario a partir del mensaje del usuario. Las posibilidades son:
            - Consulta: Si el usuario realiza una consulta sobre un dataset de hoteles con el siguiente schema: {schema}
            - Otro: Cualquier cosa que no tenga nada que ver con realizar una consulta a los datos de NH
            
            Responde únicamente con las palabras [Consulta, Otro]. En caso de no saber a que se refiere responde Otro
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

prompt_general = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente que trabaja en NH Hoteles. Tu tarea es ayudar al usuario a entender la información del modelo de datos de NH.
            Puedes realizar la siguiente tarea:
            - Consulta: Si el usuario realiza una consulta sobre un dataset, se puede generar un gráfico y una respuesta en formato de audio para poder escucharla. El esquema del dataset es el siguiente {schema}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")        
    ]
)


def create_history(messages):
    """
    Creates a ChatMessageHistory object based on the given list of messages.

    Args:
        messages (list): A list of messages, where each message is a dictionary with "role" and "content" keys.

    Returns:
        ChatMessageHistory: A ChatMessageHistory object containing the user and AI messages.

    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192):
    """
    Invokes the language chain model to generate a response based on the given question and chat history.

    Args:
        question (str): The question to be asked.
        messages (list): List of previous chat messages.
        model_name (str, optional): The name of the language chain model to use. Defaults to "llama3-70b-8192".
        temperature (float, optional): The temperature parameter for controlling the randomness of the model's output. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 8192.

    Yields:
        str: The generated response from the language chain model.

    """
    db = af.db_connection.get_db()
    llm = get_model(model_name, temperature, max_tokens)
    history = create_history(messages)
    sql_history = create_history(sql_messages)
    aux = {}
    
    response = ""
    
    config = {
        "input": question, 
        "chat_history": history.messages, 
    }
    
    intent_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_intent
        | llm
        | StrOutputParser()
    )
    
    res_intent = intent_chain.invoke(config).strip().lower()
    print(f"La intención del usuario es -> {res_intent}")
    
    if "consulta" in res_intent:
        sql_chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt_create_sql
            | llm.bind(stop=["\nSQl:"])
            | StrOutputParser()
        )
        
        chain = (
            prompt_create_sql_response
            | llm
            | StrOutputParser()
        )
        
        plot_chain = (
            prompt_custom_chart
            | llm
            | StrOutputParser()
        )
        
        config = {
        "input": question, 
        "chat_history": sql_history.messages,
        "few_shots": af.create_few_shots() 
        }
        
        query = sql_chain.invoke(config)
        query = clean_query(query)
        print(query)
        sql_history.add_user_message(question)
        #sql_history.add_ai_message(query)
        print("Ejecutando consulta...")
        flag_correct_query = False
        try:   
            result = db.run(query)
            #print("RESULTADO ANTES", result)
            result, _ = af.dividir_en_bloques_por_grupo(resultados_sql=eval(result), max_filas=50)
            flag_correct_query = True
            print("Consulta ejecutada correctamente")
        except:
            result = f"No se ha podido ejecutar la consulta. Indica al usuario que existe un problema a la hora de realizar la consulta SQL {query} en la base de datos. Responde de forma breve"
            traceback.print_exc()
            
        config = {
        "input": question, 
        "chat_history": history.messages, 
        "query": query,
        "response": result,
        "schema": get_schema
        }
    else:
        config["schema"] = get_schema
        chain = prompt_general | llm | StrOutputParser()
        
    for chunk in chain.stream(config):
        response+=chunk
        yield chunk
    
    
    history.add_user_message(question)
    history.add_ai_message(response)
    
    if "consulta" in res_intent and flag_correct_query == True:
        try:
            print(result, type(result))
            list_result = result
            if len(list_result) > 1:
                
                del config["schema"]
                plot_code = plot_chain.invoke(config)
                #print(plot_code)
                plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")
                exec(plot_code)
                
                aux["figure_p"] = eval("[fig]")
        except Exception as e:
            print(f"Error al generar el gráfico {e}")
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux
    
