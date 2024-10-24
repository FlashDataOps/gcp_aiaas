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
import pandas as pd

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")

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
        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]

'''
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(
    persist_directory=INDEX_PATH,
    embedding_function=embeddings
    )
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
        }
    )
'''

# First we need a prompt that we can pass into an LLM to generate this search query
prompt_create_sql = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Basándote en la tabla esquema de abajo, escribe una consulta SQl que pueda responder a la pregunta del usuario:
            {schema}

            Utiliza el historial para adaptar la consulta SQL. No añadas respuestas en lenguaje natural.
            
            IMPORTANTE: Si tu consulta selecciona todas las filas, limita los resultados obtenidos a 20. Por ejemplo: SELECT * from TABLE LIMIT 20
            RESPONDE ÚNICAMENTE CON CÓDIGO SQL. NO AÑADAS PALABRAS EN LENGUAJE NATURAL.
            LA CONSULTA DEBE ESTAR PREPARADA PARA SER EJECUTADA EN LA BASE DE DATOS
            
            No incluyas introducción, ni introduzcas en una lista la resupuesta
            
            - Pregunta: {input}
            - Query SQL:
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

prompt_create_sql_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Basándote en los siguientes datos, responde en lenguaje natural:
            Debes utilizar los siguientes datos para responder a la pregunta del usuario:
            {schema}
            
            - Pregunta: {input}
            - Query SQL: {query}
            - Respuesta: {response}
            
            En tu respuesta debes indicar la query SQL que se ha generado para la pregunta del usuario. Escribela de forma que sea fácil de leer. Evita que sea una única línea e indenta bien cada linea de la query.
            
            No utilices el formato que te he dado anteriormente para responder a la pregunta, muestra la información en uno o dos párrafos
            Siempre muestralo de una forma bonita y ordenada, utilizando tablas o bullets points con saltos de línea a ser posible.
            UTILIZA FORMATO MARKDOWN
            RESPONDE EN ESPAÑOL DE FORMA DETALLADA
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

prompt_response_sql = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Debes utilizar los siguientes datos para responder a la pregunta del usuario:
            - Query SQL: {query}
            - Respuesta: {result}
            
            En tu respuesta debes indicar la query SQL que se ha generado para la pregunta del usuario. Escribela de forma que sea fácil de leer. Evita que sea una única línea e indenta bien cada linea de la query.
            
            Siempre muestralo de una forma bonita y ordenada, utilizando tablas o bullets points con saltos de línea a ser posible.
            UTILIZA FORMATO MARKDOWN
            RESPONDE EN ESPAÑOL DE FORMA DETALLADA
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

prompt_analysis = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu trabajo es responder la pregunta del usuario utilizando únicamente la siguientes filas de un csv: {context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

prompt_ml = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual que es capaz de llamar a un modelo de machine learning de clasificación y ofrecer el resultado al usuario.
            - Parámetros de llamada: {params}
            - Resultado de la llamada {result}
            
            Debes dar una respuesta que incluya los parámetros utilizados para llamar al modelo en forma de tabla y de forma clara y sencilla pero que se note que es la parte importante del mensaje el resultado obtenido.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

@lru_cache(maxsize=None)
def get_custom_chain(model_name, temperature, max_tokens, params, type=None, db=None):
    """
    Create and return a retrieval-augmented generation (RAG) chain.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): The temperature parameter for generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        rag_chain: The retrieval-augmented generation chain.

    """
    model = get_model(model_name, temperature, max_tokens)
    
    if params is not None:
        chain = prompt_ml | model | StrOutputParser()
    else:
        
        #chain = create_pandas_dataframe_agent(model, df, agent_type="openai-tools", verbose=True)
        if type == "pandas_code":
            chain = create_sql_query_chain(model, db)
        else:
            chain = (
                prompt_response_sql | model | StrOutputParser()
            )
            '''
            chain = (
            {
                "context":retriever | af.format_docs, 
                "input": RunnablePassthrough()
            } 
            | prompt_analysis 
            | model 
            | StrOutputParser()
            )
            '''

    return chain


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

def invoke_chain(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, json_params=None, db_name=None):
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
    chain = get_custom_chain(model_name, temperature, max_tokens, json_params, db)
    
    history = create_history(messages)
    sql_history = create_history(sql_messages)
    aux = {}
    
    response = ""
    
    config = {
        "input": question, 
        "chat_history": history.messages, 
    }
    if json_params is not None:
        ml_result = af.simulate_model_prediction(json_params)
        config["params"] = json_params    
        config["result"] = ml_result
    else:
        llm = get_model(model_name, temperature, max_tokens)
        sql_chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt_create_sql
            | llm.bind(stop=["\nSQLResult:"])
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
        }
        query = sql_chain.invoke(config)
        query = clean_query(query)
        print("Query ->", query)
        sql_history.add_user_message(question)
        #sql_history.add_ai_message(query)
        try:
            print(db.get_table_info())
        except:
            traceback.print_exc()
            
        
        result = db.run(query)
        print(result)
        
        config = {
        "input": question, 
        "chat_history": history.messages, 
        "query": query,
        "response": result,
        "schema": get_schema
        }
        
        '''
        chain_pandas_code = get_custom_chain(model_name, temperature, max_tokens, json_params, "pandas_code", db)
        query = chain_pandas_code.invoke({"question": question})
        query = query.split("SQLQuery:")[-1].strip()
        result = db.run(query)
        chain = get_custom_chain(model_name, temperature, max_tokens, json_params, db)
        
        config["query"] = query    
        config["result"] = result
        '''
        
        
    for chunk in chain.stream(config):
        response+=chunk
        yield chunk
    
    
    history.add_user_message(question)
    history.add_ai_message(response)
    list_result = eval(result)
    if len(list_result) > 1:
        del config["schema"]
        plot_code = plot_chain.invoke(config)
        
        try:
            plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")
            exec(plot_code)
            aux["figure"] = eval("[fig]")
        except Exception as e:
            print(f"Error al generar el gráfico {e}")
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux