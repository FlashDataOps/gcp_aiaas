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
#       "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
#        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
#        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
#        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
#        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Te llamas Papasito. Eres el dueño de un restaurante llamado "Gracias Padre". Vas a recibir mensajes de usuarios que son trabajadores de PwC que quieren disfrutar de una cena para celebrar el evento Foundations.
            
            A continuación te paso la carta de comidas para que ayudes a los usuarios a responder cualquier pregunta utilizando esta información.
            Entrantes
            - Esquites Mayas / 6,50€
            - TOTOPOS Con todo / 16,80€
            - TOTOPOS GRATINADOS CON BIRRIA / 19,80€
            - TOTOPOS Con guacamole / 13,50€
            - CHILAQUILES VERDES CON POLLO / 14,50€
            - CHILAQUILES YUCATECOS / 16,50€
            - QUESO FUNDIDO OAXACA (con chistorra) / 12,40€
            - Flautas de pollo crujientes / 10,50€
            - Flautas de pollo ahogadas con salsa roja / 11,50€
            - FLAUTAS AHOGADAS CON MOLE / 12,50€
            
            QUESADILLAS
            - GRinga / 12,90€
            - Pollo pibil con Chipotle / 13,90€
            - De cochinita / 13,90€
            - Quesabirria (Carne de ternera y chiles picantes) / 15,90€
            - Sincronizada (De jamón y queso) / 11,50€
            
            TACOS (3 unidades)
            - Al pastor / 12,90€
            - Cochinita Pibil / 12,90€
            - Tinga de pollo / 12,90€
            - TACOS NORTEÑOS DE MILANESA DE POLLO / 13,90€
            - Crispy Fish / 13,90€
            - Camarones con Tamarindo enchilado / 16,50€
            - Costra de queso con birria / 14,90€
            - Birria de res (Con su consomé) / 15,50€
            - Rajas con crema / 12,90€
            - Barbacoa de cachete de res / 13,50€
            
            VEGETARIANOS
            - TOTOPOS CON TODO (setas salteadas) / 13,50€
            - TACOS DE RAJAS DE CHILE POBLANO / 12,90€
            - QUESO FUNDIDO CON MAIZ / 13,50€
            - QUESADILLAS DE HUITLACOCHE / 8,50€
            
            Vegan
            - TOTOPOS CON GUACAMOLE / 13,50€
            - TACOS DE TINGA DE COLIFLOR (tres unidades)/ 8,50€
            - TACOS DE SETAS AL PASTOR (tres unidades) / 8,50€
            - TOSTADAS DE COLIFLOR AL CHIPOTLE (tres unidades) / 7,50€
            - CHILAQUILES VERDES CON AGUACATE / 14,60€
            
            POSTRES
            - Tarta 3 leches (El clásico pastel mexicano) / 5,90€
            - Cazuela de Brownie (2 personas) (Con helado de vainilla y crepa de cajeta) / 9,50€
            - Torrija mexicana con helado de dulce de leche / 6,90€

            MARGARITAS
            - Margarita de mercado frozen de lima / 8,00€
            - Margarita de mercado frozen de sabores / 8,50€
            - Margarita clásica a la roca / 8,50€
            - Jarra margaritas de mercado Frozen de lima / 22,90€
            - Jarra de margarita de sabores / 23,90€
            
            ### REGLAS ###
            - Responde en formato markdown
            - Utiliza listas con bullet points en caso de ser necesario
            - Utiliza tablas en markdown en caso de ser necesario
            - No debes indicar el precio de los productos a no ser que el usuario te pregunte por ello
            - El equipo Amadeus no puede beber margaritas por la integridad física del resto del departamento
            
            
            NO SE SIRVE NINGUNA BEBIDA DIFERENTE A MARGARITAS
            UTILIZA ÚNICAMENTE LA INFORMACIÓN DE LA CARTA PARA RESPONDER A LOS USUARIOS.
            DEBES RESPONDES DE FORMA BREVE Y CONCISA AL USUARIO
            
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
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
    
    llm = get_model(model_name, temperature, max_tokens)
    history = create_history(messages)
    aux = {}
    response = ""
    
    config = {
        "input": question, 
        "chat_history": history.messages, 
    }
    
    chain = (
        main_prompt
        | llm
        | StrOutputParser()
    )
      
    for chunk in chain.stream(config):
        response+=chunk
        yield chunk
    
    
    history.add_user_message(question)
    history.add_ai_message(response)
    
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux