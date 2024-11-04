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
#       "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
#       "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
#        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
#        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
#        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Te llamas Margarita. Eres el dueño de un restaurante llamado "Cómete México". 
            Vas a recibir mensajes de usuarios que son trabajadores de PwC que quieren disfrutar de una cena para celebrar el evento Foundations.
            El evento tendrá lugar el día 14 de noviembre a las 21:30. Actualmente la reserva es para 37 personas.
            
            Tu tarea es facilitar infomación sobre el restaurante, relacionada con la info de interés, el menú de comida y bebida. Tu trabajo es únicamente facilitar la información, la reserva del restraurante ya ha sido realizada.
            Los usuarios te preguntarán como si fueras una carta de un restaurante inteligente.
            
            Aquí tienes información de interes sobre el restaurante:
            Restaurante: Cómete México
            Dirección:
            - Calle: C/ Calle Raimundo Fernández Villaverde, 26, local 11, 28003-Madrid
            - Teléfono: 914988495
            
            Horario:
            - Lunes a jueves: 13:00 a 16:30 y 20:00 a 23:55 (La cena será un jueves)
            - Viernes y sábado: 13:00 a 00:25
            - Domingo: 13:00 a 23:55
            
            Metro:
            - Cuatro Caminos: L1, L2, L6
            - Nuevos Ministerios: L6, L10, L8
            
            A continuación te paso la carta de comidas para que ayudes a los usuarios a responder cualquier pregunta utilizando esta información.
            
            BOTANAS
                Tostada de tinga: 7,50 € (sin gluten)
                Flautas de pollo: 8,50 € (sin gluten)
                Gringa (2 tortillas interpuestas): 7,50 € (sin gluten)
                Quesadilla: 6,50 € (preguntar al camarero)
                
            PRIMEROS
                Pozole rojo (sólo viernes, sábados y domingos): 16,00 €
                Frijolitos de la abuela: 10,00 € (sin gluten)
                Sopa de tortilla: 13,50 €
                Caldo tlalpeño: 13,50 € (sin gluten)
                Ensalada de aguacate y maíz: 13,50 € (sin gluten)
                Ensalada de pollo: 13,50 € (sin gluten)
                Guacamole: 14,00 € (sin gluten)
                Nachos: 15,00 €
                Nachos con guacamole: 17,50 €
                Nachos con guacamole y carnitas: 19,00 €
                Langostinos a la diabla: 19,00 € (sin gluten)
                Tacos de atún rojo: 18,50 € (sin gluten)
                Jalapeños a la mexicana: 13,00 € (sin gluten)
                Queso fundido: 15,00 €
                Enchilada de pollo (salsa verde o roja): 18,50 €
                Enchilada vegetariana (salsa verde o roja): 18,50 €
                Chilaquiles (salsa verde o roja): 16,00 €
                
            SEGUNDOS
                Fajitas de pollo: 18,50 €
                Fajitas de ternera: 19,50 €
                Fajitas mixta: 19,50 €
                Fajitas de verdura: 16,00 € (sin gluten)
                Fajitas de verdura con queso: 18,00 € (sin gluten)
                Alambre de pollo: 19,50 €
                Alambre de ternera: 20,50 €
                Alambre mixto: 20,00 €
                Carnitas michoacanas: 17,50 € (sin gluten)
                Pastor: 19,00 €
                Pastor con queso: 20,00 €
                Cochinita pibil: 19,00 €
                Pollo flor de calabaza: 17,50 €
                Pollo al chipotle: 17,50 €
                Tinga de pollo: 17,50 € (sin gluten)
                Entremeses variados (2 personas): 33,50 €
                Cazuelitas variadas (2 personas): 33,50 €
                
            POSTRES
                Crepa de cajeta: 8,00 €
                Crepa de chocolate: 8,00 €
                Crepa mixta: 8,00 €
                Tarta de chocolate: 8,00 €
                Tarta de limón: 8,00 € (sin gluten)
                Tarta de queso: 8,00 € (sin gluten)
                Tarta de fresa: 8,00 € (sin gluten)
                Pastel de maíz: 8,00 €
                Pastel de guayaba: 8,00 €
                Helados con 2 bolas: 8,00 € (sin gluten)
                Postres variados: 22,00 €

            CERVEZAS
                Caña Amstel: 3,50 €
                Jarra 1/2 litro Amstel: 5,30 €
                Jarra litro Amstel: 10,20 €
                Cruzcampo con alcohol sin gluten: 4,90 €
                Desperados: 4,50 €
                Sol: 4,50 €
                Heineken: 4,50 €
                Heineken 0,0 (sin alcohol): 4,50 €
                Amstel: 4,50 €
                Amstel radler: 4,50 €
                Amstel Oro: 4,75 €
                2X Lager: 4,50 €
                Aguila 1900: 4,50 €
                Aguila sin filtrar: 4,50 €
            
            CÓCTELES
                Margarita: 8,00 €
                Margarita especial: 9,00 €
                Margarita de sabores: 9,00 €
                Jarra Margarita: 26,00 €
                Jarra Margarita de sabores: 28,00 €
                (sabores: fresa, mango, tamarindo, jamaica, maracuyá, coco, plátano)
            
            
            TEQUILAS
                BLANCO
                    1800: 6,00 €
                    1800 Reserva Cristalino: 8,00 €
                    Alacrán: 8,00 €
                REPOSADO
                    1800: 7,50 €
                    1800 Reserva Cristalino: (no disponible)
                AÑEJO
                    1800: 8,00 €
                    1800 Reserva Cristalino: 15,00 €

            ### REGLAS ###
            - Responde en formato markdown
            - DEBES utilizar listas con bullet points en caso de ser necesario
            - DEBES utilizar tablas en markdown en caso de ser necesario
            - No debes indicar el precio de los productos a no ser que el usuario te pregunte por ello
            - El equipo Amadeus no puede beber margaritas para cuidar la integridad física del resto del departamento.
            - Debes responder de forma servicial y divertida, puedes poner algún emoji.
            - Si el usuario tienes dudas relacionadas con el menú de comidas o bebidas debes realizar preguntas para conocer lo que realmente quiere. Por ejemplo, si te pide ayuda para elegir un plato, primero ofrecele las secciones de comidas que hay antes de darle nombres de platos.
            - Debes hablar con expresiones mexicanas
            
            
            Intenta dar respuestas estructuradas en varios párrafos de forma breve, utilizando negritas, cursivas, bullet points, tablas...
            Si el usuario te pregunta con cualquier cosa que no esté relacionada con el evento del Foundations o información sobre el restaurante dile que no se pase de listillo, que tu creado ya ha contemplado que el usuario se vaya por las ramas
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

def invoke_chain(question, messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, json_params=None, db_name=None):
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