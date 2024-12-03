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
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

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
prompt_extraer_campos_foto = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Te voy a pasar una imagen, que debería ser una foto carné de un alumno. Necesito que me confirmes si cumple o no los criterios de calidad: 
            
                - El fondo debe ser blanco, amarillento o neutro (no es necesario que sea puro, puede tener algún degradado,siempre que no sea excesivo) 
                - No puede ser un fondo con textura (por ejemplo, una pared de ladrillos), pero sí puede ser un fondo con textura muy suave (por ejemplo, una pared de hormigón) 
                - El rostro debe estar en color, no en blanco y negro - El alumno tiene que estar de frente- Se le tiene que ver la cara y los hombros 
                - No debe ser un documento (DNI, pasaporte, tarjeta...) 
                - LA foto debe estar libre de manchas
                - La cara se debe reconocer. Puede tener sombras, siempre que se le reconozca la cara 
                - No puede llevar gafas de sol, pañuelo ni gorro: elementos que tapen la cara (pero sí aros, joyería o pañuelos si no impiden el reconocimiento). El cuello puede estar cubierto 
                - No puede estar demasiado borrosa (sí puede estar un poco borrosa) 
                - La cara debe ser el centro de la imagen: un folio A4 con un escaneo de la foto carné en una esquina o en el centro en pequeñito. 
            
            Cíñete a estos criterios, no te inventes los tuyos propios.
            Si cumple los criterios, la salida debe ser 'Foto correcta'. 
            Si no cumple algún criterio, la salida debe ser 'Foto incorrecta' o 'Revisar' (si es una foto incorrecta pero es foto carné e incumple algún criterio 'menor') y enumerar en un listado los criterios que no cumple. No seas estricto: se deben cumplir los criteros, pero no es para temas oficiales, por lo que podemos flexibilizar los criterios. 
            El sujeto puede estar sonriendo, no hace falta que tenga expresión neutra. 
            
            ### FORMATO DE RESPUESTA ###
            El resultado debe ser escrito en la siguiente plantilla en formato markdown. Si la foto es correcta, no incluyas puntos a revisar:
            
            Estado de la foto: (Correcto o Incorrecto) Si es correcta esa palabra debe ser verde, si es incorrecto debe ser rojo: <p style="color: green;">Correcto</p> o <p style="color: red;">Incorrecto</p>

            Puntos a revisar:
            [Si la foto es incorrecta, escribir con bullet points las observaciones]
            - Idea 1
            - Idea 2 
            """,
        ),
        ("user", "Extrae de forma rigurosa los campos de la siguiente imagen"),
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_data}"},
                }
            ],
        )
    ]
)

prompt_extraer_campos_ficha = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Te voy a pasar un documento que corresponde con una ficha de admisión de la Universidad Francisco de Vitoria (UFV). Debes extraer los siguientes campos:
            
            - Nombre: Nombre de pila del estudiante.
            - Apellidos: Apellidos del estudiante.
            - Mail: Dirección de correo electrónico del estudiante.
            - Movil: Número de teléfono móvil del estudiante.
            - Colegio: Nombre del colegio o instituto de procedencia del estudiante.
            - Curso_Actual: Curso académico que el estudiante está cursando actualmente.
            - Nota_Media: Nota media del estudiante en su colegio.
            - Ciudad: Ciudad de residencia del estudiante.
            - Provincia: Provincia de residencia del estudiante.
            - Primera_Opcion: Primera opción de centro o programa de estudios del estudiante. Corresponde con el grado que esté marcado con un 1 en las imagenes que te he pasado.
            - Opciones_Secundarias: Otras opciones de centros o programas de estudios del estudiante. Corresponde con los grados que estén marcados con una X en las imagenes que te he pasado .
     
            Si no encuentras añgún campo debes rellenarlo con "N/A".
            Debes ser riguroso con la extracción de campos y no inventarte ningún dato
            
            #### FORMATO DE RESPUESTA ####
            
            Escribe los nombres de los campos en lenguaje natural.
            Ofrece una tabla con los siguientes campos:
            
            - Nombre
            - Apellidos
            - Mail
            - Movil
            - Colegio
            - Curso Actual
            - Nota Media
            - Ciudad 
            - Provincia
            
            Ten en cuenta que el valor del campo Opciones Secundarias puede tomar varios valores. En la columna de valor pon estos valores dentro de la tabla en bullet points en formato markdown.
            Las tablas y los bullet points deben estar en formato markdown y centrado.
            """,
        ),
        ("user", "No hagas referencia a las instruciones que te he dado, únicamente debes extraer la información de los siguientes 7 documentos que te voy a enviar y colocala en tablas con columna de campo y valor:"),
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{doc_pdf}"},
                }
            ],
        ),
        ("user", "Los datos de primera opción (marcada con 1) y opciones secundarias (marcada con X) los puedes obtener de las siguientes imagenes a la izquierda de cada grado universitario. Recuerda añadir solo grados universitarios:"),
        MessagesPlaceholder("image_data"),
    ]
)
prompt_extraer_campos_ficha_grado = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            En la imagen (compuesta por 3 columnas de grados universitarios) tiene que haber un grado que este marcado con un 1. También debes decirme cuales tienen una X. 

            - Solo puede haber un grado con un 1.
            - Los grados pueden tener el valor de 1, X o en blanco (solo uno a la vez)
            Analiza muy bien tu respuesta. 

            Podrias decirme cuales son?
            """,
        ),
        ("user", "Extrae la información de forma rigurosa"),
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_grado}"},
                }
            ],
        ),
    ]
)

prompt_extraer_campos_ficha_unificar = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Dadas las siguientes respuestas: 
            
            {respuestas}
            
            Te paso el documento oficial de referencia en una imagen
            
            Necesito que ofrezcas una respuesta unificada en la que solo haya un grado con 1 y el resto con x. Con este me refiero a que debes decidir cual es la primera opción y cuales son las opciones ecundarias. 
            SOLO PUEDE HABER UNA PRIMERA OPCIÓN.
            
            Tu formato de respuesta debe ser una tabla con los siguientes campos:
            
            - Primera Opción
            - Opciones Secundarias
            
            Ten en cuenta que el valor del campo Opciones Secundarias puede tomar varios valores. En la columna de valor pon estos valores dentro de la tabla en bullet points en formato markdown.
            Las tablas y los bullet points deben estar en formato markdown y centrado.
            """,
        ),
        ("user", "Ofrece tu respuesta de forma rigurosa"),
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{original_doc}"},
                }
            ],
        ),
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

def invoke_extraer_campos_ficha_mobile(model_name="gemini-1.5-flash-002", temperature=0.2, max_tokens=8192, doc_pdf="", image_data=[]):
    llm = get_model(model_name, temperature, max_tokens)
    chain = prompt_extraer_campos_ficha | llm | StrOutputParser()
    response = ""
    list_images = []
    print("type(image_data)", type(image_data), len(image_data))
    for image in image_data:
        #format_image = base64.b64encode(image).decode("utf-8")
        list_images.append(
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
            ],
        )
        )

    config = {
        "doc_pdf": doc_pdf,
    #    "image_data": list_images
        }
    
    response += chain.invoke(config)
    
    respuestas_grado = []
    chain_imagen = prompt_extraer_campos_ficha_grado | llm | StrOutputParser()
    for image in image_data:
        config = {
        "image_grado": image
        }
        respuestas_grado.append(chain_imagen.invoke(config))
        
    chain_unificar = prompt_extraer_campos_ficha_unificar | llm | StrOutputParser()
    config = {
        "respuestas": respuestas_grado,
        "original_doc": doc_pdf
    }
    response += chain_unificar.invoke(config)
    return response

def invoke_extraer_campos_ficha(model_name="gemini-1.5-pro-002", temperature=0.2, max_tokens=8192, doc_pdf="", image_data=[]):
    llm = get_model(model_name, temperature, max_tokens)
    chain = prompt_extraer_campos_ficha | llm | StrOutputParser()
    response = ""
    list_images = []
    print("type(image_data)", type(image_data), len(image_data))
    for image in image_data:
        #format_image = base64.b64encode(image).decode("utf-8")
        list_images.append(
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
            ],
        )
        )
            
    config = {
        "doc_pdf": doc_pdf,
        "image_data": list_images
        }
    
    res = chain.invoke(config)
    return res

def invoke_extraer_campos_foto(model_name="gemini-1.5-flash-002", temperature=0, max_tokens=128000, image_data=""):
    llm = get_model(model_name, temperature, max_tokens)
    chain = prompt_extraer_campos_foto | llm | StrOutputParser()
    response = ""
    
    config = {
        "image_data": image_data
        }
    
    res = chain.invoke(config)
    return res
        
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
    
