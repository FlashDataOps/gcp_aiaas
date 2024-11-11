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
        "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]

# First we need a prompt that we can pass into an LLM to generate this search query
prompt_create_sql = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Basándote en la tabla esquema de abajo, escribe una consulta SQl que pueda responder a la pregunta del usuario:
            {schema}

            Este dataset contiene registros de transacciones financieras individuales de un banco. Cada fila representa una transacción en la que se detalla información sobre el origen y destino de los fondos, incluyendo la cantidad transferida, la moneda utilizada, el tipo de transacción, el dispositivo desde el cual se realizó, y otros aspectos relevantes. Este conjunto de datos es útil para analizar patrones de transacciones, realizar estudios de comportamiento financiero y detectar actividades inusuales o fraudulentas. Sobre este dataset se realizarán consultas SQL para evaluar el comportamiento transaccional de las cuentas.

            A continuación te facilito un resumen de las columnas más importantes en el dataset, su significado detallado y cualquier otra información relevante de cada columna.

                - Timestamp -> Fecha y hora en que se realizó la transacción (formato: dd/mm/yyyy hh).
                - From Bank -> Código numérico del banco emisor. Identifica la institución financiera que origina la transacción.
                - Account -> Identificador único de la cuenta desde donde se envían los fondos.
                - To Bank -> Código numérico del banco receptor. Identifica la institución financiera que recibe los fondos.
                - Account.1 -> Identificador único de la cuenta donde se reciben los fondos.
                - Amount Received -> Monto recibido en la cuenta destino. La unidad depende de la moneda especificada en "Receiving Currency".
                - Receiving Currency -> Moneda en la que se recibe el monto.
                - Amount Paid -> Monto pagado en la cuenta de origen. La unidad depende de la moneda especificada en "Payment Currency".
                - Payment Currency -> Moneda en la que se paga el monto.
                - Payment Format -> Método de pago utilizado para realizar la transacción.
                - Target -> Indicador de si la transacción está marcada como sospechosa o regular.
                - Country from -> Código del país desde donde se origina la transacción.
                - Country to -> Código del país destino de la transacción.
                - Merchant Type -> Tipo de comercio asociado a la transacción.
                - Device Used -> Dispositivo utilizado para realizar la transacción.
                - Transaction Type -> Tipo de transacción realizada.
                - IP Address -> Dirección IP del dispositivo desde el cual se realizó la transacción.
                - Distance -> Distancia aproximada entre el origen y el destino de la transacción, medida en metros.
                - Previous Transactions -> Número de transacciones previas realizadas desde la misma cuenta de origen.
                - Time Since Last Transaction -> Tiempo transcurrido desde la última transacción, medido en segundos.
                - ID -> Identificador único de la transacción.

            A continuación te doy el nombre de columnas y sus posibles valores para ayudarte a realizar filtros a la hora de hacer consultas:
                Receiving Currency:
                    - Australian Dollar -> Dólar australiano (A$)
                    - Bitcoin -> Criptomoneda Bitcoin (₿)
                    - Brazil Real -> Real brasileño (R$)
                    - Canadian Dollar -> Dólar canadiense (C$)
                    - Euro -> Euro (€)
                    - Mexican Peso -> Peso mexicano (MXN)
                    - Ruble -> Rublo ruso (₽)
                    - Rupee -> Rupia india (₹)
                    - Saudi Riyal -> Riyal saudí (SAR)
                    - Shekel -> Nuevo shekel israelí (₪)
                    - Swiss Franc -> Franco suizo (CHF)
                    - UK Pound -> Libra esterlina (£)
                    - US Dollar -> Dólar estadounidense ($)
                    - Yen -> Yen japonés (¥)
                    - Yuan -> Yuan chino (CNY)


                Payment Currency:
                    - Australian Dollar -> Dólar australiano (A$)
                    - Bitcoin -> Criptomoneda Bitcoin (₿)
                    - Brazil Real -> Real brasileño (R$)
                    - Canadian Dollar -> Dólar canadiense (C$)
                    - Euro -> Euro (€)
                    - Mexican Peso -> Peso mexicano (MXN)
                    - Ruble -> Rublo ruso (₽)
                    - Rupee -> Rupia india (₹)
                    - Saudi Riyal -> Riyal saudí (SAR)
                    - Shekel -> Nuevo shekel israelí (₪)
                    - Swiss Franc -> Franco suizo (CHF)
                    - UK Pound -> Libra esterlina (£)
                    - US Dollar -> Dólar estadounidense ($)
                    - Yen -> Yen japonés (¥)
                    - Yuan -> Yuan chino (CNY)


                Payment Format:
                    - ACH -> Transacción automatizada entre bancos.
                    - Bitcoin -> Pago realizado mediante criptomoneda Bitcoin.
                    - Cash -> Pago realizado en efectivo.
                    - Cheque -> Pago realizado mediante cheque.
                    - Credit Card -> Pago con tarjeta de crédito.
                    - Reinvestment -> Transacción de reinversión de fondos.
                    - Wire -> Transferencia bancaria electrónica.

                Target:
                    - TRUE -> Transacción marcada como sospechosa o potencialmente fraudulenta.
                    - FALSE -> Transacción regular.

                Country from:
                    - AD -> Andorra
                    - AL -> Albania
                    - AR -> Argentina
                    - BE -> Bélgica
                    - BR -> Brasil
                    - CA -> Canadá
                    - CN -> China
                    - CR -> Costa Rica
                    - CU -> Cuba
                    - DE -> Alemania
                    - ES -> España
                    - FR -> Francia
                    - GB -> Reino Unido
                    - GI -> Gibraltar
                    - GN -> Guinea
                    - IL -> Israel
                    - IQ -> Irak
                    - IR -> Irán
                    - IT -> Italia
                    - JP -> Japón
                    - KE -> Kenia
                    - KR -> Corea del Sur
                    - LU -> Luxemburgo
                    - MA -> Marruecos
                    - ME -> Montenegro
                    - US -> Estados Unidos

                Country to:
                    - AD -> Andorra
                    - AL -> Albania
                    - AR -> Argentina
                    - BE -> Bélgica
                    - BR -> Brasil
                    - CA -> Canadá
                    - CN -> China
                    - CR -> Costa Rica
                    - CU -> Cuba
                    - DE -> Alemania
                    - ES -> España
                    - FR -> Francia
                    - GB -> Reino Unido
                    - GI -> Gibraltar
                    - GN -> Guinea
                    - IL -> Israel
                    - IQ -> Irak
                    - IR -> Irán
                    - IT -> Italia
                    - JP -> Japón
                    - KE -> Kenia
                    - KR -> Corea del Sur
                    - LU -> Luxemburgo
                    - MA -> Marruecos
                    - ME -> Montenegro
                    - US -> Estados Unidos

                Merchant Type:
                    - Entertainment -> Comercio de entretenimiento.
                    - Grocery -> Tienda de comestibles.
                    - Luxury Goods -> Bienes de lujo.
                    - Retail -> Comercio minorista.

                Device Used:
                    - ATM -> Cajero automático.
                    - Desktop -> Ordenador de mesa / computadora de escritorio.
                    - Mobile -> Teléfono móvil.
                    - POS Terminal -> Terminal de punto de venta.

                Transaction Type:
                    - ATM Withdrawal -> Retiro en cajero automático.
                    - In-Person Payment -> Pago en persona.
                    - Online Purchase -> Compra en línea.
                    - Wire Transfer -> Transferencia entre cuentas.
                            
            
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


prompt_ml = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual que es capaz de llamar a un modelo de machine learning de clasificación y ofrecer el resultado al usuario.
            - Parámetros de llamada: {params}
            - Resultado de la llamada {result}
            
            Debes dar una respuesta que incluya 5 parámetros más importantes para identificar la transacción (en español, sin utilizar los nombres de las columnas) utilizados para llamar al modelo en forma de tabla y de forma clara y sencilla pero que se note que es la parte importante del mensaje el resultado obtenido. Debes dar el valor exacto que se indica en los parámetros de llamada.
            Si el resultado obtenido es False, siginifica que la transacción no ha sido identificada como alerta, si el resultado es True, significa que estamos ante una alerta que se debería revisar manualmente. En tu respuesta, no digas el valor en bruto obtenido.
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
            - ML: Si el usuario quiere realizar una llamada de inferencia a un modelo de machine learning. 
            - Consulta: Si el usuario realiza una consulta sobre un dataset con el siguiente schema: {schema}
            - Explicabilidad: Si el usuario necesita detalles sobre la explicabildiad del modelo de forma general o de un dato concreto.
            - Otro: Cualquier cosa que no tenga nada que ver con las otras tres
            
            Responde únicamente con las palabras [ML, Consulta, Explicabilidad, Otro]. En caso de no saber a que se refiere responde Otro
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
prompt_explicabilidad = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual y tu especialidad es utilizar tus capacidades de visión para responder a las preguntas del usuario.
            Tus respuestas deben ser precisas y deben estar basadas únicamente en lo que puedas observar en el documento enviado por el usuario.
            
             A continuación te facilito un resumen de las columnas más importantes en el dataset, su significado detallado y cualquier otra información relevante de cada columna.

                - Timestamp -> Fecha y hora en que se realizó la transacción (formato: dd/mm/yyyy hh).
                - From Bank -> Código numérico del banco emisor. Identifica la institución financiera que origina la transacción.
                - Account -> Identificador único de la cuenta desde donde se envían los fondos.
                - To Bank -> Código numérico del banco receptor. Identifica la institución financiera que recibe los fondos.
                - Account.1 -> Identificador único de la cuenta donde se reciben los fondos.
                - Amount Received -> Monto recibido en la cuenta destino. La unidad depende de la moneda especificada en "Receiving Currency".
                - Receiving Currency -> Moneda en la que se recibe el monto.
                - Amount Paid -> Monto pagado en la cuenta de origen. La unidad depende de la moneda especificada en "Payment Currency".
                - Payment Currency -> Moneda en la que se paga el monto.
                - Payment Format -> Método de pago utilizado para realizar la transacción.
                - Target -> Indicador de si la transacción está marcada como sospechosa o regular.
                - Country from -> Código del país desde donde se origina la transacción.
                - Country to -> Código del país destino de la transacción.
                - Merchant Type -> Tipo de comercio asociado a la transacción.
                - Device Used -> Dispositivo utilizado para realizar la transacción.
                - Transaction Type -> Tipo de transacción realizada.
                - IP Address -> Dirección IP del dispositivo desde el cual se realizó la transacción.
                - Distance -> Distancia aproximada entre el origen y el destino de la transacción, medida en metros.
                - Previous Transactions -> Número de transacciones previas realizadas desde la misma cuenta de origen.
                - Time Since Last Transaction -> Tiempo transcurrido desde la última transacción, medido en segundos.
                - ID -> Identificador único de la transacción.
                
            Tu respuesta debe explicar de forma clara una interpretación del significado de la imagen. 
            No debes hacer referencia a los valores que se observan en la imagen.
            No debes hacer referencia a la imagen que ves. Debes ofrecer una explicación al usuario como si fueras tu mismo el que conoce la información.
            Tu interpretación debe estar dirigida a una persona de negocio.
            
            Respecto al formato de la respuesta, debes estructurar tus diferentes ideas en párrafos, utilizar bullet points y remarcar en negrita las palabras más relevantes.
            Usa formato markdown.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_data}"},
                }
            ],
        ),
        
    ]
)

prompt_general = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un bot desarrollado por el Equipo 5 y participas en el concurso del Tongations. Tu tarea es ayudar al usuario a realizar cualquiera de tus tres funcionalidades:
            - ML: Si el usuario quiere realizar una llamada de inferencia a un modelo de machine learning. 
            - Consulta: Si el usuario realiza una consulta sobre un dataset.
            - Explicabilidad: Si el usuario necesita detalles sobre la explicabildiad del modelo de forma general o de un dato concreto.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),   
    ]
)

prompt_local_shap = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente virtual y tu especialidad es utilizar tus capacidades de visión para responder a las preguntas del usuario.
            Tus respuestas deben ser precisas y deben estar basadas únicamente en lo que puedas observar en el documento enviado por el usuario.
            
             A continuación te facilito un resumen de las columnas más importantes en el dataset sobre transacciones, su significado detallado y cualquier otra información relevante de cada columna.

                - Timestamp -> Fecha y hora en que se realizó la transacción (formato: dd/mm/yyyy hh).
                - From Bank -> Código numérico del banco emisor. Identifica la institución financiera que origina la transacción.
                - Account -> Identificador único de la cuenta desde donde se envían los fondos.
                - To Bank -> Código numérico del banco receptor. Identifica la institución financiera que recibe los fondos.
                - Account.1 -> Identificador único de la cuenta donde se reciben los fondos.
                - Amount Received -> Monto recibido en la cuenta destino. La unidad depende de la moneda especificada en "Receiving Currency".
                - Receiving Currency -> Moneda en la que se recibe el monto.
                - Amount Paid -> Monto pagado en la cuenta de origen. La unidad depende de la moneda especificada en "Payment Currency".
                - Payment Currency -> Moneda en la que se paga el monto.
                - Payment Format -> Método de pago utilizado para realizar la transacción.
                - Target -> Indicador de si la transacción está marcada como sospechosa o regular.
                - Country from -> Código del país desde donde se origina la transacción.
                - Country to -> Código del país destino de la transacción.
                - Merchant Type -> Tipo de comercio asociado a la transacción.
                - Device Used -> Dispositivo utilizado para realizar la transacción.
                - Transaction Type -> Tipo de transacción realizada.
                - IP Address -> Dirección IP del dispositivo desde el cual se realizó la transacción.
                - Distance -> Distancia aproximada entre el origen y el destino de la transacción, medida en metros.
                - Previous Transactions -> Número de transacciones previas realizadas desde la misma cuenta de origen.
                - Time Since Last Transaction -> Tiempo transcurrido desde la última transacción, medido en segundos.
                - ID -> Identificador único de la transacción.

            Te paso además la información sobre la transacción asociada al diagrama de la iamgen:
            
            {transaction}
            
            Tu respuesta debe explicar de forma clara una interpretación del significado de la imagen. 
            No debes hacer referencia a los valores que se observan en la imagen.
            No debes hacer referencia a la imagen que ves. Debes ofrecer una explicación al usuario como si fueras tu mismo el que conoce la información.
            Tu interpretación debe estar dirigida a una persona de negocio.
            
            Respecto al formato de la respuesta, debes estructurar tus diferentes ideas en párrafos, utilizar bullet points y remarcar en negrita las palabras más relevantes.
            Usa formato markdown.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_data}"},
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

def invoke_chain_shap(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, json_params=None, db_name=None, model_params = None, id_transaction = None):
    llm = get_model(model_name, temperature, max_tokens)
    history = create_history(messages)
    path_file = f"gs://single-cirrus-435319-f1-bucket/foundations/plots_shap/shap_local_{id_transaction}.png"
    config = {
        "input": question, 
        "chat_history": history.messages,
        "transaction": model_params,
        "image_data": path_file
    }
    
    shap_vision_local_chain = (
        prompt_local_shap
        | llm
        | StrOutputParser()
    )

    for chunk in shap_vision_local_chain.stream(config):
        yield chunk

def invoke_chain(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, json_params=None, db_name=None, model_params = None):
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
    #chain = get_custom_chain(model_name, temperature, max_tokens, json_params, db)
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
    invoke_chain.intent = res_intent
    
    if "ml" in res_intent:
        chain = prompt_ml | llm | StrOutputParser()
        ml_result = af.simulate_model_prediction(model_params)
        #exit_status_local_shap, id_transaction, fig = af.generate_and_upload_shap_local(model_params)
        #aux["shap"] = [fig]
        #print(exit_status_local_shap, id_transaction, fig)
        config["params"] = model_params    
        config["result"] = ml_result
        print(config)
    elif "explicabilidad" in res_intent:
            config["image_data"] = "gs://single-cirrus-435319-f1-bucket/foundations/shap_global.png"
            chain = prompt_explicabilidad | llm | StrOutputParser()

    elif "consulta" in res_intent:
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
        sql_history.add_user_message(question)
        #sql_history.add_ai_message(query)
        
        try:   
            result = db.run(query)
        except:
            result = f"No se ha podido ejecutar la consulta. Indica al usuario que existe un problema a la hora de realizar la consulta SQL {query} en la base de datos. Responde de forma breve"
        
        config = {
        "input": question, 
        "chat_history": history.messages, 
        "query": query,
        "response": result,
        "schema": get_schema
        }
    else:
        chain = prompt_general | llm | StrOutputParser()
        
    for chunk in chain.stream(config):
        response+=chunk
        yield chunk
    
    
    history.add_user_message(question)
    history.add_ai_message(response)
    
    if res_intent == "consulta":
        try:
            list_result = eval(result)
            if len(list_result) > 1:
                del config["schema"]
                plot_code = plot_chain.invoke(config)
                
                plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")
                exec(plot_code)
                aux["figure"] = eval("[fig]")
        except Exception as e:
            print(f"Error al generar el gráfico {e}")
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux