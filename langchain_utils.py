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
        (
            "system",
            """Basándote en la tabla esquema de abajo, escribe una consulta SQL para SQLLite que pueda responder a la pregunta del usuario:
            {schema}
            
            A continuación te paso una descripción general de los datos que te vas a encontrar:
            
            Cada día se ejecuta un proceso que calcula diferentes datos para los Hoteles NH. Este día se conoce como Business Date (fecha actual). En PowerBI se puede seleccionar el Business Date de “hoy” o el del miércoles de la semana anterior por temas internos de NH. El PowerBI del que disponemos es el del 02 de Octubre de 2024. La fecha Stay Date hace referencia a la fecha de la que se muestran los datos (reservas para dicho día, ocupaciones etc).
            Si tomamos el día de hoy como Bussines Date, On-The-Books (OTB) serán las reservas activas a día de hoy para los diferentes Stay Dates, Actuals serán las ocupaciones activas hoy para cada Stay Date y Pick Up son las ocupaciones necesarias para alcanzar el Forecast.
            Los datos que hemos descargado se corresponden con aquellos de la tabla Rev-for-pre-outputs-rev desde el 01-09-2024 hasta el 30-09-2024, de todos los segmentos salvo Others y TNCD (Transient Restricted). Además, se han tomado los datos del maestro de hoteles, maestro de segmentos y maestro de métricas.

            
            A continuación te facilito un resumen de las columnas más importantes en el dataset:
            Campos y variables.
            •	Segment -> es el tipo de clientes a los que se hace referencia.
            •	Metric -> tipo de servicio al que se hace referencia.
            •	Actuals -> ocupaciones activas (en euros).
            •	OTB -> Reservas activas (en euros).
            •	Forecast -> Previsión (en euros), ingresos esperados.
            •	Local_Currency -> Moneda local.
            •	Exchange_rate_to_EUR -> relación de la moneda local con EUR.
            •	Hotel_ID -> Identificador del hotel.
            •	Business_date -> día en el que se observa la situación (actualidad).
            •	Stay_date -> Día al que hacen referencia los datos.
            •	Pick_Up -> Forecast - (Actuals + OTB). Lo que falta para llegar al forecast.
            •	Pts -> Period to stay (diferencia entre stay date y business date).
            •	Hotel_Type -> Hotel / Restaurante.
            •	Hotel_Name -> nombre del hotel.
            •	Hotel_Status -> OPEN / SIGNED.
            •	Hotel_BU -> Business Unit: BU America / BU Northern Europe / BU Southern Europe.
            •	Hotel_SubBU -> Sub Business Unit, agrupación de países.
            •	Hotel_Rooms -> nº de habitaciones del hotel.
            •	Hotel_Cluster -> Agrupación de hoteles.
            •	Hotel_Consolidate -> indica si el hotel se considera maduro (lleva años abierto).
            •	RRM -> Revenue Manager.

            Algunas métricas útiles son:
            •	Actuals_business_date -> df[df ['Stay_date']<df ['Business_date']]['Actuals'].sum(). La suma de los € de ocupaciones activas a día de hoy (Business Date).
            •	OTB_business_date  -> df['OTB'].sum(). Suma de reservas totales en €.
            •	Forecast_business_date -> df['Forecast'].sum(). Suma de predicciones en €.
            •	Total_business_date -> Actuals_business_date + OTB_business_date  + Forecast_business_date. Suma de ocupaciones activas, reservas y previsiones.
            •	Perc_expected_revenue -> (actuals_business_date + OTB_business_date) / total_business_date. Porcentaje de € sobre el total.

            
            A continuación te doy unas columnas y sus posibles valores para ayudarte a filtrar:
            •	Segment: 
                •	BUGR -> Business groups (grupos de negocio)
                •	COMP -> Complementary (complementarios)
                •	CORP -> Corporative (corporativos)
                •	CREW -> para hoteles cerca de aeropuertos, tripulación 
                •	LEGR -> Grupos de ocio (Leisure Groups)
                •	MECO -> Meetings & Conferences (reuniones y conferencias)
                •	OTHE -> Others (otros)
            •	Metric:
                •	FPB -> BKF + F&B 
                •	BKF -> Breakfast (desayuno)
                •	EVENTS -> Eventos
                •	RN -> Room Nights  
                •	RP -> Room Revenue RREV (dinero generado con las habitaciones)
                •	F&B -> Food & beverage
            
            Otras siglas y sus significados:
                •	ADR -> precio medio de la habitación
                •	TREV -> Total Revenue (dinero total generado)
                •	OREV -> Other Revenue (dinero no generado por las habitaciones)
                •	TREV -> RREV + OREV
                •	OTB -> On-the-books / Reservas

                
            Aquí te muestro algunas métricas calculadas de utilidad.
            •	Actuals_business_date -> df[df ['Stay_date']<df ['Business_date']]['Actuals'].sum(). La suma de los € de ocupaciones activas a día de hoy (Business Date).
            •	OTB_business_date  -> df['OTB'].sum(). Suma de reservas totales en €.
            •	Forecast_business_date -> df['Forecast'].sum(). Suma de predicciones en €.
            •	Total_business_date -> Actuals_business_date + OTB_business_date  + Forecast_business_date. Suma de ocupaciones activas, reservas y previsiones.
            •	Perc_expected_revenue -> (actuals_business_date + OTB_business_date) / total_business_date. Porcentaje de euros € obtenidos o reservados sobre el total estimado.
            
            Ten en cuenta que TREV = Total_business_date; RREV = Total_business_date filtrado para métrica RP; OREV = Total_business_date para todas las métricas excepto RP.
            
            Utiliza el historial para adaptar la consulta SQL. No añadas respuestas en lenguaje natural.
            
            RESPONDE ÚNICAMENTE CON CÓDIGO SQL. NO AÑADAS PALABRAS EN LENGUAJE NATURAL.
            LA CONSULTA DEBE ESTAR PREPARADA PARA SER EJECUTADA EN LA BASE DE DATOS
            
            No incluyas introducción, ni introduzcas en una lista la resupuesta
            
            """,
        ),
        ("placeholder", "{few_shots}"), 
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
            
            
            La respuesta debe tener dos seccions. Por un lado, de forma breve y concisa una frase con la respuesta a la pregunta del usuario. Por otro lado, si es posible, debes incluir un parrafo con un insight que se pueda extraer del resultado de la base de datos. No incluyas cabeceras para cada sección, directamente las información.
            No hagas referencia a la base de datos en ningún momento, ni a la consulta realizada ni a los resultados extraidos en bruto.
            Siempre muestralo de una forma bonita y ordenada, utilizando tablas o bullets points con saltos de línea a ser posible.
            UTILIZA FORMATO MARKDOWN
            RESPONDE EN ESPAÑOL DE BREVE A LA PREGUNTA DEL USUARIO
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

prompt_summary_of_the_day = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Genera un resumen de la información del día a partir de los datos de NH Hoteles. 
            Cada día se ejecuta un proceso que calcula diferentes datos para los Hoteles NH. Este día se conoce como Business Date (fecha actual). En PowerBI se puede seleccionar el Business Date de “hoy” o el del miércoles de la semana anterior por temas internos de NH. El PowerBI del que disponemos es el del 02 de Octubre de 2024. La fecha Stay Date hace referencia a la fecha de la que se muestran los datos (reservas para dicho día, ocupaciones etc).
            Si tomamos el día de hoy como Bussines Date, On-The-Books (OTB) serán las reservas activas a día de hoy para los diferentes Stay Dates, Actuals serán las ocupaciones activas hoy para cada Stay Date y Pick Up son las ocupaciones necesarias para alcanzar el Forecast.
            Los datos que hemos descargado se corresponden con aquellos de la tabla Rev-for-pre-outputs-rev desde el 01-09-2024 hasta el 30-09-2024, de todos los segmentos salvo Others y TNCD (Transient Restricted). Además, se han tomado los datos del maestro de hoteles, maestro de segmentos y maestro de métricas.
            Asegurate de no solo mostar el resultado sino también de interpretarlo y explicarlo, a ser posible de forma natural para un alto cargo de una empresa.

            Aquí tienes el esquema de la base de datos: {schema}
            
            A continuación te facilito un resumen de las columnas más importantes en el dataset:
            Campos y variables.
            •	Segment -> es el tipo de clientes a los que se hace referencia.
            •	Metric -> tipo de servicio al que se hace referencia.
            •	Actuals -> ocupaciones activas (en euros).
            •	OTB -> Reservas activas (en euros).
            •	Forecast -> Previsión (en euros), ingresos esperados.
            •	Local_Currency -> Moneda local.
            •	Exchange_rate_to_EUR -> relación de la moneda local con EUR.
            •	Hotel_ID -> Identificador del hotel.
            •	Business_date -> día en el que se observa la situación (actualidad).
            •	Stay_date -> Día al que hacen referencia los datos.
            •	Pick_Up -> Forecast - (Actuals + OTB). Lo que falta para llegar al forecast.
            •	Pts -> Period to stay (diferencia entre stay date y business date).
            •	Hotel_Type -> Hotel / Restaurante.
            •	Hotel_Name -> nombre del hotel.
            •	Hotel_Status -> OPEN / SIGNED.
            •	Hotel_BU -> Business Unit: BU America / BU Northern Europe / BU Southern Europe.
            •	Hotel_SubBU -> Sub Business Unit, agrupación de países.
            •	Hotel_Rooms -> nº de habitaciones del hotel.
            •	Hotel_Cluster -> Agrupación de hoteles.
            •	Hotel_Consolidate -> indica si el hotel se considera maduro (lleva años abierto).
            •	RRM -> Revenue Manager.

            A continuación te doy unas columnas y sus posibles valores para ayudarte a filtrar:
            •	Segment: 
                •	BUGR -> Business groups (grupos de negocio)
                •	COMP -> Complementary (complementarios)
                •	CORP -> Corporative (corporativos)
                •	CREW -> para hoteles cerca de aeropuertos, tripulación 
                •	LEGR -> Grupos de ocio (Leisure Groups)
                •	MECO -> Meetings & Conferences (reuniones y conferencias)
                •	OTHE -> Others (otros)
            •	Metric:
                •	FPB -> BKF + F&B 
                •	BKF -> Breakfast (desayuno)
                •	EVENTS -> Eventos
                •	RN -> Room Nights  
                •	RP -> Room Revenue RREV (dinero generado con las habitaciones)
                •	F&B -> Food & beverage
            
            Otras siglas y sus significados:
                •	ADR -> precio medio de la habitación
                •	TREV -> Total Revenue (dinero total generado)
                •	OREV -> Other Revenue (dinero no generado por las habitaciones)
                •	TREV -> RREV + OREV
                •	OTB -> On-the-books / Reservas
            
            PASOS A SEGUIR para generar el resumen diario:
            Asegurate de no solo mostar el resultado sino también de interpretarlo y explicarlo, a ser posible de forma natural para un alto cargo de una empresa.
            
            - Saludar e indicar la fecha del Business Date {business_date}.

            - Total Revenue: 
                Cuanto es TREV y su desglose en: Cuanto es Actuals, cuanto es OTB y cuanto es PickUp.
                Con ello decir cual es el porcentaje de completitud respecto a lo esperado.
                Datos a utilizar: {trev}
                Columnas: [Total_Actuals	Total_OTB	Total_PickUp	TREV	perc_expected_revenue]
                
            - Room Revenue:
                Cuanto es RREV y su desglose en: Cuanto es Actuals, cuanto es OTB y cuanto es PickUp.
                Con ello decir cual es el porcentaje de completitud respecto a lo esperado.
                Datos a utilizar: {rrev}
                Columnas: [Total_Actuals_RREV	Total_OTB_RREV	Total_PickUp_RREV	RREV	perc_expected_revenue_RREV]

            - Other Revenue:
                Cuanto es OREV y su desglose en: Cuanto es Actuals, cuanto es OTB y cuanto es PickUp.
                Con ello decir cual es el porcentaje de completitud respecto a lo esperado.
                Datos a utilizar: {orev}
                Columnas: [Total_Actuals_OREV	Total_OTB_OREV	Total_PickUp_OREV	OREV	perc_expected_revenue_OREV]

            - Pequeño resumen de lo anterior:
                - Cual es el porcentaje del TREV asociado a RREV y OREV.
                - Cual es el porcentaje de Actuals, OTB y Pick UP.
                - Resumen más subjetivo: si el RREV va bien / mal o similares; comparación con niveles habituales de cada punto...
                Datos a utilizar: {pctg_rrev_orev}, {pctg_actuals_OTB_pickup}
                Columnas: [TREV	RREV	OREV	perc_RREV_of_TREV	perc_OREV_of_TREV]
                Columnas: [TREV	RREV	OREV	perc_Actuals_of_TREV	perc_Actuals_of_RREV	perc_Actuals_of_OREV	perc_OTB_of_TREV	perc_OTB_of_RREV	perc_OTB_of_OREV	perc_PickUp_of_TREV	perc_PickUp_of_RREV	perc_PickUp_of_OREV]

            - Revenue por Hotel Business Unit:
                Contar para cada BU cual es el Actuals, OTB, Pick Up y Forecast (y que % supone PickUp para saber si falta mucho para lo esperado)
                Dentro de cada BU explicar brevemente como va cada Hotel Country (quizá mencionar solo el forecast y el pickup)
                Datos a utilizar: {hotelBU}
                Columnas: [Hotel_Country	SUM(Actuals)	SUM(OTB)	SUM(Pick_Up)]

            - Comparativa de forecast por Hotel Sub Business Unit (qué países están a la cabeza y a la cola).
              Datos a utilizar: {hotelSubBU}
              Columnas: [Hotel_SubBU	Total_Forecast]

            - Métricas:
                Explicar el desempeño de cada métrica
                Datos a utilizar: {metrics}
                Columnas: [Metric	Total_Actuals	Total_OTB	Total_PickUp	Total_Forecast	Perc_PickUp_to_Forecast	Perc_OTB_to_Forecast	Perc_Actuals_to_Forecast	Perc_Actuals_OTB_to_Forecast]
                """,
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

def summary_of_the_date_generation(model_name, temperature, max_tokens):
    """
    Generates a summary of the day's data with visualizations for each metric.
    """
    # Get data
    db = af.db_connection.get_db_summary_of_the_day()
    results = af.summary_of_the_day_query(db)
    (res_business_date, res_TREV, res_RREV, res_OREV, res_pctg_RREV_OREV, 
     res_pctg_actuals_OTB_PickUp, res_hotelBU, res_subBU, res_country, res_metric) = results

    # Initialize LLM and get text summary
    llm = get_model(model_name, temperature, max_tokens)
    summary_chain = prompt_summary_of_the_day | llm | StrOutputParser()
    
    config = {
        "business_date": res_business_date,
        "trev": res_TREV,
        "rrev": res_RREV,
        "orev": res_OREV,
        "pctg_rrev_orev": res_pctg_RREV_OREV,
        "pctg_actuals_OTB_pickup": res_pctg_actuals_OTB_PickUp,
        "hotelBU": res_hotelBU,
        "hotelSubBU": res_subBU,
        "country": res_country,
        "metrics": res_metric
    }
    
    initial_message = summary_chain.invoke(config)
    
    # Setup plotting chain
    plot_prompt = ChatPromptTemplate.from_messages([(
        "system", """Crea una visualización profesional usando plotly para el siguiente conjunto de datos.
                    La visualización debe ser clara, informativa y adecuada para presentaciones de negocios.
                    Asegúrate de:
                    1. Importar todas las librerías necesarias al inicio
                    2. Usar una paleta de colores profesional
                    4. No usar configuraciones de múltiples ejes (xaxis2, etc.)
                    5. No incluir fig.show()
                    6. Indicar claramente el contenido de la visualización, tanto en título como en etiquetas.
                    7. Hacer el gráfico más adecuado para cada conjunto de datos.
                    8. No inventes métricas o datos, utiliza los datos proporcionados.
                    9. No hables de ingresos, beneficios, proporciones etc, sino de las variables:
                        A continuación te facilito un resumen de las columnas más importantes en el dataset:
                        Campos y variables.
                        •	Segment -> es el tipo de clientes a los que se hace referencia.
                        •	Metric -> tipo de servicio al que se hace referencia.
                        •	Actuals -> ocupaciones activas (en euros).
                        •	OTB -> Reservas activas (en euros).
                        •	Forecast -> Previsión (en euros), ingresos esperados.
                        •	Local_Currency -> Moneda local.
                        •	Exchange_rate_to_EUR -> relación de la moneda local con EUR.
                        •	Hotel_ID -> Identificador del hotel.
                        •	Business_date -> día en el que se observa la situación (actualidad).
                        •	Stay_date -> Día al que hacen referencia los datos.
                        •	Pick_Up -> Forecast - (Actuals + OTB). Lo que falta para llegar al forecast.
                        •	Pts -> Period to stay (diferencia entre stay date y business date).
                        •	Hotel_Type -> Hotel / Restaurante.
                        •	Hotel_Name -> nombre del hotel.
                        •	Hotel_Status -> OPEN / SIGNED.
                        •	Hotel_BU -> Business Unit: BU America / BU Northern Europe / BU Southern Europe.
                        •	Hotel_SubBU -> Sub Business Unit, agrupación de países.
                        •	Hotel_Rooms -> nº de habitaciones del hotel.
                        •	Hotel_Cluster -> Agrupación de hoteles.
                        •	Hotel_Consolidate -> indica si el hotel se considera maduro (lleva años abierto).
                        •	RRM -> Revenue Manager.

                    Responde solo con código Python, sin texto adicional."""), 
        ("user", "Crea una visualización para los datos: {data}")
    ])
    
    plot_chain = RunnablePassthrough.assign(schema=get_schema)| plot_prompt | llm | StrOutputParser()
    figures = []
    
    # Define datasets to visualize
    datasets = {
        "TREV": res_TREV,
        "RREV": res_RREV,
        "OREV": res_OREV,
        "Revenue_Distribution": res_pctg_RREV_OREV,
        "Revenue_Components": res_pctg_actuals_OTB_PickUp,
        "Business_Units": res_hotelBU,
        "Sub_Business_Units": res_subBU,
        "Metrics": res_metric
    }
    
    # Generate plots
    for name, data in datasets.items():
        try:
            globals_dict = {}
            exec("import plotly.graph_objects as go\nimport plotly.express as px\nimport pandas as pd\nimport numpy as np", globals_dict)
            
            plot_code = plot_chain.invoke({"data": data}).strip()
            plot_code = plot_code.replace("```python", "").replace("```", "")
            
            exec(plot_code, globals_dict)
            
            if "fig" in globals_dict:
                figures.append({
                    "name": name,
                    "figure": globals_dict["fig"]
                })
            
        except Exception as e:
            print(f"Error generating plot for {name}: {str(e)}")
            continue

    return {
        "content": initial_message,
        "aux": {"figures": figures}
    }


    

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
            list_result = result
            if len(list_result) > 1:
                del config["schema"]
                plot_code = plot_chain.invoke(config)
                #print(plot_code)
                plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")
                exec(plot_code)
                
                aux["figure"] = eval("[fig]")
        except Exception as e:
            print(f"Error al generar el gráfico {e}")
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux
    
