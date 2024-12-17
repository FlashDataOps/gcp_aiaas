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


from langchain_openai import ChatOpenAI

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
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini")
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
            - Unnamed: 0: El identificador numérico único, podemos llamarle id.
            - company: Nombre de la empresa que fabrica o distribuye el producto.
            - diet_type: Tipo de dieta para la cual está diseñado el producto.
            - diet_subtype: Subtipo más específico de la dieta.
            - national_code: Código nacional del producto, puede ser un identificador único en el país.
            - product_name: Nombre comercial del producto.
            - package_type: Tipo de empaque en el que se presenta el producto.
            - packages_per_unit: Número de paquetes por unidad de venta.
            - package_volume: Volumen del paquete, generalmente en mililitros o gramos.
            - unit_type: Tipo de unidad de medida utilizada.
            - presentation_flavor: Sabor del producto según se presenta.
            - has_multiflavor: Indica si el producto viene en múltiples sabores.
            - osmolarity: Medida de la concentración osmótica del producto.
            - caloric_density: Densidad calórica del producto, generalmente medida en calorías por mililitro o gramo.
            - product_status: Estado del producto, puede indicar si está activo, descontinuado, etc.
            - commercialization_date: Fecha en que el producto comenzó a comercializarse.
            - status_date: Fecha en que se actualizó el estado del producto.
            - diet_type_description: Descripción detallada del tipo de dieta.
            - diet_subtype_description: Descripción detallada del subtipo de dieta.
            - has_gluten: Indica si el producto contiene gluten.
            - has_lactose: Indica si el producto contiene lactosa.
            - protein_source: Fuente de proteína del producto.
            - carbohydrate_source: Fuente de carbohidratos del producto.
            - fat_source: Fuente de grasas del producto.
            - fiber_source: Fuente de fibra del producto.
            - caloric_distribution: Distribución calórica entre macronutrientes.
            - ingredients_list: Lista de ingredientes del producto.
            - calories_per_100: Calorías por cada 100 gramos o mililitros del producto.
            - fat_per_100: Gramos de grasa por cada 100 gramos o mililitros del producto.
            - sat_fat_per_100: Gramos de grasa saturada por cada 100 gramos o mililitros del producto.
            - mct_per_100: Gramos de triglicéridos de cadena media por cada 100 gramos o mililitros del producto.
            - omega_ratio_w6/w3: Relación entre ácidos grasos omega-6 y omega-3.
            - mono_fat_per_100: Gramos de grasas monoinsaturadas por cada 100 gramos o mililitros del producto.
            - poly_fat_per_100: Gramos de grasas poliinsaturadas por cada 100 gramos o mililitros del producto.
            - carbohydrates_per_100: Gramos de carbohidratos por cada 100 gramos o mililitros del producto.
            - sugar_per_100: Gramos de azúcares por cada 100 gramos o mililitros del producto.
            - fiber_per_100: Gramos de fibra por cada 100 gramos o mililitros del producto.
            - protein_per_100: Gramos de proteína por cada 100 gramos o mililitros del producto.
            - protein_equiv_per_100: Equivalente en gramos de proteína por cada 100 gramos o mililitros del producto.
            - calories_per_container: Calorías totales por contenedor.
            - fat_per_container: Gramos de grasa totales por contenedor.
            - sat_fat_per_container: Gramos de grasa saturada totales por contenedor.
            - mct_per_container: Gramos de triglicéridos de cadena media totales por contenedor.
            - omega_ratio_w6/w3_per_container: Relación de omega-6/omega-3 totales por contenedor.
            - mono_fat_per_container: Gramos de grasas monoinsaturadas totales por contenedor.
            - poly_fat_per_container: Gramos de grasas poliinsaturadas totales por contenedor.
            - carbohydrates_per_container: Gramos de carbohidratos totales por contenedor.
            - sugar_per_container: Gramos de azúcares totales por contenedor.
            - fiber_per_container: Gramos de fibra totales por contenedor.
            - protein_per_container: Gramos de proteína totales por contenedor.
            - protein_equiv_per_container: Equivalente en gramos de proteína totales por contenedor.
            - sat_fat_pct_total_calories: Porcentaje de calorías totales provenientes de grasas saturadas.
            - sat_fat_pct_total_fats: Porcentaje de grasas saturadas del total de grasas.
            - mct_pct_total_calories: Porcentaje de calorías totales provenientes de triglicéridos de cadena media.
            - mct_pct_total_fats: Porcentaje de triglicéridos de cadena media del total de grasas.
            - mono_fat_pct_total_calories: Porcentaje de calorías totales provenientes de grasas monoinsaturadas.
            - mono_fat_pct_total_fats: Porcentaje de grasas monoinsaturadas del total de grasas.
            - poly_fat_pct_total_calories: Porcentaje de calorías totales provenientes de grasas poliinsaturadas.
            - poly_fat_pct_total_fats: Porcentaje de grasas poliinsaturadas del total de grasas.
            
            A continuación te paso algunas columnas categóricas y sus posibles valores:
            - company: 
                - Aventia Pharma S.L.
                - Danone Nutricia S.R.L.
                - Fresenius Kabi España S.A.U.
                - Nestlé España S.A.
            
            - unit_type
                - ML: mililiter
                - G: gram
            
            - has_multiflavor:
                - Y: yes
                - N: no
            
            - product_status
                - ALTA
                - BAJA
            
            - has_gluten:
                - 0: no
                - 1: yes
            
            - has_lactose:
                - 0: no
                - 1: yes
            
            A continuación te paso un listado de preguntas tipo y la forma en la que deberías responderlas:
            
            1. ¿Qué código nacional tiene el producto xxx?
            El chat tiene que devolver el valor de la columna national_code para el producto que
            te pidan. Aquí hay que tener cuidado con los productos que tienen diferentes
            sabores, ya que el producto_name es el mismo. Por lo que, en los casos que haya
            más de uno, debería devolverlos todos especificando el sabor para cada uno de
            ellos. También podemos probar el caso donde pregunte ¿Qué código nacional tiene
            el producto xxx con sabor xxx? que en ese caso sólo tiene que devolver uno.
            A partir de aquí, si crees que para la demo del miércoles tenemos que realizar las preguntas
            por código nacional (identificador único) para que funcione correctamente. No nos importa
            en la demo pedir primero al chat el código. Si por nombre también lo coge bien, ¡pues ideal!
            También, dinos si has podido solucionar el tema de la memoria, y una vez tengamos el
            código y hayamos especificado el código de producto, va a ser necesario volverlo a
            introducir o será suficiente con hacerle referencia al anterior.
            
            2. ¿Dime si el producto xxx tiene lactosa/gluten?
            Debe comprobar si el campo has_lactose/has_gluten son 0 (y devolver que no tiene)
            o 1 (y devolver que sí tiene).
            
            3. ¿Qué sabores tiene el producto xxx?
            Este es el ejemplo que hemos probado antes. Tiene que poder devolver todos los
            sabores (hay uno por fila), que se especifican en la columna presentation_flavor.
            Preguntas derivadas de esta serían ¿Tiene más de un sabor el producto xxx? ¿El
            producto xxx tiene sabor Vainilla?
            
            4. ¿Dime los productos de la empresa xxx que tengan más de xx g de grasas,
            calorías…?
            Aquí hay 3 casuísticas posibles:
            - Que te lo pidan por 100ml / 100 g: que entonces hay que coger los campos que
            acaban en _per_100
            - Que te lo pidan por envase: que entonces hay que coger los campos que acaban
            en _per_container
            - Que no te especifiquen: por defecto deberían devolverse ambos en la respuesta,
            por 100ml / 100 g (que habrá que ir a buscar a la columna unit_type si son G o
            ML) y por envase (especificando la cantidad del envase package_volume +
            unit_type).
            Si es complicado que devuelva ambas, lo hablamos y especificamos uno de los
            otros dos por defecto. En el caso de que sea demasiado para mañana que se pueda
            preguntar grasas, grasas saturadas, calorías… podemos coger solo 2 y en la demo
            probamos con esos.
            
            5. Compara la información nutricional del producto xxxx y el producto yyyy en formato
            tabla.
            Consideraremos información nutricional a los siguientes campos: protein_per_100
            (llamarlo proteína) carbohydrates_per_100 (llamarlo carbohidratos), fat_per_100
            (llamarlo grasas), fiber_per_100 (llamarlo fibra) y sugar_per_100 (llamarlo azúcar).
            Debe ponerlos en una tabla para ambos productos y que así sea fácilmente
            comprable, las unidades de todo es gramos (g). Debe especificar en la respuesta
            que los valores que se proporcionan son valores cada 100 g o cada 100 ml (que
            habrá que ir a buscar a la columna unit_type si son G o ML).
            
            6. Dime el detalle de la fuente de proteínas del producto xxxx.
            Tiene que devolver la información del campo protein_source (que suele ser una o
            dos frases).
            
            7. ¿Cuál es la distribución calórica del producto xxx?
            Cuando se pregunte por información nutricional o distribución calórica se debe
            devolver el campo caloric_distribution. Esta variable puede tener dos formatos
            “número 1 / número 2 / número 3 / número 4” o “número 1 / número 2 / número 3 /
            número 4”. Debe devolver:
            - Proteínas: número 1
            - Carbohidratos: número 2
            - Grasas: Número 3
            - Fibra: Número 4 (que puede no aparecer en ocasiones)           
            
            Como nota importante un producto se identifica por los siguientes campos: 
            - product_name
            - presentation_flavor
            - package_type
            - packages_per_unit
            
            Todas las preguntas sobre un producto deben ser respondidas teniendo en cuenta el anterior punto agrupandolo por tipo de envase, tamaño y sabor.
            Por lo tanto un producto deberia consultarse como  national_code, product_name, package_type, packages_per_unit, presentation_flavor.
            Por ejemplo:
            Pregunta: ¿Cuántas calorias por envase tiene FRESUBIN THICKENED?
            Respuesta: SELECT national_code, product_name, package_type, packages_per_unit, presentation_flavor, calories_per_container
                       FROM nestle_db
                       WHERE product_name = 'FRESUBIN THICKENED';

                   
            Utiliza el historial para adaptar la consulta SQL. No añadas respuestas en lenguaje natural.
            
            RESPONDE ÚNICAMENTE CON CÓDIGO SQL. NO AÑADAS PALABRAS EN LENGUAJE NATURAL.
            LA CONSULTA DEBE ESTAR PREPARADA PARA SER EJECUTADA EN LA BASE DE DATOS
            
            No incluyas introducción, ni introduzcas en una lista la resupuesta
            
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
            """Basándote en los siguientes datos, responde en lenguaje natural. Minimiza el uso de "bullet points", ya que el texto deberá ser leído por un bot de voz.
            Es muy importante (lo más importante), que si se tienen muchos datos no se mencionen todos, sino que se haga un resumen general obteniendo siempre insights.
            Si se pregunta por una cuestión muy específica, se puede responder con más detalle, pero siempre intentando resumir.
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
            
            - Evita hacer tablas comparativas. Se original y muestra una visualizacion que acompañe a una tabla.
            
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
            - Consulta: Si el usuario realiza una consulta sobre un dataset de productos y sus fichas con valores nutricionales con el siguiente schema: {schema}
            - Otro: Cualquier cosa que no tenga nada que ver con realizar una consulta a los datos de Nestlé
            
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
            """Eres un asistente que trabaja en Nestlé. Tu tarea es ayudar al usuario a entender la información del modelo de datos de Nestlé.
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
        "chat_history": sql_history.messages
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
            print("RESULTADO ANTES", result)
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
                print(plot_code)
                plot_code = plot_code.replace("```python", "").replace("```", "").replace("fig.show()", "")
                exec(plot_code)
                
                aux["figure_p"] = eval("[fig]")
        except Exception as e:
            print(f"Error al generar el gráfico {e}")
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux
    
