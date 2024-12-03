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
            """Based on the schema table:
                {schema}

                Write an SQL query for SQLite that can answer the user's question.
                Do not include an introduction or format the response in a list; only provide the SQL query.
            """,
        ),
        ("user", "{input}")
    ]
)

prompt_create_sql_response = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        (
            "system",
            """Based on the following data, respond in natural language.
                You must use the following data to answer the user's question:
                {schema}

                Question: {input}
                SQL Query: {query}
                Answer: {response}
                Use markdown formatting and include visual elements to make the response more engaging, such as tables when necessary.
                Always use English as the text language.
            """,
        ),
        ("user", "{input}"),
    ]
)

prompt_custom_chart = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Respond only with Python code
            You must use the following data to write Python Plotly code that represents the answer obtained with the following query:

            SQL Query: {query}
            Answer: {response}
            ONLY INCLUDE PYTHON CODE IN YOUR RESPONSE. DO NOT INCLUDE NATURAL LANGUAGE TO INTRODUCE YOUR ANSWER.
            MAKE THE CHART BEAUTIFUL AND VISUALLY APPEALING. IT SHOULD BE READY TO PRESENT TO A VERY IMPORTANT CLIENT.

            ENSURE THAT THE RESPONSE CONTAINS ONLY PYTHON CODE.
            """,
        ),
        ("user", "{input}"),
    ]
)


prompt_intent = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        (
            "system",
            """Your task is to determine the user's intent based on their message. The possibilities are:
                
                Query: If the user is making a query about contractual data with the following schema: {schema}. The user must specifically say: "in the dataset..." or "in our data...". If not, respond with Other.
                Other: If the user wants to generate text based on previous knlowledge, or if they want to get information about a text or field of knowledge. The user doesn't mention the data.
                
                Respond only with the words [Query, Other]. Unless really clear, respond with Other.
            """,
        ),
        ("user", "{input}"),
    ]
)

prompt_general = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        (
            "system",
            """You are an assistant working at FCC constructions. Your task is to help the user understand the information in the database.
                You can perform the following task:

                Query: If the user makes a query about a dataset, you can generate a chart and text. The dataset schema is as follows: {schema}
                
                These are the documents available for now:
                
                ### First
                CONSTRUCTION CONTRACT
                This Construction Contract ("Contract") is entered into on the date specified below between the Ministry of Infrastructure and Transport ("Client") and Global Construction Inc. ("Contractor"). The purpose of this Contract is to formalize the agreement between the parties for the construction of a roadway project located in Zaragoza, Aragón, Spain. Both parties agree to the terms and conditions outlined herein, which define their rights, responsibilities, and obligations to ensure the successful completion of the project.
                Contract Number: CON-2024-001 Start Date: 2024-12-10 Estimated Completion Date: 2026-03-15
                Contracting Parties
                Client Name: Ministry of Infrastructure and Transport Client Tax ID: A12345678
                Contractor Name: Global Construction Inc. Contractor Tax ID: B98765432
                Project Information
                Type of Construction: Roadway Location: Zaragoza, Aragón, Spain Approved Budget: €45,000,000 Total Duration (days): 460 Project Status: Planned
                Technical Details
                Project Dimensions:
                •
                Length: 25 km
                •
                Width: 15 m
                •
                Height/Depth: 3 m
                Main Materials Used: Concrete, asphalt, reinforced steel, aluminum signage
                Project Description: The project entails the construction of a new expressway connecting Zaragoza with nearby towns to reduce travel time and improve regional connectivity. It will include four lanes (two in each direction), safety shoulders, rest areas, and a bridge over the Ebro River.
                Financial Clauses
                Payment Terms:
                •
                20% advance payment at the start of the project.
                •
                60% in monthly installments based on progress certified by technical supervision.
                •
                20% upon final delivery and approval by the client.
                Delay Penalties:
                •
                Penalty of €25,000 for each week of unjustified delay.
                Schedule and Milestones
                MILESTONE
                START DATE
                ESTIMATED COMPLETION DATE
                MILESTONE DESCRIPTION TOPOGRAPHIC STUDIES 2024-12-15 2025-02-01 Terrain evaluation and design adjustments.
                EARTHWORKS
                2025-02-15
                2025-07-30
                Initial land preparation. FOUNDATION PLACEMENT 2025-08-01 2025-12-31 Structural base for the roadway.
                PAVING
                2026-01-15
                2026-02-28
                Asphalt laying along the route. SIGNAGE INSTALLATION 2026-03-01 2026-03-10 Installation of safety signs and signals.
                
                ### SECOND
                CONSTRUCTION CONTRACT
                Contract Number: CON-2024-002 Start Date: 2024-11-01 Estimated Completion Date: 2025-12-20
                This Construction Contract ("Contract") is entered into on the date specified below between the City of Valencia Public Works Department ("Client") and Apex Infrastructure Ltd. ("Contractor"). The purpose of this Contract is to formalize the agreement between the parties for the construction of a pedestrian suspension bridge located in Valencia, Spain. Both parties agree to the terms and conditions outlined herein, which define their rights, responsibilities, and obligations to ensure the successful completion of the project.
                Contracting Parties
                Client Name: City of Valencia Public Works Department Client Tax ID: C12398765
                Contractor Name: Apex Infrastructure Ltd. Contractor Tax ID: D45612389
                Project Information
                Type of Construction: Pedestrian Bridge Location: Valencia, Spain Approved Budget: €18,500,000 Total Duration (days): 415 Project Status: Planned
                Technical Details
                Project Dimensions:
                •
                Length: 180 m
                •
                Width: 6 m
                •
                Height/Depth: 25 m
                Main Materials Used: Reinforced concrete, structural steel, tempered glass for railings
                Project Description: The project involves the construction of a modern pedestrian suspension bridge crossing the Turia River in Valencia. The bridge will feature a sleek, contemporary design, incorporating energy-efficient lighting and eco-friendly materials. It will connect key recreational areas on both sides of the river, promoting accessibility and enhancing the urban landscape.
                Financial Clauses
                Payment Terms:
                •
                25% advance payment upon contract signing.
                •
                50% in monthly installments based on progress verified by independent inspectors.
                •
                25% upon final inspection and project approval.
                Delay Penalties:
                •
                Penalty of €20,000 for every week of unjustified delay.
                Schedule and Milestones
                MILESTONE
                START DATE
                ESTIMATED COMPLETION DATE
                MILESTONE DESCRIPTION SITE PREPARATION 2024-11-05 2025-01-15 Clearing, surveying, and foundation preparation.
                FOUNDATION WORK
                2025-01-20
                2025-04-30
                Laying and securing the bridge foundation. SUSPENSION SYSTEM SETUP 2025-05-10 2025-09-20 Installation of cables and main structure.
                DECK CONSTRUCTION
                2025-10-01
                2025-11-30
                Building the pedestrian walkway and railings. FINAL TOUCHES AND INSPECTION 2025-12-01 2025-12-15 Adding lighting, testing, and inspection.
                
                Use Markdown to format the answer and include tables if necessary.
            """,
        ),
        
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
    
    if "uery" in res_intent:
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
        print("Executing query...")
        flag_correct_query = False
        try:   
            result = db.run(query)
            #print("RESULTADO ANTES", result)
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
    
    if "uery" in res_intent and flag_correct_query == True:
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
    
