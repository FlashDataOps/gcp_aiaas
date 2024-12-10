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
            """Based on the schema table:
                {schema}

                Write an SQL query for SQLite that can answer the user's question.
                Do not include an introduction or format the response in a list; only provide the SQL query.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ]
)

prompt_create_sql_response = ChatPromptTemplate.from_messages(
    [
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
        ("placeholder", "{chat_history}"),
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
            Create the most appropriate plot for the data provided (Answer).
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
        
        (
            "system",
            """Your task is to determine the user's intent based on their message. The possibilities are:
                
                Query: If the user is making a query about contractual data with the following schema: {schema}. The user must specifically say: "in the dataset..." or "in our data...". If not, respond with Other.
                Other: If the user wants to generate text based on previous knlowledge, or if they want to get information about a text or field of knowledge. The user doesn't mention the data.
                
                Respond only with the words [Query, Other]. Unless really clear, respond with Other.
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
            """You are an assistant working at FCC constructions. Your task is to help the user understand the information in the database.
                You can perform the following task:

                Query: If the user makes a query about a dataset, you can generate a chart and text. The dataset schema is as follows: {schema}
                
                These are the documents available for now:
                
                ## FIRST
                LEASING AGREEMENT
                This leasing agreement is entered into on December 10, 2024, in Madrid, Spain, by and
                between the following parties:
                Inmobiliaria Madrid Centro S.L., hereinafter referred to as the “Lessor,” with its registered
                office at Calle Gran Vía 45, Madrid, 28013, represented by Laura Martínez López in her capacity
                as Commercial Director, and Boutique Style S.L., hereinafter referred to as the “Lessee,” with
                its registered office at Calle Serrano 22, Madrid, 28001, represented by Carlos Fernández Díaz,
                acting as General Manager.
                Both parties, recognizing their full legal capacity to enter into this agreement, agree to the terms
                outlined below regarding the leasing of certain properties for commercial use.
                1. Description of the Premises
                The properties subject to this agreement consist of two units located in central Madrid. The first
                unit, with a total area of 120 square meters, is situated at Plaza Mayor 5, Madrid, 28012. The
                second unit, measuring 80 square meters, is located at Calle Arenal 8, Madrid, 28013.
                Together, the premises provide a combined area of 200 square meters. The Lessee agrees to
                use these properties exclusively for the purposes defined in this agreement and in compliance
                with all relevant regulations.
                2. Rent and Payment Terms
                The Lessee shall pay the Lessor a fixed monthly rent of €4,000, which must be paid no later than
                the fifth calendar day of each month via bank transfer to the account designated by the Lessor.
                In addition to the fixed rent, the Lessee agrees to pay a variable rent equivalent to 5% of net
                monthly sales generated within the premises. This variable rent shall be calculated based on
                gross income, excluding applicable taxes, and settled monthly in arrears.
                3. Common Expenses
                The Lessee is responsible for contributing to the maintenance and operational costs associated
                with the leased premises. This includes a monthly fee of €200 to cover the maintenance of
                common areas within the building, as well as all utility expenses, such as water and electricity,
                based on actual consumption recorded by individual meters. Furthermore, the Lessee shall pay
                €100 per month for additional cleaning services provided by the Lessor to ensure the upkeep of
                the premises.
                4. Term and Renewal
                The initial term of this agreement is established as three (3) years, commencing on January 1,
                2025, and ending on December 31, 2027. Upon the expiration of the initial term, the agreement
                will be automatically renewed for successive periods of one (1) year, unless either party notifies
                the other of their intention not to renew, with at least two months’ written notice prior to the
                expiration of the current term.
                5. Termination
                This agreement may be terminated under the following conditions:
                • Either party may terminate the agreement by providing three months’ prior written
                notice. In such a case, the Lessee shall pay a penalty equivalent to one month of fixed
                rent.
                • The Lessor reserves the right to terminate the agreement immediately if the Lessee fails
                to pay rent for two consecutive months. Notice of such termination shall be provided in
                writing.
                6. Obligations of the Lessee
                The Lessee agrees to maintain the premises in good condition throughout the lease term and to
                promptly address any damage caused during their occupancy. The Lessee shall comply with all
                building rules and regulations and ensure that the premises are used exclusively for lawful
                purposes.
                Additionally, the Lessee shall provide a security deposit equivalent to two months’ fixed rent.
                This deposit will be retained by the Lessor as a guarantee and will be refunded to the Lessee at
                the end of the lease term, subject to any necessary deductions for damages or unpaid amounts.
                7. Final Provisions
                Both parties agree that any amendments or modifications to this agreement must be made in
                writing and signed by both the Lessor and the Lessee. This agreement is governed by the laws of
                Spain, and any disputes arising from its interpretation or execution shall be resolved by the
                courts of Madrid.
                In witness whereof, both parties affix their signatures below to confirm their understanding and
                acceptance of the terms outlined in this agreement.
                Lessor:
                Name: Laura Martínez López
                Title: Commercial Director
                Signature: ____________________________
                Lessee:
                Name: Carlos Fernández Díaz
                Title: General Manager
                Signature: ____________________________
                Date: December 10, 2024
                
                
                ### SECOND
                LEASING AGREEMENT
                This leasing agreement is entered into on December 10, 2024, in Barcelona, Spain, by and
                between the following parties:
                Catalonia Properties S.L., hereinafter referred to as the “Lessor,” with its registered office at
                Avenida Diagonal 100, Barcelona, 08019, represented by Marta Sánchez Rovira in her capacity
                as Property Manager, and Mediterranean Retail S.A., hereinafter referred to as the “Lessee,”
                with its registered office at Passeig de Gràcia 20, Barcelona, 08007, represented by Javier Ortiz
                Pérez, acting as Chief Operating Officer.
                Both parties acknowledge their legal capacity to enter into this agreement and agree to the
                terms outlined below regarding the leasing of certain premises for commercial purposes.
                1. Description of the Premises
                The properties covered by this agreement include two retail spaces located in Barcelona. The
                first unit, with a total area of 150 square meters, is situated at Plaça Catalunya 2, Barcelona,
                08002. The second unit, measuring 100 square meters, is located at Carrer de Mallorca 50,
                Barcelona, 08008. Together, the leased spaces provide a combined area of 250 square meters.
                The Lessee agrees to utilize these spaces exclusively for commercial retail purposes.
                2. Rent and Payment Terms
                The Lessee agrees to pay the Lessor a fixed monthly rent of €7,500, which must be paid on or
                before the tenth calendar day of each month via bank transfer to the Lessor's designated
                account. In addition to the fixed rent, the Lessee shall pay a variable rent equal to 8% of
                monthly net sales generated within the leased premises. This variable rent shall be calculated
                and settled on a quarterly basis.
                3. Common Expenses
                The Lessee is responsible for the following additional expenses related to the operation and
                maintenance of the premises:
                • A contribution of €300 per month for common area maintenance.
                • Payment for utilities, including water, electricity, and waste management, as invoiced
                based on actual consumption.
                • An additional charge of €150 per month for security services provided by the Lessor.
                4. Term and Renewal
                The initial term of this agreement is established as five (5) years, commencing on February 1,
                2025, and ending on January 31, 2030. At the conclusion of the initial term, the agreement shall
                automatically renew for additional two-year periods unless either party provides written notice
                of termination at least three months prior to the expiration of the current term.
                5. Termination
                This agreement may be terminated under the following conditions:
                • By either party, with a minimum of six months’ written notice, without penalties.
                • Immediately by the Lessor, in the event that the Lessee defaults on three consecutive
                monthly payments, breaches the terms of use, or causes significant damage to the
                premises.
                6. Obligations of the Lessee
                The Lessee agrees to:
                • Maintain the premises in good and tenantable condition.
                • Adhere to all municipal and building regulations related to the use of the property.
                • Provide a security deposit equal to three months’ fixed rent, refundable at the end of
                the agreement, minus any deductions for damages or unpaid charges.
                7. Dispute Resolution
                Any disputes arising from the interpretation or execution of this agreement shall be governed by
                Spanish law and submitted to the jurisdiction of the courts of Barcelona.
                8. Miscellaneous
                This agreement represents the entire understanding between the Lessor and Lessee. Any
                modifications must be in writing and signed by both parties.
                Signatures
                Lessor:
                Name: Marta Sánchez Rovira
                Title: Property Manager
                Signature: ____________________________
                Lessee:
                Name: Javier Ortiz Pérez
                Title: Chief Operating Officer
                Signature: ____________________________
                Date: December 10, 2024
                
                Use Markdown to format the answer and include tables if necessary.
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


def invoke_chain(question, messages, sql_messages, model_name="gemini-1.5-flash-002", temperature=0, max_tokens=8192):
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
    
