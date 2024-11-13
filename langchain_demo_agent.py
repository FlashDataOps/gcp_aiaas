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
from langchain import hub

load_dotenv()

def get_schema(_):
    db = af.db_connection.get_db()
    schema = db.get_table_info()
    return schema

def run_query(query):
    try:
        db = af.db_connection.get_db()
        return db.run(query)
    except Exception as e:
        return f"Error -> {e}"

def clean_query(query):
    return query.replace("```sql", "").replace("```", "").replace("[SQL:", "").replace("]", "").strip()

prompt_intent = hub.pull("intent_demo_foundations")
prompt_create_sql = hub.pull("create_sql")
prompt_fix_sql_query = hub.pull("fix_sql_query")
prompt_gen_response = hub.pull("gen_response")
prompt_general_agent_007 = hub.pull("general_agent_007")
prompt_gen_plot_agent_007 = hub.pull("gen_plot_agent_007")

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
        "gemini-1.5-flash-002":ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1",verbose=True),
        "gemini-1.5-pro-002":ChatVertexAI(model_name="gemini-1.5-pro-002",project="single-cirrus-435319-f1",verbose=True),
    }
    return llm[model_name]

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

def invoke_intent(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_intent | llm | StrOutputParser() | (lambda x: x.strip().lower())
    history = create_history(messages)
    aux = {}
    response = ""
    
    config = {
        "question": question, 
        "chat_history": history.messages, 
    }
    
    response = chain.invoke(config)


    history.add_user_message(question)
    history.add_ai_message(response)
    
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux
    
    return response
    
def invoke_create_sql(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_create_sql | llm | StrOutputParser() | (lambda x: x.replace("```", "").replace("\\n", ""))
    history = create_history(messages)
    aux = {}
    response = ""
    
    config = {
        "question": question, 
        "chat_history": history.messages, 
    }
    
    response = chain.invoke(config)

    history.add_user_message(question)
    history.add_ai_message(response)
    
    
    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux
    
    return response

def invoke_fix_sql(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, query_sql=None):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_fix_sql_query | llm | StrOutputParser()
    history = create_history(messages)
    aux = {}
    response = ""
    
    config = {
        "question": question, 
        "chat_history": history.messages,
        "query": query_sql, 
    }
    
    response = chain.invoke(config)
    
    history.add_user_message(question)
    history.add_ai_message(response)
    
    
    invoke_fix_sql.response = response
    invoke_fix_sql.history = history
    invoke_fix_sql.aux = aux
    
    return response
   
   
def invoke_gen_response(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, sql_result=None):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_gen_response | llm | StrOutputParser()    
    history = create_history(messages)
    aux = {}
    final_response = ""

    config = {
        "question": question, 
        "chat_history": history.messages,
        "sql_result": sql_result
    }

    for chunk in chain.stream(config):
        final_response+=chunk
        yield chunk
        
    history.add_user_message(question)
    history.add_ai_message(final_response)


    invoke_gen_response.response = final_response
    invoke_gen_response.history = history
    invoke_gen_response.aux = aux

def invoke_general_agent_007(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, sql_result=None):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_general_agent_007 | llm | StrOutputParser()    
    history = create_history(messages)
    aux = {}
    final_response = ""

    config = {
        "question": question, 
        "chat_history": history.messages
    }

    for chunk in chain.stream(config):
        final_response+=chunk
        yield chunk
        
    history.add_user_message(question)
    history.add_ai_message(final_response)


    invoke_general_agent_007.response = final_response
    invoke_general_agent_007.history = history
    invoke_general_agent_007.aux = aux

def invoke_gen_plot_agent_007(question, messages, sql_messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192, sql_result=None, sql_query=None):

    llm = get_model(model_name, temperature, max_tokens)
    chain = RunnablePassthrough.assign(schema=get_schema) | prompt_gen_plot_agent_007 | llm | StrOutputParser()    
    history = create_history(messages)
    aux = {}
    final_response = ""

    config = {
        "question": question, 
        "chat_history": history.messages,
        "sql_result": sql_result,
        "sql_query": sql_query
        
    }

    response = chain.invoke(config)
        
    history.add_user_message(question)
    history.add_ai_message(final_response)


    invoke_gen_plot_agent_007.response = response
    invoke_gen_plot_agent_007.history = history
    invoke_gen_plot_agent_007.aux = aux
    
    return response

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