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
        "llama-3.3-70b-versatile": ChatGroq(temperature=temperature,model_name="llama-3.3-70b-versatile", max_tokens=max_tokens),
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
                Always use English as the text language. Show the query in the answer in SQL format.
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
            You must use the following data to write Python Plotly code that represents the answer obtained with the following query:

            SQL Query: {query}
            Answer: {response}
            Create the most appropriate plot for the data provided (Answer), make sure the code is correct and complete.
            ONLY INCLUDE PYTHON CODE IN YOUR RESPONSE. DO NOT INCLUDE NATURAL LANGUAGE TO INTRODUCE YOUR ANSWER.
            MAKE THE CHART BEAUTIFUL AND VISUALLY APPEALING. IT SHOULD BE READY TO PRESENT TO A VERY IMPORTANT CLIENT, WHOSE INFLUENCE IS HUGE IN OUR REVENUE.
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
                
                #### Account Information - Account Number 01234
                Dear Customer,  Houston Waste Solutions an FCC Environmental of Texas Company appreciates your recent order for an open top container.  We are confident you will find our 
                services both accommodating and efficient. As with any type of disposal system, there are certain types of restrictions and guidelines that must be observed to 
                eliminate safety hazards and equipment damage. These are outlined as follows:   
                LOADING RESTRICTIONS: 
                • 20yd ONLY brick, concrete, dirt, and metal material may only fill half the container and must be evenly distributed in container. 
                • Debris must not exceed the top of the container so that the drivers can properly tarp the container before hauling. 
                • Do not drop heavy items into the containers from extreme heights, as this will damage the floor and structure of the container. 
                • The customer is expected to pay for repairs resulting from unnecessary and negligent damage to the container while in possession of the container. 
                • The backdoor must be closed before the driver can lift the container. 
                • The customer is responsible for fines and penalties resulting from overweight containers.  The maximum weight is 10 tons. 
                • No tires or hazardous waste is acceptable in the landfill/transfer station. 
                • No white goods (appliances) are acceptable in the landfill/transfer station. 
                • Mattresses or other bulky items may result in additional fees from the landfill/transfer station.   
                SERVICE GUIDELINES: 
                • Please call our office at (281) 999-0030 for any service needs.  All service requests should be honored by the next business day if called in by 2 pm. 
                Emergency pulls may result in additional fees. 
                • Please make sure that there are no obstacles blocking the approach to the container when our driver arrives to service.  A blocked container may result in a 
                trip  charge. 
                • HWS/FCC reserves the right to delay or postpone service due to hazardous conditions caused by inclement weather. 
                • HWS/FCC reserves the right to place container in a safe location if the specified location is deemed hazardous or unsafe by the driver. 
                • HWS/FCC cannot be held responsible for damage to asphalt, concrete or landscaping that is caused due to the weight of the container, weather 
                conditions or normal truck movements, especially residential.          
                Customer INT  766
                
                
                TERMS: 
                • Due to the investment in equipment and hiring employees as undertaken by HWS/FCC to fulfill our commitment to its customers, in turn the customer 
                acknowledges this agreement is for the entire length of the project. 
                • Once again, we appreciate your business and look forward to a mutually beneficial relationship on all your projects. 
                • The customer assumes full responsibility for the care and custody of HWS/FCC Environmental’s equipment while it is located on their construction site. In the 
                event the equipment is damaged, including but not limited to damage caused by fire, contact with other equipment, or theft, the customer will be liable for the 
                cost of repair or replacement. This responsibility remains in effect from the time the equipment is delivered to the site until it is picked up by FCC 
                Environmental. 
                HARD TO HANDLE RATES: 
                • Tree stumps and root balls - $150.00 each 
                • Tires - $100 per car tire, $250 per Tractor or 18-wheeler tire 
                • Refrigerator – Will be reloaded and returned to customer and Customer will be charged an additional haul fee. 
                • Paint/Cleaning Products/Hazardous Material – Will be reloaded and returned to customer and Customer will be charged an additional haul fee. 
                • Prescriptions/Needles/Medical Waste – Will be reloaded and returned to customer and Customer will be charged an additional haul fee.
                
                
                #### Account Information - Account Number 01235
                1. SERVICES. Customer grants to Company the exclusive right, and Company through itself and its Affiliates, subsidiaries and related entities shall furnish Equipment and Services, 
                as set forth on page 1 (the “Service Summary”), to collect and dispose of and/or recycle all of Customer’s Waste Materials (collectively, the “Services”) at Customer’s Service Location(s), subject 
                to the terms and conditions contained herein (collectively, the “Agreement”). This Agreement shall remain valid and enforceable with respect to the Services in the event Customer changes its 
                Service  Location(s),  and  Company  shall  maintain  the  right  to  collect  Waste  Materials  at  Customer’s  new  service  location(s)  if  such  location(s)  is  within  Company’s  service  area.  Customer  
                represents and warrants that the materials to be collected under this Agreement shall be only “Waste Materials” as defined herein. For purposes of this Agreement, “Waste Materials” shall mean 
                all non-hazardous solid waste, organic waste, and if applicable, recyclable materials generated by Customer or at Customer’s Service Location(s). Waste Materials excludes, and Customer agrees 
                not  to  deposit  or  permit  the  deposit  for  collection  of  (i)  any  waste  tires,  (ii)  bio-hazardous,  biomedical,  corrosive,  flammable,  explosive,  infectious,  radioactive,  volatile,  regulated  medical  or  
                hazardous  waste,  toxic  substance  or  material,  as  defined  by,  characterized  or  listed  under  applicable  federal,  state,  or  local  laws  or  regulations,  (iii)  any  other  items  or  material  prohibited  by  
                federal,  state  or  local  laws  or  regulations,  or  that  could  adversely  affect  the  operation  or  useful  life  of  Company’s  equipment  or  facilities,  or  (iv)  waste  reasonably  deemed  unacceptable  by 
                Company (collectively, “Excluded Materials”). Title to Customer’s Waste Materials is transferred to Company at the time of Company’s receipt or collection. Title to and liability for Excluded 
                Materials shall always remain with Customer at all times.  2. TERM. The Initial Term, commencing on the Start Date, and any subsequent Renewal Term of this Agreement (collectively, the “Term”) is set forth in the Service Summary.  If no
                Term is listed on the Service Summary, the Term shall be thirty-six (36) months.  Unless otherwise specified on the Service Summary, at the end of the Initial Term and any subsequent Renewal 
                Term, the Term shall automatically renew for an additional twelve (12) months (a “Renewal Term”) at the then current Service levels and applicable Charges, unless either party gives written 
                notice to the other party of its intent to terminate at least ninety (90) days, but no more than one hundred twenty (120) days, prior to the termination of the then-existing term. Company may, in 
                its sole discretion, terminate this Agreement (i) if as a result of Customer’s breach of this Agreement, or (ii) for any reason upon thirty (30) days prior written notice to the Customer. 3. CHARGES; CHANGES. The initial charges, fees and other amounts for Services rendered and/or equipment furnished by Company and payable by Customer (“Charges”) are set 
                forth on the Service Summary. Company also reserves the right to charge Customer for Additional Services (defined below) provided by Company to Customer, whether or not requested by 
                Customer, including, but not limited to: extra pickup charges, container overages and overflows, container relocation or removal, account reactivation services, equipment repair and maintenance, 
                and/or any other necessary expenses incurred by Company (“Additional Services”), all at such standard prices or rates that Company is charging its customers in the service area at such time. 
                Changes in the frequency of collection, collection schedule, number, capacity and/or type of equipment, and any changes to the Charges payable under this Agreement, may be agreed to orally, 
                in writing or by other actions and practices of the parties, including, without limitation, electronic or online acceptance or payment of the invoice reflecting such changes, and written notice to 
                Customer of any such changes and Customer’s failure to object to such changes, which shall be deemed to be Customer’s affirmative consent to such changes.  Company reserves the right, and 
                Customer  acknowledges  that  it  should  expect  Company  to  increase  or  add  Charges  payable  by  Customer  hereunder  during  the  Term  for:  (i)  any  increase  in  or  other  modification  made  by  
                Company to the Fuel Surcharge, Environmental Charge, and/or any other charges included or referenced in the Service Summary; (iv) any increases in disposal, transportation, and/or disposal 
                costs,  including  fuel  surcharges;  (v)  any  increased  costs  due  to  uncontrollable  circumstances,  including,  without  limitation,  changes  in  applicable  local,  state,  or  federal  laws  or  regulations,  
                including the imposition of or increase in taxes, fees or surcharges, or acts of God such as floods, fires, hurricanes and/or natural disasters; and (vi) for increases in the Consumer Price Index 
                (“CPI”) for Water, Sewer and Trash Collection Services published by U.S. Bureau of Labor Statistics, or with written notice to Customer, any other national, regional or local CPI, with such 
                increases in CPI being measured from the Start Date, or as applicable, Customer’s last CPI based price increase date. 4. INVOICES; PAYMENT TERMS. Company shall send all invoices for Charges and any required notices to Customer under this Agreement to Customer’s billing address and/or 
                email address specified in the Service Summary. Customer shall pay all invoiced Charges within thirty (30) days of the invoice date, by check mailed to Company’s payment address on Customer’s 
                invoice, through Company’s online payment portal, or through Company’s autopay option. Online payments may by subject to applicable convenience fees and other costs charged by Company, 
                from  time to  time. Any Customer invoice  balance not paid within thirty  (30) days of the date of invoice is subject to a late charge, and any Customer check  returned for  insufficient funds is 
                subject to a non-sufficient funds charge, both to the maximum extent allowed by applicable law. Customer acknowledges that a late charge is not to be considered as interest on debt or a finance 
                charge, and is a reasonable charge for the anticipated loss and cost to Company for late payment. If payment is not made when due, Company retains the right to suspend Services until the past 
                due balance is paid in full. In addition to full payment of outstanding balances, Customer shall be required to pay a reactivation charge to resume suspended Services. If Services are suspended 
                for more than fifteen (15) days, Company may, in its sole discretion, immediately terminate this Agreement for default and recover any equipment and all amounts owed hereunder, including 
                liquidated damages under Section 6.   5. EQUIPMENT, ACCESS. All equipment furnished by Company shall at all times remain the property of the Company; however, Customer shall have care, custody and control of
                the  equipment  and  shall  bear  responsibility  and  liability  for  all  loss  or  damage  to  the  equipment  and  for  its  contents  while  at  Customer’s  location.  If  equipment  is  damaged  or  lost  while  in  
                Customer’s care, custody and control, Customer shall pay the new replacement cost of such equipment.  Customer shall not overload, move or alter the equipment or allow a third party to do so, 
                and shall use such equipment only for its intended purpose. During the Term, all equipment shall be maintained in the condition in which it was provided, normal wear and tear excepted. Customer 
                shall provide safe and unobstructed access to the equipment on the scheduled collection day. Customer shall pay, if charged by Company, any additional charges, determined by Company in its 
                sole discretion, for overloading, moving or altering the equipment or allowing a third party to do so, and for any service modifications caused by or resulting from Customer’s failure to provide 
                access.  Company  may  suspend  Services  or  terminate  this  Agreement  in  the  event  Customer  violates  any  of  the  requirements  of  this  provision.    Customer  agrees  that  Company  shall  not  be  
                responsible for any damage to Customer’s pavement, curbing or any other surface resulting from the equipment or Services and further warrants that Customer’s property is sufficient to bear the 
                weight of Company’s equipment and vehicles.   6. LIQUIDATED  DAMAGES.  Customer  acknowledges  that  the  actual  damages  to  Company  in  the  event  of  Customer’s  early  termination  or  breach  of  contract  is  impractical  or
                extremely  difficult  to  determine  or  prove.    As  such,  the  parties agree  that  the  following  liquidated  damages  are  reasonable  and commensurate  with  the  anticipated  loss  to Company  resulting  
                therefrom, and such payment of liquidated damages is not a penalty, but rather an agreed upon charge for Customer’s early termination or breach of contract. In the event Customer terminates 
                this Agreement prior to the expiration of the Initial Term or any Renewal Term, or in the event Company terminates this Agreement for Customer’s default pursuant to Section 4, Customer shall 
                pay, in addition to Company’s attorneys’ fees if any, the following liquidated damages,: (a) if the remaining Term (including any applicable Renewal Term) under this Agreement is six (6) or 
                more months, Customer shall pay the average of its six (6) monthly Charges immediately prior to default or termination (or, if the Start Date is within six (6) months of Company’s last invoice 
                date, the average of all monthly Charges) multiplied by six (6); or (b) if the remaining Term is less than six (6) months, Customer shall pay the average of its six (6) most recent monthly Charges 
                multiplied by the number of months remaining in the Term.  In addition to and not in any way limiting the foregoing, Company shall be entitled to recover all losses, damages and costs, including 
                attorneys’ fees and costs, resulting from Customer’s breach of any other provision of this Agreement in addition to all other remedies available at law or in equity. 7. INDEMNITY. CUSTOMER AGREES TO INDEMNIFY, DEFEND AND HOLD HARMLESS COMPANY AND ITS AFFILIATES FROM AND AGAINST ANY AND ALL
                CLAIMS, ACTIONS, DEMANDS, LIABILITY AND EXPENSE OF EVERY KIND (INCLUDING, BUT NOT LIMITED TO, COURT COSTS, ATTORNEYS’ AND OTHER 
                PROFESSIONAL  FEES)  IN  ANY  WAY  ARISING  FROM  OR  RELATED  TO  (i)  ANY  BODILY  INJURY  (INCLUDING  DEATH),  PROPERTY  DAMAGE  OR  VIOLATION  OF  LAW  
                CAUSED BY CUSTOMER’S BREACH OF THIS AGREEMENT OR BY ANY NEGLIGENT ACT OR OMISSION OR WILLFUL MISCONDUCT OF CUSTOMER OR ITS EMPLOYEES, 
                AGENTS  OR  CONTRACTORS;  OR  (ii)  CUSTOMER’S  USE,  OPERATION  OR  POSSESSION  OF  ANY  EQUIPMENT  FURNISHED  BY  COMPANY.    COMPANY  SHALL  NOT  BE 
                LIABLE TO CUSTOMER FOR CONSEQUENTIAL, INCIDENTAL OR PUNITIVE DAMAGES ARISING OUT OF THE PERFORMANCE OR BREACH OF THIS AGREEMENT.   8. RIGHT OF FIRST REFUSAL. Customer grants to Company a right of first refusal to match any offer relating to services similar to those provided hereunder which Customer
                receives,  or  intends  to  make,  upon  termination  of  this  Agreement  for  any  reason  and  Customer  shall  give  Company  prompt  written  notice  of  any  such  offer  and  a  reasonable  opportunity  to  
                respond to it. 9. FORCE MAJEURE.  Except for the obligation to make payments hereunder, neither party shall be in default or have any liability for its failure to perform or delay in performance
                caused by events beyond its reasonable control, including, but not limited to, strikes, riots, imposition of laws or governmental orders, fires, acts of God, pandemics, natural disasters, and the 
                inability to obtain equipment, and the affected party shall be excused from performance during the occurrence of such events. 10. MISCELLANEOUS. This Agreement shall be binding on and shall inure to the benefit of the parties hereto and their respective successors and assigns. In addition to, and not in 
                limitation of, the foregoing, the terms and provisions of this Agreement may be amended and modified as agreed to by the parties as provided in Section 3. Subject to the foregoing, this Agreement 
                represents  the  entire  agreement  between  the  parties  and  supersedes  any  and  all  other  agreements  for  the  same  Services  at  the  same  Customer  Service  Locations  covered  by  this  Agreement,  
                whether written or oral, that may exist between the parties. If Customer moves its place of business to another location within Company’s collection areas, Company has the right and may elect 
                to continue to provide services at the new location in accordance with this Agreement. The Customer hereby expressly consents to the assignment of this Agreement by Company to any Affiliate, 
                subsidiary, successor, assign, or purchaser of any part of its business and expressly consents to be bound by all the terms herein to any such successors, assigns or purchasers.  This Agreement 
                shall be construed in accordance with the law of the state in which the Services are provided. Should any litigation be commenced between the parties or their respective successors, affiliates, 
                agents or assigns, relating to or concerning the Services or this Agreement, or the rights and obligations of the parties hereunder, such litigation shall be commenced, and each party submits to 
                the jurisdiction of the state and federal courts of the state where the Services are performed. In the event Company successfully enforces its rights against Customer hereunder, Customer shall be 
                required to pay Company’s attorneys’ fees and court costs. BOTH CUSTOMER AND COMPANY HEREBY IRREVOCABLY WAIVE ANY AND ALL RIGHT TO TRIAL BY JURY IN 
                ANY  LEGAL  PROCEEDING  ARISING  OUT  OF  OR  RELATED  TO  THE  SERVICES  OR  THIS  AGREEMENT.  All  written  notification  to  Company  required  by  this  Agreement  shall  be  
                effective upon receipt and delivered by Certified Mail, Return Receipt Requested, courier or by hand to Company’s address on the first page of the Service Summary, provided that Company 
                may provide written notice to Customer of a different address for written notice to Company. If any provision of this Agreement is held to be illegal, invalid or unenforceable under present or 
                future  laws,  such  provision  shall  be  fully  severable;  the  Agreement  shall  be  construed  and  enforced  as  if  such  illegal,  invalid  or  unenforceable  provision  had  never  comprised  a  part  of  the  
                Agreement; and the remaining provisions of the Agreement shall remain in full force and effect and shall not be affected by the illegal, invalid or unenforceable provision or by its severance.  
                Furthermore, in lieu of such illegal, invalid or unenforceable provisions, there shall be added automatically as a part of the Agreement, a provision as similar in terms to such illegal, invalid or 
                unenforceable provision as may be possible and be legal, valid or enforceable. Notwithstanding the termination of this Agreement, Sections 5, 6, 7, 10 and Customer’s obligation to make payments 
                for  all  Charges  and  other  amounts  due  or payable  hereunder  through  the  termination  date  shall  survive  the  termination  of  this Agreement.  The  term  “Affiliate(s)”  shall mean  an  entity  that  is  
                controlling, controlled by, or is under control with Company, where control may be either management authority, contract, or equity interest.  
                
                
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
    
