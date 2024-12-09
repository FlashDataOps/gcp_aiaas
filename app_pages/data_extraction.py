import os
import pandas as pd
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
import pdfplumber
import json

load_dotenv()

<<<<<<< HEAD
def render_or_update_model_info():
    """
    Renders or updates the model information on the webpage.
=======
@st.cache_data
def get_model(model="gemini-1.5-flash-002"):
    return ChatVertexAI(model_name="gemini-1.5-flash-002",project="single-cirrus-435319-f1")
# Initialize session for folder clearing
if "folder_cleared" not in st.session_state:
    st.session_state["folder_cleared"] = False

PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)
>>>>>>> a890407e6ef27767c5871be21abd7b1c96318d9a

def render_or_update_model_info(model_name):
    with open("./design/styles2.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content2.html") as f:
        html = f.read().format(model_name)
    st.markdown(html, unsafe_allow_html=True)
<<<<<<< HEAD


render_or_update_model_info()

# Initialize session for folder clearing
if "folder_cleared" not in st.session_state:
    st.session_state["folder_cleared"] = False

PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

=======
    
>>>>>>> a890407e6ef27767c5871be21abd7b1c96318d9a
# Clear PDF folder on start if not done yet
if not st.session_state["folder_cleared"]:
    for file in os.listdir(PDF_FOLDER):
        os.remove(os.path.join(PDF_FOLDER, file))
    st.session_state["folder_cleared"] = True

<<<<<<< HEAD
# Sidebar PDF list
st.sidebar.image("Logo-pwc.png", width=60)
=======
# Render or update model information
render_or_update_model_info("gemini-1.5-flash-002")

# Sidebar PDF list
>>>>>>> a890407e6ef27767c5871be21abd7b1c96318d9a
st.sidebar.header("Uploaded PDFs")
uploaded_pdfs = [file for file in os.listdir(PDF_FOLDER) if file.endswith(".pdf")]
if uploaded_pdfs:
    for pdf_file in uploaded_pdfs:
        pdf_name = pdf_file[:-4]
        st.sidebar.markdown(f"📄 **{pdf_name}**")
else:
    st.sidebar.write("No PDFs uploaded yet.")
    
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.image("Logo-pwc.png", width=60)

def extract_text_with_pdfplumber(file):
    """Extract text from PDF using pdfplumber."""
    extracted_text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error extracting text: {e}"
    return extracted_text.strip()

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved: {uploaded_file.name}")

    pdf_text = extract_text_with_pdfplumber(file_path)

    if pdf_text:
        # English prompt
        prompt = ChatPromptTemplate.from_messages([
            ('system', """
            You are an advanced AI model specialized in processing and extracting structured data from lease contracts.
            Your task is to extract the following fields from the contract and return the result as a JSON object.
            Do not include any additional text outside the JSON. If a field is not found in the text, return "NaN".

            The JSON structure should be as follows:

            \{{
            "General Information": \{{
                "Introduction": "...",
                "Contract Number": "...",
                "Date": "...",
                "Place": "..."
            \}},
            "Involved Parties": \{{
                "Landlord": \{{
                "Name": "...",
                "Address": "...",
                "Representative": "...",
                "Contact": "..."
                \}},
                "Tenant": \{{
                "Name": "...",
                "Address": "...",
                "Representative": "...",
                "Contact": "..."
                \}}
            \}},
            "Property Details": \{{
                "Local 1 (surface area)": "...",
                "Local 2 (surface area)": "...",
                "Total Surface Area": "..."
            \}},
            "Rents": \{{
                "Fixed Monthly Rent": "...",
                "Variable Rent": "..."
            \}},
            "Common Expenses": \{{
                "Maintenance of Common Areas": "...",
                "Utilities (water, electricity)": "...",
                "Other Services": "..."
            \}},
            "Termination Options": \{{
                "First Option (conditions)": "...",
                "Second Option (conditions)": "..."
            \}},
            "Renewals": \{{
                "Initial Duration": "...",
                "Automatic Renewals": "..."
            \}},
            "Other Conditions": \{{
                "Tenant Obligations": "...",
                "Guarantee": "..."
            \}},
            "Signatures": \{{
                "Landlord (name and title)": "...",
                "Tenant (name and title)": "...",
                "Signature Date": "..."
            \}}
            \}}

            If a field cannot be found, return "NaN".
            It is very important that it comes out in json format
            """),
            ('user', "Here is the PDF content to process:\n\n{pdf_text}")
        ])

<<<<<<< HEAD
        llm = ChatGroq(model="llama-3.1-70b-versatile")
        chain = (prompt | llm | StrOutputParser())

        try:
            with st.spinner("Extracting fields with ChatGroq..."):
                json_content = chain.invoke({"pdf_text": pdf_text})
                
                # Parse the JSON string returned by the model
                try:
                    data = json.loads(json_content)
                except json.JSONDecodeError:
                    st.error("The model did not return valid JSON.")
                    st.write(json_content)
                    st.stop()

                # Function to flatten JSON
                def flatten_json(y):
                    out = {}
                    def flatten(x, name=''):
                        if isinstance(x, dict):
                            for a in x:
                                flatten(x[a], name + a + ' - ')
                        else:
                            out[name[:-3]] = x
                    flatten(y)
                    return out

                flat_data = flatten_json(data)

                # Convert dict to DataFrame
                df = pd.DataFrame.from_dict(flat_data, orient='index', columns=['Value'])
                df.index.name = 'Field'
                st.dataframe(df)
=======
        llm = get_model(model="gemini-1.5-flash-002")
        chain = (prompt | llm | StrOutputParser())

        try:
            with st.spinner("Extracting fields with Gemini..."):
                json_content = chain.invoke({"pdf_text": pdf_text})

                # Parse the JSON string returned by the model
                try:
                    json_content = json_content.replace("```","").replace("json", "")
                    data = json.loads(json_content)
                    #print(json_content)
                except json.JSONDecodeError:
                    st.error("The model did not return valid JSON.")
                    st.write(json_content)
                    st.stop()

                # Function to flatten JSON
                def flatten_json(y):
                    out = {}
                    def flatten(x, name=''):
                        if isinstance(x, dict):
                            for a in x:
                                flatten(x[a], name + a + ' - ')
                        else:
                            out[name[:-3]] = x
                    flatten(y)
                    return out

                flat_data = flatten_json(data)

            # Convert dict to DataFrame
            df = pd.DataFrame.from_dict(flat_data, orient='index', columns=['Value'])
            df.index.name = 'Field'
            st.dataframe(df, use_container_width=True)
>>>>>>> a890407e6ef27767c5871be21abd7b1c96318d9a
        except Exception as e:
            st.error(f"Error extracting fields with Gemini: {e}")
    else:
        st.error("No text was extracted from the PDF. Please check the file.")
