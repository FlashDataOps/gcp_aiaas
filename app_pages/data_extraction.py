import os
import pandas as pd
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pdfplumber
import json

load_dotenv()

def render_or_update_model_info():
    """
    Renders or updates the model information on the webpage.

    Args:
        model_name (str): The name of the model.

    Returns:
        None
    """
    with open("./design/styles2.css") as f:
        css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/content2.html") as f:
        html = f.read()
    st.markdown(html, unsafe_allow_html=True)


render_or_update_model_info()

# Initialize session for folder clearing
if "folder_cleared" not in st.session_state:
    st.session_state["folder_cleared"] = False

PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Clear PDF folder on start if not done yet
if not st.session_state["folder_cleared"]:
    for file in os.listdir(PDF_FOLDER):
        os.remove(os.path.join(PDF_FOLDER, file))
    st.session_state["folder_cleared"] = True

#st.title("Contract Field Extractor")
#st.write("Upload a PDF contract to extract predefined fields.")

# Sidebar PDF list
st.sidebar.image("Logo-pwc.png", width=60)
st.sidebar.header("Uploaded PDFs")
uploaded_pdfs = [file for file in os.listdir(PDF_FOLDER) if file.endswith(".pdf")] #+ #! TODO METER LA LISTA DE LOS CONTRATOS DEL CSV[]
if uploaded_pdfs:
    for pdf_file in uploaded_pdfs:
        pdf_name = pdf_file[:-4]
        st.sidebar.markdown(f"📄 **{pdf_name}**")
else:
    st.sidebar.write("No PDFs uploaded yet.")

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

        llm = ChatGroq(model="llama-3.1-70b-versatile")
        chain = (prompt | llm | StrOutputParser())

        try:
            with st.spinner("Extracting fields from contract..."):
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
        except Exception as e:
            st.error(f"Error extracting fields with ChatGroq: {e}")
    else:
        st.error("No text was extracted from the PDF. Please check the file.")
