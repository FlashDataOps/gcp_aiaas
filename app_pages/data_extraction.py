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
uploaded_pdfs = [file for file in os.listdir(PDF_FOLDER) if file.endswith(".pdf")] 
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
                    You are an advanced AI model specialized in processing and extracting structured data from contracts.
                    Your task is to extract the following fields from the contract and return the result as a JSON object.
                    Do not include any additional text outside the JSON. If a field is not found in the text, return "NaN".

                    The JSON structure should be as follows:

                    {{
                        "Account Information": {{
                            "Account Number": "...",
                            "Delivery Date": "...",
                            "Permanent/Temporary": "..."
                        }},
                        "Billing Information": {{
                            "Business Name": "...",
                            "Address": "...",
                            "City": "...",
                            "State": "...",
                            "Zip": "...",
                            "Contact": "...",
                            "Phone": "...",
                            "Email": "...",
                            "Invoice Delivery Preference": "..."
                        }},
                        "Service Information": {{
                            "Site Name": "...",
                            "Address": "...",
                            "City": "...",
                            "State": "...",
                            "Zip": "...",
                            "Contact": "...",
                            "Phone": "...",
                            "Email": "...",
                            "PO Number": "..."
                        }},
                        "Front End Services": {{
                            "Qty_container_size_item1": "...",
                            "Frequency_item1": "...",
                            "Locks_casters_item1": "...",
                            "Delivery_item1": "...",
                            "ExtraPickup_item1": "...",
                            "MonthlyRate_item1": "...",
                            "Qty_container_size_item2": "...",
                            "Frequency_item2": "...",
                            "Locks_casters_item2": "...",
                            "Delivery_item2": "...",
                            "ExtraPickup_item2": "...",
                            "MonthlyRate_item2": "...",
                            "Qty_container_size_item3": "...",
                            "Frequency_item3": "...",
                            "Locks_casters_item3": "...",
                            "Delivery_item3": "...",
                            "ExtraPickup_item3": "...",
                            "MonthlyRate_item3": "...",
                        }},
                        "Open Top/Compactor Services": {{
                            "Quantity_container_size_item1": "...",
                            "Compactor_item1": "...",
                            "MonthlyRental_item1": "...",
                            "TripRelocationCharge_item1": "...",
                            "Frequency_item1": "...",
                            "Delivery_item1": "...",
                            "HaulRate_item1": "...",
                            "Disposal_item1": "...",
                            "Quantity_container_size_item2": "...",
                            "Compactor_item2": "...",
                            "MonthlyRental_item2": "...",
                            "TripRelocationCharge_item2": "...",
                            "Frequency_item2": "...",
                            "Delivery_item2": "...",
                            "HaulRate_item2": "...",
                            "Disposal_item2": "...",
                            "Quantity_container_size_item3": "...",
                            "Compactor_item3": "...",
                            "MonthlyRental_item3": "...",
                            "TripRelocationCharge_item3": "...",
                            "Frequency_item3": "...",
                            "Delivery_item3": "...",
                            "HaulRate_item3": "...",
                            "Disposal_item3": "..."
                        }},
                        "Additional Information": {{
                            "Tax Exempt": "...",
                            "Wait Time Rate": "...",
                            "Inactivity Fee": "...",
                            "Special Instructions": "..."
                        }},
                        "Signatures": {{
                            "Contractor Name": "...",
                            "Representative Signature": "...",
                            "Representative Name": "...",
                            "Representative Date": "...",
                            "Customer Name": "...",
                            "Customer Signature": "...",
                            "Customer Date": "..."
                        }}
                    }}

                    Important extraction guidelines:
                    1. Extract all table entries for Front End Services and Open Top/Compactor Services as individual fields
                    2. Use _itemX naming convention for each row's fields
                    3. Extract all available information precisely as it appears in the document
                    4. Ensure each field is extracted as a separate, distinct value
                    5. If a field is not found, use "NaN"
                    6. Preserve exact formatting of numbers, dates, and text
                    7. Always extract all columns mentioned above, and if the field is not found set it to NaN
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

                st.dataframe(df, width=800)
                
        except Exception as e:
            st.error(f"Error extracting fields with ChatGroq: {e}")
    else:
        st.error("No text was extracted from the PDF. Please check the file.")
