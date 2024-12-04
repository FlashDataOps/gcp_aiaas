import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

st.title("Contract Field Extractor with ChatGroq")
st.write("Upload a PDF contract to extract predefined fields using ChatGroq.")

# Función para extraer texto usando pdfplumber
def extract_text_with_pdfplumber(file):
    """
    Extrae texto de un archivo PDF utilizando pdfplumber.
    """
    extracted_text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() + "\n"  # Concatenar texto de cada página
    except Exception as e:
        return f"Error extracting text: {e}"
    return extracted_text.strip()

# Subir el archivo PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.write("Processing the uploaded PDF...")

    # Extraer texto usando pdfplumber
    pdf_text = extract_text_with_pdfplumber(uploaded_file)

    # Mostrar texto extraído
    if pdf_text:

        # Crear el prompt con los campos específicos que quieres extraer
        prompt = ChatPromptTemplate.from_messages([
            ('system', """
                You are an advanced AI trained to process and extract structured data from legal contracts. 
                Your task is to extract the specified fields with high accuracy and consistency. 
                Always return the output in CSV format, with fields separated by semicolons (;). 
                Do not include any additional text or commentary outside the CSV.

                The fields to extract are:
                - contract_number: The unique identifier for the contract.
                - start_date: The date the contract begins.
                - estimated_completion_date: The projected end date for the contract.
                - client_name: The name of the client or contracting entity.
                - client_tax_id: The official tax identification number of the client.
                - contractor_name: The name of the contractor or company executing the project.
                - contractor_tax_id: The official tax identification number of the contractor.
                - type_of_construction: The type or category of the construction project (e.g., Roadway, Building, etc.).
                - location: The geographic location of the project (e.g., city, region, country).
                - approved_budget: The total approved budget for the project, including currency.
                - total_duration_days: The total number of days allocated for the project.
                - project_status: The current status of the project (e.g., Planned, In Progress, Completed).
                - payment_terms: A description of how payments are structured (e.g., advance payments, installments, final settlement).
                - delay_penalties: The penalties outlined for project delays, including the amount and frequency (if applicable).
                - milestones: A list of project milestones, each formatted as:
                  milestone_1_start_date; milestone_1_estimated_completion_date; milestone_1_description;
                  milestone_2_start_date; milestone_2_estimated_completion_date; milestone_2_description;

                Formatting Guidelines:
                - Use semicolons (;) to separate fields.
                - Ensure all field names are consistent and in snake_case (e.g., "contract_number").
                - Use ISO 8601 format for all dates (e.g., "YYYY-MM-DD").
                - If a field is missing in the contract, leave the field empty.
                - Maintain the order of the fields exactly as listed above.

                Example output:
                contract_number;start_date;estimated_completion_date;client_name;client_tax_id;contractor_name;...
                12345;2024-01-01;2024-12-31;Client ABC;ABC123;Contractor XYZ;...
            """),
            ('user', f"Here is the content of the PDF you need to process: {pdf_text}")
        ])

        # Cargar el modelo LLM de ChatGroq
        llm = ChatGroq(model="llama-3.1-70b-versatile")
        chain = (prompt | llm | StrOutputParser())

        # Invocar el modelo con el texto del PDF
        try:
            with st.spinner("Extracting fields with ChatGroq..."):
                csv_content = chain.invoke({"pdf_text": pdf_text})

            st.subheader("Extracted Fields in CSV Format")
            st.text_area("CSV Output", csv_content, height=300)

            # Botón para descargar el CSV
            st.download_button(
                label="Download as CSV",
                data=csv_content.encode('utf-8'),
                file_name=f"{uploaded_file.name.split('.')[0]}_fields.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error extracting fields with ChatGroq: {e}")
    else:
        st.error("No text was extracted from the PDF. Please check the file.")
