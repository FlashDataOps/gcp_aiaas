import os
import pandas as pd
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# Inicializar la sesión para el control de borrado
if "folder_cleared" not in st.session_state:
    st.session_state["folder_cleared"] = False

# Ruta de la carpeta PDF
PDF_FOLDER = "pdfs"
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
# Crear la carpeta si no existe
os.makedirs(PDF_FOLDER, exist_ok=True)

# Limpiar la carpeta al iniciar si aún no se ha hecho
if not st.session_state["folder_cleared"]:
    for file in os.listdir(PDF_FOLDER):
        os.remove(os.path.join(PDF_FOLDER, file))
    st.session_state["folder_cleared"] = True

# st.title("Contract Field Extractor")
st.write("Upload a PDF contract to extract predefined fields.")

# Mostrar los archivos PDF disponibles en la barra lateral
st.sidebar.header("Uploaded PDFs")
uploaded_pdfs = [file for file in os.listdir(PDF_FOLDER) if file.endswith(".pdf")]

if uploaded_pdfs:
    for pdf_file in uploaded_pdfs:
        pdf_name = pdf_file[:-4]  # Eliminar extensión .pdf
        st.sidebar.markdown(
            f"📄 **{pdf_name}**"
        )
else:
    st.sidebar.write("No PDFs uploaded yet.")


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


render_or_update_model_info()

if uploaded_file:
    # Guardar el archivo en la carpeta 'pdfs' (sobrescribir si ya existe)
    file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved: {uploaded_file.name}")

    # Extraer texto usando pdfplumber
    pdf_text = extract_text_with_pdfplumber(file_path)

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
                - If a field is missing in the contract, use "NaN".
                - Always include the column names in the output.
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

            # Convertir el CSV a DataFrame y transponerlo
            csv_lines = csv_content.split("\n")

            if len(csv_lines) > 1:
                headers = csv_lines[0].split(";")
                values = csv_lines[1].split(";")
                df = pd.DataFrame([values], columns=headers).transpose()
                df.columns = ["Value"] 
                df.index.name = "Field"
                st.dataframe(df)
            else:
                st.warning("The extracted CSV appears to be empty or malformed.")

        except Exception as e:
            st.error(f"Error extracting fields with ChatGroq: {e}")
    else:
        st.error("No text was extracted from the PDF. Please check the file.")
