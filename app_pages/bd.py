from os import sep
from sqlalchemy import table
import streamlit as st
import time
from stqdm import stqdm
import aux_functions as af
import pandas as pd
import numpy as np
# Título de la aplicación
st.title("Upload a file to create a database")
# Texto descriptivo
st.write("You can upload a file to create a database. The file can be a CSV or an Excel file. You can also select the separator and the encoding of the file.")

# Área de arrastrar y soltar para subir el archivo
uploaded_file = st.file_uploader("Select a file from device or drop it here", type=["csv", "xlsx"])

# Crear dos columnas
col1, col2 = st.columns(2)

# Campo en la primera columna (Dropdown para seleccionar el separador)
with col1:
    if not uploaded_file:
        separador = st.selectbox("Separator", [";", ","])
    elif ".csv" in uploaded_file.name:
        separador = st.selectbox("Separator", [";", ","])
    else:
        sheets = pd.ExcelFile(uploaded_file).sheet_names
        separador = st.selectbox("Sheet", sheets)

# Campo en la segunda columna (Dropdown para seleccionar el encoding)
with col2:
    encoding = st.selectbox("Encoding", ["utf-8", "latin-1"])

# Nombre de la tabla
table_name = st.text_input('Give the table a name')

df = pd.DataFrame()

# Botón para subir el archivo
if st.button("Upload FIle"):
    if uploaded_file is not None:
        with st.spinner('Creating database...'):
            try:
                df = af.db_connection.upload_db_from_settings(file=uploaded_file, table_name=table_name, sep=separador, encoding=encoding)
                st.success(f"File correctly uploaded: {uploaded_file.name}. Todo fue bien.")
            except Exception as e:
                st.error(fr"Error uploading the file. Contact with PwC -> {e}")
        
        # Aquí puedes añadir lógica para interactuar con la base de datos usando el archivo subido
    else:
        st.error("No file selected. Please upload a file to continue.")

if len(df) > 0:     
    st.dataframe(df)