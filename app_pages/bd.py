from os import sep
from sqlalchemy import table
import streamlit as st
import time
from stqdm import stqdm
import aux_functions as af
import pandas as pd
import numpy as np
import traceback
# Título de la aplicación
st.title("Interacción con la Base de Datos")
# Texto descriptivo
st.write("Puedes subir ficheros para interactuar con la base de datos. Usa el área de abajo para arrastrar y soltar tu archivo.")

# Área de arrastrar y soltar para subir el archivo
uploaded_file = st.file_uploader("Arrastra y suelta tu archivo aquí o selecciona un archivo", type=["csv", "xlsx"])

# Crear dos columnas
col1, col2 = st.columns(2)

# Campo en la primera columna (Dropdown para seleccionar el separador)
with col1:
    if not uploaded_file:
        separador = st.selectbox("Separador", [";", ","])
    elif ".csv" in uploaded_file.name:
        separador = st.selectbox("Separador", [";", ","])
    else:
        sheets = pd.ExcelFile(uploaded_file).sheet_names
        separador = st.selectbox("Sheet", sheets)

# Campo en la segunda columna (Dropdown para seleccionar el encoding)
with col2:
    encoding = st.selectbox("Encoding", ["utf-8", "latin-1"])

# Nombre de la tabla
table_name = st.text_input('Ingresa el nombre de la tabla')

df = pd.DataFrame()

# Botón para subir el archivo
if st.button("Subir Archivo"):
    if uploaded_file is not None:
        with st.spinner('Creando base de datos...'):
            try:
                df = af.db_connection.upload_db_from_settings(file=uploaded_file, table_name=table_name, sep=separador, encoding=encoding)
                st.success(f"Archivo subido: {uploaded_file.name}. Todo fue bien.")
            except Exception as e:
                traceback.print_exc()
                st.error(fr"Error al subir el archivo. Contacta con Monty. -> {e}")
        
        # Aquí puedes añadir lógica para interactuar con la base de datos usando el archivo subido
    else:
        st.error("No se ha subido ningún archivo. Por favor, selecciona un archivo para continuar.")

if len(df) > 0:     
    st.dataframe(df)