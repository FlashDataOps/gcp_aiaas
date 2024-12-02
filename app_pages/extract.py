from os import sep
from sqlalchemy import table
import streamlit as st
import time
from stqdm import stqdm
import aux_functions as af
import pandas as pd
import numpy as np
import langchain_utils as lu
import base64
import traceback
# Título de la aplicación
st.title("Extracción de campos")
# Texto descriptivo
st.write("Puedes subir una ficha de admisión de la UFV y obtener toda la información del docuemnto.")

# Área de arrastrar y soltar para subir el archivo
uploaded_file = st.file_uploader("Arrastra y suelta tu archivo aquí o selecciona un archivo", type=["pdf"])

if uploaded_file:
    if uploaded_file.type == "image/png":
        # Leer la imagen como binario
        image = uploaded_file.read()
        # Convertir la imagen a Base64
        encoded_image = base64.b64encode(image).decode('utf-8')
        # Usar HTML para centrar la imagen
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded_image}" alt="Imagen Subida" width="220">
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

campos = ""

# Botón para subir el archivo
if st.button("📤 Extraer campos", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner('Extrayendo campos...'):
            try:
                name_file = uploaded_file.name.split(".")[0]
                path_ficha_gcp = f"ufv-demo/ficha-admision/{name_file}"
                print("Buscando si existe fichero en blob")
                lista_blobs = af.list_blobs(folder_name=path_ficha_gcp)
                if not fr"{path_ficha_gcp}/{uploaded_file.name}" in lista_blobs:
                    print("No existe, se procede a subir")
                    af.upload_blob(file=uploaded_file, folder_name=path_ficha_gcp)
                    time.sleep(5)
                else:
                    print("Existe")
                
                doc_pdf = f"gs://single-cirrus-435319-f1-bucket/{path_ficha_gcp}/{uploaded_file.name}"
                print("Extrayendo Imagenes...")
                image_data = af.extract_areas_from_pdf_base64(pdf_path=uploaded_file, page_number=2)
                print("Imagenes extraidas ->", {len(image_data)})
                print(doc_pdf, type(doc_pdf))
                print(image_data, type(image_data[0]))
                campos = lu.invoke_extraer_campos_ficha(
                    doc_pdf=doc_pdf,
                    image_data=image_data
                )
                #if "correcta" in campos.lower() or "correcto" in campos.lower():
                #    st.success(f"Foto de perfil correcta")
                #else:
                #    st.error(f"Foto de perfil incorrecta")
            except Exception as e:
                traceback.print_exc()
                st.error(fr"Error -> {e}")
        
        # Aquí puedes añadir lógica para interactuar con la base de datos usando el archivo subido
    else:
        st.error("No se ha subido ningún archivo. Por favor, selecciona un archivo para continuar.")

if len(campos) >0:    
    st.markdown(campos, unsafe_allow_html=True)