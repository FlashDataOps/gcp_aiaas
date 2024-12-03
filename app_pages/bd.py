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
from datetime import datetime
import os
with open("./design/photo/styles.css", encoding="utf-8") as f:
    css = f.read()
    st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

    with open("./design/photo/content.html", encoding="utf-8") as f:
        html = f.read()
    st.markdown(html, unsafe_allow_html=True)

# Texto descriptivo
st.write("Puedes subir una foto y comprobar si es correcta para un carnet de estudiante.")

# Área de arrastrar y soltar para subir el archivo
uploaded_file = st.file_uploader("Arrastra y suelta tu archivo aquí o selecciona un archivo", type=["png", "jpeg"])

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
    if uploaded_file.type == "image/jpeg":
        # Leer la imagen como binario
        image = uploaded_file.read()
        # Convertir la imagen a Base64
        encoded_image = base64.b64encode(image).decode('utf-8')
        # Usar HTML para centrar la imagen
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{encoded_image}" alt="Imagen Subida" width="220">
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
    
    if uploaded_file.type == "image/jpg":
        # Leer la imagen como binario
        image = uploaded_file.read()
        # Convertir la imagen a Base64
        encoded_image = base64.b64encode(image).decode('utf-8')
        # Usar HTML para centrar la imagen
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpg;base64,{encoded_image}" alt="Imagen Subida" width="220">
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")


    
campos = ""

# Botón para subir el archivo
if st.button("📤 Analizar imagen", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner('Analizando imagen...'):
            try:
                
                # Obtener el tiempo actual
                ahora = datetime.now()

                # Formatear como día_mes_hora_minuto_segundo
                formato_personalizado = ahora.strftime("%d_%m_%H_%M_%S")
                # Separar el nombre del archivo de su extensión
                nombre_base, extension = os.path.splitext(uploaded_file.name)

                # Crear el nuevo nombre con la extensión al final
                nuevo_nombre = f"{nombre_base}_{formato_personalizado}{extension}"

                uploaded_file.name = nuevo_nombre
                path_carnet_gcp = "ufv-demo/foto-carnet"
                print("Buscando si existe fichero en blob")
                lista_blobs = af.list_blobs(folder_name=path_carnet_gcp)
                if not fr"{path_carnet_gcp}/{uploaded_file.name}" in lista_blobs:
                    print("No existe, se procede a subir")
                    af.upload_blob(file=uploaded_file, folder_name=path_carnet_gcp)
                    time.sleep(5)
                else:
                    print("Existe")
                
                image_data = f"gs://single-cirrus-435319-f1-bucket/ufv-demo/foto-carnet/{uploaded_file.name}"
                campos = lu.invoke_extraer_campos_foto(
                    image_data=image_data
                )
                #if "correcta" in campos.lower() or "correcto" in campos.lower():
                #    st.success(f"Foto de perfil correcta")
                #else:
                #    st.error(f"Foto de perfil incorrecta")
            except Exception as e:
                st.error(fr"Error -> {e}")
        
        # Aquí puedes añadir lógica para interactuar con la base de datos usando el archivo subido
    else:
        st.error("No se ha subido ningún archivo. Por favor, selecciona un archivo para continuar.")

if len(campos) >0:    
    st.markdown(campos, unsafe_allow_html=True)