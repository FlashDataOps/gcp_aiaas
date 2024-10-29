import streamlit as st

# Ruta del archivo PDF
pdf_path = './documentation/Documentación MontyBot.pdf'

# Introducción con secciones disponibles en el documento
st.write("## Documentación de MontyBot")
st.write("""
Esta documentación incluye las siguientes secciones:
- **Introducción**: Breve explicación sobre MontyBot y su propósito.
  - Instrucciones para ejecutar MontyBot
  - Futuras actualizaciones
  - Configuraciones extra
- **Análisis del Código**: Explicación de los principales scripts de MontyBot.
  - Descripción del Script `create_index.py`
  - Descripción del Script `langchain_utils.py`
  - Descripción del Script `app.py`
- **Resumen y Conclusión**: Conclusiones y futuro desarrollo de MontyBot.
""")

# Función para mostrar el botón de descarga del PDF
def mostrar_pdf(pdf_path):
    # Lee el archivo PDF
    with open(pdf_path, "rb") as file:
        pdf_data = file.read()
        # Botón para descargar el archivo
        st.download_button(label="Descargar PDF", data=pdf_data, file_name="Documentación MontyBot.pdf")

mostrar_pdf(pdf_path)
