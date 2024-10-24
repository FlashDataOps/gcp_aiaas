import streamlit as st
st.write("# Welcome to MontyBot! 👋")

# Crear opciones en la barra lateral
st.sidebar.title("Opciones de la barra lateral")

# 1. Selectbox - Menú desplegable
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ("Opción 1", "Opción 2", "Opción 3")
)

# 2. Slider - Deslizador para seleccionar un número
valor_slider = st.sidebar.slider("Selecciona un valor", 0, 100, 50)

# 3. Text Input - Caja de texto
texto = st.sidebar.text_input("Introduce tu nombre", "Nombre")

# 4. Checkbox - Casilla de verificación
checkbox = st.sidebar.checkbox("Marcar esta casilla")

# 5. Radio Buttons - Botones de radio
radio = st.sidebar.radio(
    "Selecciona una opción de radio:",
    ("Primera opción", "Segunda opción", "Tercera opción")
)

# 6. SelectSlider - Deslizador con valores personalizados
valor_select_slider = st.sidebar.select_slider(
    "Selecciona un nivel:",
    options=["Bajo", "Medio", "Alto"]
)

# 7. Date Input - Seleccionar una fecha
fecha = st.sidebar.date_input("Selecciona una fecha")

# 8. Time Input - Seleccionar una hora
hora = st.sidebar.time_input("Selecciona una hora")

# 9. File Uploader - Subir un archivo
archivo = st.sidebar.file_uploader("Sube un archivo", type=["csv", "txt"])

# 10. Color Picker - Selector de color
color = st.sidebar.color_picker("Elige un color")

st.markdown(
    """
Monty Bot es un bot personalizado de IA Generativa diseñado para crear DEMOS. Es capaz de conectarse a cualquier Modelo de Lenguaje de Pago (coming soon) o de código abierto.

## Instrucciones para Ejecutar el Bot

1. **Añadir Documentación**
   - Agrega la documentación necesaria en la carpeta `data`.

2. **Indexar Documentos**
   - Indexa los documentos utilizando el archivo `indexar.ipynb`.

3. **Ejecutar el Bot**
   - Ejecuta el bot con el comando:
     ```sh
     streamlit run app.py
     ```

**IMPORTANTE:** Desactivar VPN antes de ejecutar el bot.

## Futuras Actualizaciones

- Conexión con Gemini y OpenAI
- Sistema de feedback de cada mensaje del bot
- Mejoras de rendimiento

## Edición del Flujo de Funcionamiento del Bot

Para editar el flujo de funcionamiento del bot, es necesario modificar el método `invoke_chain` en `langchain_utils.py`.

## Configuración del Entorno

En el archivo `.env` es necesario añadir la API KEY de GROQ y, de forma opcional, se puede integrar Langsmith.

```plaintext
# Ejemplo de .env
GROQ_API_KEY=tu_api_key_aqui
LANGSMITH_API_KEY=tu_api_key_aqui (opcional)
"""
)