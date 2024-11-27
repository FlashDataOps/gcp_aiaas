import streamlit as st

welcome = st.Page("./app_pages/portada.py", title="Portada", icon="👓")

extraccion = st.Page("./app_pages/extraccion.py", title="Extraccion", icon="👓")

chatbot = st.Page("./app_pages/Chatbot.py", title="ChatBot", icon="🧑‍💼")

informes = st.Page("./app_pages/Generacion_Informe.py", title="Informes", icon="🗒️")

# bd = st.Page("./app_pages/bd.py", title="Añadir base de datos", icon="🔧")

pg = st.navigation(
    {
        "Bienvenida":[welcome],
       "Funcionalidades": [chatbot, informes, extraccion],
    }
    )

st.set_page_config(
    page_title="HowdenBot",
    page_icon="🤖",
)

pg.run()