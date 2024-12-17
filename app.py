import streamlit as st

nlp2sql = st.Page("./app_pages/nlp2sql.py", title="Nestlé", icon="🤖")
bd = st.Page("./app_pages/bd.py", title="Añadir base de datos", icon="🔧")

pg = st.navigation(
    {
       "Bots": [nlp2sql],
       #"Ajustes": [bd] 
    }
    )

st.set_page_config(
    page_title="Nestlé",
    page_icon="🤖",
)

pg.run()