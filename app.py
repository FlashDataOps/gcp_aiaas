import streamlit as st

#welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")
#doc = st.Page("./app_pages/doc.py", title="Documentación", icon="📚")

#nlp2sql = st.Page("./app_pages/nlp2sql.py", title="NH Hoteles", icon="🤖")
extract = st.Page("./app_pages/extract.py", title="Extracción de campos", icon="🤖")
#rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")
#ppt = st.Page("./app_pages/ppt_gen.py", title="PPT Generator", icon="📊")

bd = st.Page("./app_pages/bd.py", title="Check Foto de perfil", icon="🔍")
#index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")
pg = st.navigation(
    {
       #"Información": [welcome, doc],
       "Funcionalidades": [extract, bd]
    }
    )

st.set_page_config(
    page_title="UFV Helper",
    page_icon="🤖",
)

pg.run()