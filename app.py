import streamlit as st

#welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")
#doc = st.Page("./app_pages/doc.py", title="Documentación", icon="📚")

nlp2sql = st.Page("./app_pages/nlp2sql.py", title="FCC Chatbot", icon="🤖")
data_ext = st.Page("./app_pages/data_extraction.py", title="Data Extraction")

#rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")
#ppt = st.Page("./app_pages/ppt_gen.py", title="PPT Generator", icon="📊")

bd = st.Page("./app_pages/bd.py", title="Add database", icon="🔧")
#index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")
pg = st.navigation(
    {
       #"Información": [welcome, doc],
       "Bots": [nlp2sql],
       "Settings": [bd, data_ext] 
    }
    )

st.set_page_config(
    page_title="FCC",
    page_icon="🤖",
)

pg.run()