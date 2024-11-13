import streamlit as st

welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")
doc = st.Page("./app_pages/doc.py", title="Documentación", icon="📚")

nlp2sql = st.Page("./app_pages/nlp2sql.py", title="NLP2SQL", icon="🤖")
rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")
ppt = st.Page("./app_pages/ppt_gen.py", title="PPT Generator", icon="📊")

bd = st.Page("./app_pages/bd.py", title="Añadir base de datos", icon="🔧")
index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")

foundations = st.Page("./app_pages/foundations.py", title="Sr. Padre", icon="🙏")
demo_agent = st.Page("./app_pages/demo_agent.py", title="Agente 007", icon="👮")


pg = st.navigation(
    {
       "Información": [welcome, doc],
       "Bots": [nlp2sql, rag, ppt, demo_agent],
       "Ajustes": [bd, index] 
    }
    )

st.set_page_config(
    page_title="MontyBot",
    page_icon="🤖",
)

pg.run()