import streamlit as st


nlp2sql = st.Page("./app_pages/nlp2sql.py", title="NLP2SQL", icon="🤖")
rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")

bd = st.Page("./app_pages/bd.py", title="Añadir base de datos", icon="🔧")
index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")

demo_agent = st.Page("./app_pages/demo_agent.py", title="Prompting SQL", icon="🔍")

demo_prompt_rol = st.Page("./app_pages/demo_prompt_rol.py", title="Prompting roles", icon="🤖")

pg = st.navigation(
    {
   #    "Información": [welcome, doc],
       "Bots": [demo_prompt_rol,demo_agent],
    #    "Ajustes": [bd] 
    }
    )

st.set_page_config(
    page_title="Prompt Workshop ATC",
    page_icon="🤖",
)

pg.run()