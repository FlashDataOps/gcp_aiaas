import streamlit as st

#welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")
#doc = st.Page("./app_pages/doc.py", title="Documentación", icon="📚")

#nlp2sql = st.Page("./app_pages/nlp2sql.py", title="Chat with data", icon="🤖")
#data_extraction = st.Page("./app_pages/data_extraction.py", title="Data extraction", icon="🔍")

#rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")
#ppt = st.Page("./app_pages/ppt_gen.py", title="PPT Generator", icon="📊")

# bd = st.Page("./app_pages/bd.py", title="Add database", icon="🔧")


st.set_page_config(
    page_title="FCC",
    page_icon="🤖",
)

with st.sidebar:
    
    pg = st.navigation(
        {
            "Data extraction": [st.Page("./app_pages/data_extraction.py", title="Data extraction", icon="🔍")],
            "Bots": [st.Page("./app_pages/nlp2sql.py", title="FCC Chat with data", icon="🤖")],
            # "bd": [st.Page("./app_pages/bd.py", title="Add database", icon="🔧")],
        }
    )

pg.run()