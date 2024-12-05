import streamlit as st

#welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")
#doc = st.Page("./app_pages/doc.py", title="Documentación", icon="📚")

nlp2sql = st.Page("./app_pages/nlp2sql.py", title="Chat with data", icon="🤖")
data_ext = st.Page("./app_pages/data_extraction.py", title="Data Extraction")

#rag = st.Page("./app_pages/rag.py", title="RAG", icon="📄")
#ppt = st.Page("./app_pages/ppt_gen.py", title="PPT Generator", icon="📊")

bd = st.Page("./app_pages/bd.py", title="Add database", icon="🔧")
#index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")
# pg = st.navigation(
#     {
#        #"Información": [welcome, doc],
#        "Bots": [nlp2sql],
#        "Settings": [bd, data_ext] 
#     }
#     )

st.set_page_config(
    page_title="FCC",
    page_icon="🤖",
)

with st.sidebar:
    empty_col1, logo_col1, logo_col2, empty_col2 = st.columns([1, 4, 2, 1])  # Add empty columns for centering
    
    with logo_col1:
        # st.image("Logo-pwc.png", width=80)  # Adjust width for proper alignment
        st.write("")
        st.image("image.png", width=150)  # Adjust width for proper alignment

    with logo_col2:
        st.write("")  # Add empty space above the FCC logo
        st.image("Logotipo_de_FCC.png", width=80)  # Adjust width for proper alignment

    pg = st.navigation(
        {
            "Bots": [st.Page("./app_pages/nlp2sql.py", title="FCC Chat with data", icon="🤖")],
            "Settings": [
                st.Page("./app_pages/bd.py", title="Add database", icon="🔧"),
                st.Page("./app_pages/data_extraction.py", title="Data Extraction"),
            ],
        }
    )


pg.run()