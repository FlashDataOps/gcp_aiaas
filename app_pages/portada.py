import streamlit as st

# Definir las tres columnas, con la imagen centrada en la columna central
left_co, cent_co, last_co = st.columns([1, 5, 1])

with cent_co:
    # Definir la URL de la imagen
    image_url = "images/Howdini.png"
    st.image(image_url, use_column_width=True)

# Título y subtítulo con un estilo mejorado y centrado
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #003366;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .subtitle {
            font-size: 24px;
            font-weight: 300;
            color: #555555;
            text-align: center;
            margin-top: -10px;
        }
        .container {
            margin-top: 20px;
            background-color: #f4f4f9;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    <div class="container">
        <div class="title">Portal IAGen</div>
        <div class="subtitle">Accede a las funcionalidades de IA Generativa de Howden</div>
    </div>
""", unsafe_allow_html=True)
