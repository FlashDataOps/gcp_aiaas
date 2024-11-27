import os
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import re
import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase


# Cargar las variables del archivo .env
load_dotenv()


left_co, cent_co,last_co = st.columns([1,5,1])
with cent_co:
    image_url = "images/botimagen.png"
    st.image(image_url, use_column_width=True)

# Título y subtítulo centrados
st.markdown("""
    <div style='text-align: center;'>
        <h1>📝 Generador de Informe PDF</h1>
        <h3>Sube un archivo Excel y obtén un informe orientado a ventas para una empresa seleccionada</h3>
    </div>
    """, unsafe_allow_html=True)

db_conn = sqlite3.connect("db/database.db")
db = SQLDatabase.from_uri("sqlite:///database.db")

inicio = "SELECT * FROM Sheet1"
df = pd.read_sql_query(inicio, db_conn)

# Verificar si existe una columna de empresa
if 'Nombre' in df.columns:
    # Obtener la lista de empresas únicas
    empresas = df['Nombre'].unique()
    empresa_options = ['Selecciona una empresa'] + list(empresas)
    # Agregar un selectbox para seleccionar la empresa
    selected_empresa = st.sidebar.selectbox("Selecciona una empresa:", empresa_options)

    if selected_empresa != 'Selecciona una empresa':
        st.markdown(f"### Empresa seleccionada: **{selected_empresa}**")

        # Filtrar el DataFrame para la empresa seleccionada
        df_empresa = df[df['Nombre'] == selected_empresa]

        if not df_empresa.empty:
            df_empresa.columns = df_empresa.columns.str.replace('_', ' ')

            # Mostrar una previsualización de los datos
            st.markdown("### 📋 Previsualización de los datos:")
            st.dataframe(df_empresa.head())

            # **Paso de limpieza de datos**: Convertir columnas a tipo numérico
            numeric_columns = ['Prima Neta Total Vendida', 'Prima Neta Xselling Estimada', 'Prima Neta Prospect Estimada']

            for col in numeric_columns:
                if col in df_empresa.columns:
                    df_empresa[col] = df_empresa[col].astype(str)
                    df_empresa[col] = df_empresa[col].str.replace('[^\d,.-]', '', regex=True)
                    df_empresa[col] = df_empresa[col].str.replace(',', '.')
                    df_empresa[col] = pd.to_numeric(df_empresa[col], errors='coerce')

            # Configurar el modelo LLM con LLaMA3-70b-8192
            llm = ChatGroq(model="LLaMA3-70b-8192")

            # Generar información resumida del DataFrame de la empresa
            def generate_summary_info(df):
                info = {}
                info['shape'] = df.shape
                info['columns'] = []
                for col in df.columns:
                    col_info = {}
                    col_info['name'] = col
                    col_info['dtype'] = str(df[col].dtype)
                    # Excluir valores nulos y repetidos en el resumen
                    if df[col].dtype == 'object':
                        col_info['unique_values'] = df[col].unique().tolist()
                        col_info['top_categories'] = df[col].value_counts().head(5).to_dict()
                    elif df[col].dtype == 'datetime64[ns]':
                        col_info['min'] = df[col].min()
                        col_info['max'] = df[col].max()
                    else:
                        if not df[col].isnull().all():
                            col_info['mean'] = float(df[col].mean())
                            col_info['std'] = float(df[col].std())
                            col_info['min'] = float(df[col].min())
                            col_info['max'] = float(df[col].max())
                            col_info['sum'] = float(df[col].sum())
                        else:
                            col_info['mean'] = col_info['std'] = col_info['min'] = col_info['max'] = col_info['sum'] = None
                    info['columns'].append(col_info)
                return info

            info = generate_summary_info(df_empresa)

            # Preparar el texto para el prompt
            prompt_text = f"Tengo datos de la empresa {selected_empresa}. A continuación, un resumen de los datos:\n"

            for col_info in info['columns']:
                prompt_text += f"\nColumna: {col_info['name']}\n"
                # Excluir valores faltantes y repetidos
                if col_info['dtype'] == 'object':
                    unique_vals = col_info.get('unique_values', [])
                    if len(unique_vals) == 1:
                        prompt_text += f"Todos los valores son '{unique_vals[0]}'.\n"
                    else:
                        top_categories = ', '.join([f"{k} ({v})" for k, v in col_info['top_categories'].items()])
                        prompt_text += f"Categorías principales: {top_categories}.\n"
                elif col_info['dtype'] == 'datetime64[ns]':
                    prompt_text += f"Fechas desde {col_info['min'].date()} hasta {col_info['max'].date()}.\n"
                else:
                    if col_info['mean'] is not None:
                        prompt_text += f"Suma total: {col_info['sum']:.2f}. Media: {col_info['mean']:.2f}. Mínimo: {col_info['min']}. Máximo: {col_info['max']}.\n"

            # Crear el prompt para el LLM
            prompt = ChatPromptTemplate.from_messages([
                ('system', f"Eres un asistente para un agente de seguros. Analiza los datos proporcionados y genera un resumen detallado en formato de párrafos explicativos, incluyendo recomendaciones y conclusiones basadas en los datos. No hables sobre valores nulos o repetidos. Proporciona la información de manera natural y profesional."),
                ('user', "{data_info}")
            ])

            # Crear la cadena de procesamiento
            chain = (prompt | llm | StrOutputParser())

            # Obtener la respuesta del LLM
            response = chain.invoke({"data_info": prompt_text})

            # Mostrar el resumen en la aplicación con mejor formato
            st.markdown("### 📝 Resumen generado:")
            st.markdown(f"<div style='text-align: justify;'>{response}</div>", unsafe_allow_html=True)

            # Configurar estilo de seaborn
            sns.set_style('whitegrid')
            sns.set_palette('Set2')

            # Generar gráficos específicos solicitados
            images = []

            # 1. Comparativa de los distintos ramos vendidos
            if 'Ramo Vendido' in df_empresa.columns and 'Prima Neta Total Vendida' in df_empresa.columns:
                ramo_vendido = df_empresa[['Ramo Vendido', 'Prima Neta Total Vendida']].dropna(subset=['Ramo Vendido', 'Prima Neta Total Vendida'])
                ramo_vendido = ramo_vendido.groupby('Ramo Vendido')['Prima Neta Total Vendida'].sum().reset_index()
                if not ramo_vendido.empty:
                    plt.figure(figsize=(8, 6))
                    sns.barplot(data=ramo_vendido, x='Ramo Vendido', y='Prima Neta Total Vendida')
                    plt.title(f'Comparativa de Ramos Vendidos - {selected_empresa}', fontsize=16)
                    plt.xlabel('Ramo Vendido', fontsize=14)
                    plt.ylabel('Prima Neta Total Vendida', fontsize=14)
                    plt.xticks(rotation=45, fontsize=12)
                    plt.tight_layout()
                    # Guardar el gráfico en un objeto BytesIO
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='PNG')
                    plt.close()
                    img_buffer.seek(0)
                    images.append(('Comparativa de Ramos Vendidos', img_buffer))

                    # Mostrar en la aplicación
                    st.markdown("#### Comparativa de los distintos ramos vendidos")
                    st.image(img_buffer)
                    # Mostrar la tabla correspondiente
                    st.dataframe(ramo_vendido, use_container_width=True)

            # 2. Comparativa de ramos propuestos con prima estimada de Xselling
            if 'Ramo Propuesto' in df_empresa.columns and 'Prima Neta Xselling Estimada' in df_empresa.columns:
                df_empresa['Propension de Venta'] = df_empresa['Propension de Venta'].astype(str)
                    # Limpiar la columna 'Propension de Venta' (quitar símbolos % y convertir a float)
                df_empresa['Propension de Venta'] = (
                    df_empresa['Propension de Venta']
                    .str.replace('%', '', regex=False)  # Quitar el símbolo %
                    .str.replace(',', '.', regex=False)  # Reemplazar comas por puntos
                    .apply(lambda x: sum(map(float, re.findall(r'\d+\.\d+', x))) / len(re.findall(r'\d+\.\d+', x)) if isinstance(x, str) and re.findall(r'\d+\.\d+', x) else None)  # Calcular promedio de valores repetidos
                )
                
                # Convertir a numérico, reemplazando errores con NaN
                df_empresa['Propension de Venta'] = pd.to_numeric(df_empresa['Propension de Venta'], errors='coerce')

                # Seleccionar las columnas necesarias y eliminar filas con valores nulos en las columnas relevantes
                ramo_xsell = df_empresa[['Ramo Propuesto', 'Prima Neta Xselling Estimada', 'Propension de Venta']].dropna(
                    subset=['Ramo Propuesto', 'Prima Neta Xselling Estimada']
                )
                
                # Agrupar por 'Ramo Propuesto' y sumar 'Prima Neta Xselling Estimada' y 'Propension de Venta'
                ramo_xsell = ramo_xsell.groupby('Ramo Propuesto').agg({
                    'Prima Neta Xselling Estimada': 'sum',
                    'Propension de Venta': 'mean'  # Usar la media para la propensión de venta
                }).reset_index()
                if not ramo_xsell.empty:
                    plt.figure(figsize=(8, 6))
                    sns.barplot(data=ramo_xsell, x='Ramo Propuesto', y='Prima Neta Xselling Estimada')
                    plt.title(f'Ramos Propuestos vs Prima Neta Xselling Estimada - {selected_empresa}', fontsize=16)
                    plt.xlabel('Ramo Propuesto', fontsize=14)
                    plt.ylabel('Prima Neta Xselling Estimada', fontsize=14)
                    plt.xticks(rotation=45, fontsize=12)
                    plt.tight_layout()
                    # Guardar el gráfico
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='PNG')
                    plt.close()
                    img_buffer.seek(0)
                    images.append(('Ramos Propuestos vs Prima Neta Xselling Estimada', img_buffer))

                    # Mostrar en la aplicación
                    st.markdown("#### Ramos propuestos con la prima estimada de Xselling")
                    st.image(img_buffer)
                    # Mostrar la tabla correspondiente
                    st.dataframe(ramo_xsell, use_container_width=True)
                    

            # 3. Comparativa de ramos propuestos con prima estimada de Prospect
            if 'Ramo Propuesto' in df_empresa.columns and 'Prima Neta Prospect Estimada' in df_empresa.columns:
                    # Limpiar la columna 'Propension de Venta' (quitar símbolos % y convertir a float)
                df_empresa['Propension de Venta'] = df_empresa['Propension de Venta'].astype(str)
                
                df_empresa['Propension de Venta'] = (
                    df_empresa['Propension de Venta']
                    .str.replace('%', '', regex=False)  # Quitar el símbolo %
                    .str.replace(',', '.', regex=False)  # Reemplazar comas por puntos
                    .apply(lambda x: sum(map(float, re.findall(r'\d+\.\d+', x))) / len(re.findall(r'\d+\.\d+', x)) if isinstance(x, str) and re.findall(r'\d+\.\d+', x) else None)  # Calcular promedio de valores repetidos
                )
                
                # Convertir a numérico, reemplazando errores con NaN
                df_empresa['Propension de Venta'] = pd.to_numeric(df_empresa['Propension de Venta'], errors='coerce')

                # Seleccionar las columnas necesarias y eliminar filas con valores nulos en las columnas relevantes
                ramo_prospect = df_empresa[['Ramo Propuesto', 'Prima Neta Prospect Estimada', 'Propension de Venta']].dropna(
                    subset=['Ramo Propuesto', 'Prima Neta Prospect Estimada']
                )
                
                # Agrupar por 'Ramo Propuesto' y sumar 'Prima Neta Prospect Estimada' y 'Propension de Venta'
                ramo_prospect = ramo_prospect.groupby('Ramo Propuesto').agg({
                    'Prima Neta Prospect Estimada': 'sum',
                    'Propension de Venta': 'mean'  # Usar la media para la propensión de venta
                }).reset_index()
                
                if not ramo_prospect.empty:
                    plt.figure(figsize=(8, 6))
                    sns.barplot(data=ramo_prospect, x='Ramo Propuesto', y='Prima Neta Prospect Estimada')
                    plt.title(f'Ramos Propuestos vs Prima Neta Prospect Estimada - {selected_empresa}', fontsize=16)
                    plt.xlabel('Ramo Propuesto', fontsize=14)
                    plt.ylabel('Prima Neta Prospect Estimada', fontsize=14)
                    plt.xticks(rotation=45, fontsize=12)
                    plt.tight_layout()
                    # Guardar el gráfico
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='PNG')
                    plt.close()
                    img_buffer.seek(0)
                    images.append(('Ramos Propuestos vs Prima Neta Prospect Estimada', img_buffer))

                    # Mostrar en la aplicación
                    st.markdown("#### Ramos propuestos con la prima estimada de Prospect")
                    st.image(img_buffer)
                    # Mostrar la tabla correspondiente
                    st.dataframe(ramo_prospect, use_container_width = True)


            # Definir una clase PDF que hereda de FPDF
            class PDF(FPDF):
                def __init__(self, selected_empresa):
                    super().__init__()
                    self.selected_empresa = selected_empresa
                    # Añadir las fuentes Unicode
                    self.add_font('NotoSans', '', 'fonts/NotoSans-Regular.ttf', uni=True)
                    self.add_font('NotoSans-SemiBold', '', 'fonts/NotoSans-SemiBold.ttf', uni=True)

                def header(self):
                    # Set font
                    self.set_font('NotoSans-SemiBold', '', 15)
                    # Título
                    self.cell(0, 10, f'{self.selected_empresa}', ln=1, align='C')
                    self.ln(10)

                def footer(self):
                    # Posición a 1.5 cm del final
                    self.set_y(-15)
                    # Set font sin estilo itálico
                    self.set_font('NotoSans', '', 8)
                    # Número de página
                    self.cell(0, 10, f'Página {self.page_no()}', align='C')

            # Función para generar el informe PDF
            # Función para generar el informe PDF
            def create_pdf(response, images, selected_empresa):
                    pdf = PDF(selected_empresa)
                    pdf.alias_nb_pages()
                    pdf.add_page()

                    # Agregar portada
                    pdf.set_font('NotoSans-SemiBold', '', 20)
                    pdf.cell(0, 80, f'{selected_empresa}', ln=1, align='C')
                    pdf.set_font('NotoSans', '', 12)
                    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y")}', ln=1, align='C')
                    pdf.add_page()

                    # Agregar el resumen
                    pdf.set_font('NotoSans-SemiBold', '', 16)
                    pdf.cell(0, 10, 'Resumen de Datos', ln=1)
                    pdf.set_font('NotoSans', '', 12)
                    pdf.multi_cell(0, 10, response)
                    pdf.ln(5)

                    # Calcular el ancho efectivo de la página
                    effective_page_width = pdf.w - pdf.l_margin - pdf.r_margin


                    # Agregar los gráficos al PDF
                    for title, img_buffer in images:
                        pdf.add_page()
                        pdf.set_font('NotoSans-SemiBold', '', 14)
                        pdf.cell(0, 10, title, ln=1, align='C')
                        pdf.ln(10)
                        pdf.image(img_buffer, x=15, y=None, w=effective_page_width - 30)

                    # Guardar el PDF en un objeto BytesIO
                    pdf_buffer = BytesIO()
                    pdf.output(pdf_buffer)
                    pdf_buffer.seek(0)

                    return pdf_buffer

            # Generar el informe PDF utilizando la función
            pdf_buffer = create_pdf(response, images, selected_empresa)

            # Proporcionar un botón de descarga para el informe PDF
            st.markdown("### 📥 Descargar Informe PDF")
            st.download_button(
                label="Descargar Informe",
                data=pdf_buffer,
                file_name=f'informe_{selected_empresa}.pdf',
                mime='application/pdf'
            )

        else:
            st.warning(f"No hay datos disponibles para la empresa {selected_empresa}.")
    else:
        st.info("Por favor, selecciona una empresa para continuar.")
else:
    st.error("La columna 'Nombre' no se encontró en el archivo Excel. Por favor, asegúrate de que exista una columna llamada 'Nombre'.")
