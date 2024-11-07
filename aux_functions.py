import json
from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3
import re
import numpy as np

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def simulate_model_prediction(input_json):
    """
    Simula la llamada a un modelo de ML ya entrenado para devolver un resultado de clasificación.
    
    Args:
        input_json (str): Cadena JSON con 10 parámetros que serán usados para la predicción.
    
    Returns:
        dict: Resultado simulado de la clasificación.
    """
    
    # Convertir el string JSON a un diccionario de Python
    input_data = json.loads(input_json)

    # Verificar que el JSON tiene exactamente 10 parámetros
    if len(input_data) != 10:
        raise ValueError("El JSON de entrada debe contener exactamente 10 parámetros.")
    
    # Simular el proceso de predicción
    # Aquí podrías realizar cualquier procesamiento necesario
    # Por ahora, devolveremos una clasificación simulada
    simulated_prediction = {
        "predicción": "Clase A",  # Simula una predicción de clase
        "probabilidad": 0.85      # Simula una probabilidad asociada a la predicción
    }
    
    # Devolver el resultado de la simulación
    return simulated_prediction

def get_db_from_uri(uri):
    db = SQLDatabase.from_uri(f"sqlite:///db/{uri}")
    return db

def dividir_en_bloques_por_grupo(resultados_sql, max_filas):
    datos = np.array(resultados_sql, dtype=object)  # Aseguramos un array 2D
    filas_actuales = 0
    filas_omitidas = False
    resultado_final = []
    
    grupos_unicos, indices = np.unique(datos[:, 0], return_index=True)
    
    for i in range(len(grupos_unicos)):
        inicio = indices[i]
        fin = indices[i + 1] if i + 1 < len(indices) else len(datos)
        
        grupo_filas = datos[inicio:fin]
        num_filas_grupo = len(grupo_filas)

        if filas_actuales + num_filas_grupo > max_filas:
            filas_omitidas = True
            break
        
        resultado_final.extend([tuple(fila) for fila in grupo_filas])
        filas_actuales += num_filas_grupo

    return resultado_final, filas_omitidas

def create_few_shots(df = pd.read_excel(f"./few_shots/preguntas_respuestas_NH.xlsx")):
    lista_tuplas = []
    for _, row in df.iterrows():
        if row['Resultado'] != '':
            lista_tuplas.append(("user", row['Pregunta']))
            lista_tuplas.append(("assistant", row['Código SQL']))
    print(lista_tuplas)
    return lista_tuplas

def format_text_for_audio(texto):
    texto = re.sub(r'[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ,.?!:\n ]+', '', texto)
    
    # Formatear títulos para que tengan doble salto de línea al final
    texto = re.sub(r'(\n)([^\s]+.*?):\n', r'\1\2:\n\n', texto)
    
    # Añadir viñetas con tabulación para cada línea de lista detectada y espacio adicional entre apartados
    texto = re.sub(r'(\n)([^\n]+)(\n|$)', r'\1    - \2.\n', texto)
    texto = re.sub(r'\n\n\s*- ', '\n\n\n- ', texto)  # Triple salto de línea entre apartados
    
    # Limpiar puntos y espacios redundantes
    texto = re.sub(r'\.{2,}', '.', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    if texto and texto[-1] not in '.!?':
        texto += '.'
    
    return texto
class DB_Connection:

    def __init__(self):
        self.db = None
        self.db_name = None

    def get_db(self):
        db = SQLDatabase.from_uri(f"sqlite:///db/{self.db_name}")
        return db
    
    def upload_db_from_settings(self, file, table_name, sep, encoding):
        conn = sqlite3.connect(fr'db/{table_name}.db')
        if "csv" in file.name:
            df = pd.read_csv(file, sep=sep, encoding=encoding)
        else:
            df = pd.read_excel(file, sheet_name=sep)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        # Cierra la conexión
        conn.close()
        return df

db_connection = DB_Connection()



