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
        db = SQLDatabase.from_uri(f"sqlite:///db/nestle_db.db")
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


def summary_of_the_day_query(db):
    
    query_business_date = """SELECT Business_Date FROM df_nh_demo LIMIT 1;"""
    res_business_date = db.run(query_business_date)
    
    query_TREV = """
                    SELECT SUM(Actuals) AS Total_Actuals, SUM(OTB) AS Total_OTB, SUM(Pick_Up) AS Total_PickUp, SUM(Actuals + OTB + Forecast) AS TREV, (SUM(Actuals + OTB) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_expected_revenue 
                    FROM df_nh_demo;
                    """
    res_TREV = db.run(query_TREV)
    
    query_RREV = """SELECT 
                        SUM(Actuals) AS Total_Actuals_RREV, 
                        SUM(OTB) AS Total_OTB_RREV, 
                        SUM(Pick_Up) AS Total_PickUp_RREV, 
                        SUM(Actuals + OTB + Forecast) AS RREV, 
                        (SUM(Actuals + OTB) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_expected_revenue_RREV 
                    FROM 
                        df_nh_demo
                    WHERE 
                        Metric = 'RP';
                    """
    res_RREV = db.run(query_RREV)
    
    query_OREV = """SELECT 
                        SUM(Actuals) AS Total_Actuals_OREV, 
                        SUM(OTB) AS Total_OTB_OREV, 
                        SUM(Pick_Up) AS Total_PickUp_OREV, 
                        SUM(Actuals + OTB + Forecast) AS OREV, 
                        (SUM(Actuals + OTB) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_expected_revenue_OREV 
                    FROM 
                        df_nh_demo
                    WHERE 
                        Metric != 'RP';
                    """
    res_OREV = db.run(query_OREV)
    
    query_pctg_RREV_OREV = """SELECT 
                    -- Total Revenue (TREV)
                    SUM(Actuals + OTB + Forecast) AS TREV,
                    
                    -- Room Revenue (RREV)
                    SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS RREV,
                    
                    -- Other Revenue (OREV)
                    SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS OREV,
                    
                    -- Percentages
                    (SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_RREV_of_TREV,
                    (SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_OREV_of_TREV
                FROM 
                    df_nh_demo;"""
      
    res_pctg_RREV_OREV = db.run(query_pctg_RREV_OREV)
                  
    query_pctg_actuals_OTB_PickUp = """SELECT 
                                        -- TREV, RREV y OREV
                                        SUM(Actuals + OTB + Forecast) AS TREV,
                                        SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS RREV,
                                        SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS OREV,
                                        
                                        -- Porcentajes de Actuals
                                        (SUM(Actuals) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_Actuals_of_TREV,
                                        (SUM(CASE WHEN Metric = 'RP' THEN Actuals ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_Actuals_of_RREV,
                                        (SUM(CASE WHEN Metric != 'RP' THEN Actuals ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_Actuals_of_OREV,

                                        -- Porcentajes de OTB
                                        (SUM(OTB) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_OTB_of_TREV,
                                        (SUM(CASE WHEN Metric = 'RP' THEN OTB ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_OTB_of_RREV,
                                        (SUM(CASE WHEN Metric != 'RP' THEN OTB ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_OTB_of_OREV,

                                        -- Porcentajes de PickUp
                                        (SUM(Pick_Up) * 1.0) / SUM(Actuals + OTB + Forecast) AS perc_PickUp_of_TREV,
                                        (SUM(CASE WHEN Metric = 'RP' THEN Pick_Up ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric = 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_PickUp_of_RREV,
                                        (SUM(CASE WHEN Metric != 'RP' THEN Pick_Up ELSE 0 END) * 1.0) / SUM(CASE WHEN Metric != 'RP' THEN Actuals + OTB + Forecast ELSE 0 END) AS perc_PickUp_of_OREV
                                    FROM 
                                        df_nh_demo;
                                                """
    res_pctg_actuals_OTB_PickUp = db.run(query_pctg_actuals_OTB_PickUp)
                                          
    query_hotelBU = """SELECT 
                    Hotel_BU, SUM(Actuals), SUM(OTB), SUM(Pick_Up)
                FROM 
                    df_nh_demo
                GROUP BY Hotel_BU;
                """
    res_hotelBU = db.run(query_hotelBU)
    
    query_country = """SELECT 
                    Hotel_Country, SUM(Actuals), SUM(OTB), SUM(Pick_Up)
                FROM 
                    df_nh_demo
                GROUP BY Hotel_Country;
                """
    res_country = db.run(query_country)
              
    query_subBU = """
                    SELECT 
                        Hotel_SubBU AS Hotel_SubBU,  -- País o SubBU
                        SUM(Forecast) AS Total_Forecast
                    FROM 
                        df_nh_demo
                    GROUP BY 
                        Hotel_SubBU
                    ORDER BY
                        Total_Forecast DESC;  """
                        
    res_subBU = db.run(query_subBU)
                  
    query_metrics="""
                    -- Calcular el desempeño de todas las métricas disponibles
                    WITH Revenue_Metrics AS (
                        SELECT 
                            Metric,  -- Todas las métricas disponibles (ej. RP, BKF, FPB, etc.)
                            
                            -- Calcular los valores totales por cada métrica
                            SUM(Actuals) AS Total_Actuals,
                            SUM(OTB) AS Total_OTB,
                            SUM(Pick_Up) AS Total_PickUp,
                            SUM(Forecast) AS Total_Forecast,
                            
                            -- Cálculos de porcentajes para evaluar el desempeño
                            (SUM(Pick_Up) * 1.0) / NULLIF(SUM(Forecast), 0) AS Perc_PickUp_to_Forecast,
                            (SUM(OTB) * 1.0) / NULLIF(SUM(Forecast), 0) AS Perc_OTB_to_Forecast,
                            (SUM(Actuals) * 1.0) / NULLIF(SUM(Forecast), 0) AS Perc_Actuals_to_Forecast,
                            ((SUM(Actuals) + SUM(OTB)) * 1.0) / NULLIF(SUM(Forecast), 0) AS Perc_Actuals_OTB_to_Forecast
                        FROM 
                            df_nh_demo
                        GROUP BY 
                            Metric  -- Agrupamos solo por la métrica
                    )

                    -- Seleccionar y mostrar las métricas con sus respectivos cálculos
                    SELECT 
                        Metric,
                        Total_Actuals,
                        Total_OTB,
                        Total_PickUp,
                        Total_Forecast,
                        Perc_PickUp_to_Forecast,
                        Perc_OTB_to_Forecast,
                        Perc_Actuals_to_Forecast,
                        Perc_Actuals_OTB_to_Forecast
                    FROM 
                        Revenue_Metrics
                    ORDER BY 
                        CASE 
                            WHEN Metric = 'RP' THEN 1
                            WHEN Metric = 'BKF' THEN 2
                            WHEN Metric = 'FPB' THEN 3
                            WHEN Metric = 'RREV' THEN 4
                            ELSE 5  -- Ordenamos en el orden que prefieras o por defecto
                        END;

                    """
    
    res_metric = db.run(query_metrics)
    
    return res_business_date, res_TREV, res_RREV, res_OREV, res_pctg_RREV_OREV, res_pctg_actuals_OTB_PickUp, res_hotelBU, res_subBU, res_country, res_metric