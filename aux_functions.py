from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3
import numpy as np

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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