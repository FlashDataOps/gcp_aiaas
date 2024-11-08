import json
from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3
import pickle
import dill
import shap
import matplotlib.pyplot as plt
from google.cloud import storage
import vertexai

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def generate_and_upload_shap_local(transaccion):
    id_transaccion = -1
    try:
        # Inicializar Vertex AI
        vertexai.init(project="single-cirrus-435319-f1")
        
        with open('./model/model.pkl', 'rb') as file:
            modelo = dill.load(file)
            
        bucket_name="single-cirrus-435319-f1-bucket"
        
        id_transaccion = transaccion.ID[0]
        # Inicializar el explicador de SHAP
        explainer = shap.TreeExplainer(modelo.model)
        
        # Transformar la transacción según el modelo
        transaccion = modelo.target_encode(modelo.drop_columns(transaccion))
        
        # Obtener los valores de SHAP para la transacción
        shap_values = explainer.shap_values(transaccion)
            
        # Seleccionar los valores SHAP para una clase (por ejemplo, clase 0)
        shap_values_for_class = shap_values[0][:, 0]  # Para clase 0 (ajusta si necesitas la clase 1)
        
        # Asegurarse de que transaccion es una fila individual
        transaccion_values = transaccion.iloc[0].values  # Extrae los valores de la primera fila como numpy array

        # Obtener los nombres de las características
        feature_names = transaccion.columns.tolist()  # Nombres de las columnas del DataFrame
        
        # Crear el objeto Explanation con los nombres de las características
        explanation = shap.Explanation(
            values=shap_values_for_class, 
            base_values=shap_values[0][0, 0],  # Base value de la clase 0
            data=transaccion_values, 
            feature_names=feature_names
        )
        
        # Crear y guardar el gráfico de cascada
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        image_path = f"./plots_shap/shap_local_{id_transaccion}.png"
        fig.savefig(image_path, format="png", dpi=300, bbox_inches="tight")
        #plt.close(fig)  # Cerrar la figura para liberar memoria
        
        destination_blob_name = f"foundations/shap_local_{id_transaccion}.png"
        # Subir la imagen al bucket de Google Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(image_path)

        print(f"El gráfico de SHAP waterfall ha sido guardado y subido a {destination_blob_name} en el bucket {bucket_name}.")
        return True, id_transaccion, fig
    except Exception as e:
        print(f"Error al subir SHAP al bucket -> {e}")
        return False, id_transaccion, None


def simulate_model_prediction(df):
    """
    Simula la llamada a un modelo de ML ya entrenado para devolver un resultado de clasificación.
    
    Args:
        input_json (str): Cadena JSON con 10 parámetros que serán usados para la predicción.
    
    Returns:
        dict: Resultado simulado de la clasificación.
    """
    
    
    with open('./model/model.pkl', 'rb') as file:
        model = dill.load(file)
    
    predictions = model.predict(df)
    # Devolver el resultado de la simulación
    return predictions

def get_db_from_uri(uri):
    db = SQLDatabase.from_uri(f"sqlite:///db/{uri}")
    return db

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

