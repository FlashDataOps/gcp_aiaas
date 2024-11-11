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
        
        with open('./model/Model_GCP.pkl', 'rb') as file:
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
        
        destination_blob_name = f"foundations/plots_shap/shap_local_{id_transaccion}.png"
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

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    #generation_match_precondition = 0

    blob.upload_from_file(source_file_name, rewind=True)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def simulate_model_prediction(df):
    """
    Simula la llamada a un modelo de ML ya entrenado para devolver un resultado de clasificación.
    
    Args:
        input_json (str): Cadena JSON con 10 parámetros que serán usados para la predicción.
    
    Returns:
        dict: Resultado simulado de la clasificación.
    """
    
    
    with open('./model/Model_GCP.pkl', 'rb') as file:
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

