import json
from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3
import os
from google.cloud import storage
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from PyPDF2 import PdfReader, PdfWriter
import traceback
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
class DB_Connection:

    def __init__(self):
        self.db = None
        self.db_name = None

    def get_db(self):
        db = SQLDatabase.from_uri(f"sqlite:///db/{self.db_name}")
        return db
    
    def get_db_summary_of_the_day(self):
        db = SQLDatabase.from_uri(f"sqlite:///db/df_nh_demo.db")
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

def list_blobs(bucket_name="single-cirrus-435319-f1-bucket", folder_name="cv-demo/pdf"):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)

    # Note: The call returns a response only when the iterator is consumed.
    return [blob.name for blob in blobs]

def save_pdf_pages(file, n):
    """Guarda las primeras n páginas de un archivo PDF en un archivo temporal."""
    try:
        # Leer el contenido del archivo PDF original
        reader = PdfReader(file)
        writer = PdfWriter()

        # Agregar las primeras n páginas al nuevo archivo
        for page_num in range(min(n, len(reader.pages))):
            writer.add_page(reader.pages[page_num])

        # Guardar en un archivo temporal
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_pdf.name, 'wb') as temp_file:
            writer.write(temp_file)

        return temp_pdf.name  # Devolver la ruta del archivo temporal
    except Exception as e:
        print(f"Error procesando el archivo PDF: {e}")
        return None
    
def upload_blob(file, folder_name):
    """Sube un archivo al bucket. Si es un PDF, sube las primeras n páginas."""
    temp_file_path = None
    try:
        bucket_name = "single-cirrus-435319-f1-bucket"
        try:
            file_name = os.path.basename(file.name)
        except:
            file_name = os.path.basename(file).replace("temp_data\\", "")
        destination_blob_name = f"{folder_name}/{file_name}"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        
        try:
            # Guardar archivo temporalmente para otros tipos de archivo
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.seek(0)
                temp_file.write(file.read())  # Escribir los datos al archivo temporal
                temp_file_path = temp_file.name
        except:
            temp_file_path = file
        
        # Subir el archivo al bucket de GCS
        blob.upload_from_filename(temp_file_path, content_type=get_mime_type(file_name))
        
        print(f"File {file_name} uploaded to {destination_blob_name}.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error subiendo el archivo: {e}")
        return False

def get_mime_type(filename):

    mime_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".html": "text/html"
    }

    extension = filename[filename.rfind("."):].lower()

    return mime_types.get(extension, "application/octet-stream")

def extract_areas_from_pdf_base64(pdf_path, page_number):
    """
    Extrae 6 áreas específicas (3 tercios verticales, cada uno dividido en superior e inferior) de una página de un PDF
    y devuelve las imágenes en formato data:image/jpeg;base64.
    
    :param pdf_path: Ruta al archivo PDF.
    :param page_number: Número de página (empieza desde 1).
    :return: Lista de strings en formato data:image/jpeg;base64.
    """
    temp_file_path = None
    file_name = pdf_path.name.split(".")[0]
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pdf_path.seek(0)
        temp_file.write(pdf_path.read())  # Escribir los datos al archivo temporal
        temp_file_path = temp_file.name
    # Abrir el PDF
    doc = fitz.open(temp_file_path)
    page = doc[page_number - 1]  # La numeración empieza desde 0 en PyMuPDF

    # Renderizar la página como imagen
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Obtener las dimensiones de la imagen original
    width, height = img.size
    

    # Calcular las divisiones en tercios verticales y mitades horizontales
    third_width = width // 3
    half_height = height // 2

    # Definir las áreas de recorte para los 6 rectángulos
    areas = [
        (0, 0, third_width, half_height),                   # Izquierda_Superior
        (0, half_height, third_width, height),             # Izquierda_Inferior
        (third_width, 0, 2 * third_width, half_height),    # Centro_Superior
        (third_width, half_height, 2 * third_width, height), # Centro_Inferior
        (2 * third_width, 0, width, half_height),          # Derecha_Superior
        (2 * third_width, half_height, width, height)      # Derecha_Inferior
    ]
    
    '''
    # Calcular las divisiones en cuartos horizontales
    quarter_height = height // 4

    # Definir las áreas de recorte para los 4 rectángulos horizontales
    areas = [
        (0, 0, width, quarter_height),                      # Superior
        (0, quarter_height, width, 2 * quarter_height),     # Segundo cuarto
        (0, 2 * quarter_height, width, 3 * quarter_height), # Tercer cuarto
        (0, 3 * quarter_height, width, height)             # Inferior
    ]    
    '''
    # Recortar y almacenar las imágenes como Base64
    base64_images = []
    for index, area in enumerate(areas):
        cropped_image = img.crop(area)
        buffer = io.BytesIO()
        cropped_image.save(buffer, format="PNG")
        cropped_image.save(fr"temp_data\{file_name}_{index}.png")
        
        name_file = pdf_path.name.split(".")[0]
        path_ficha_gcp = f"ufv-demo/ficha-admision/{name_file}"
        print("Buscando si existe fichero en blob")
        lista_blobs = list_blobs(folder_name=path_ficha_gcp)
        print(fr"{path_ficha_gcp}/{file_name}_{index}.png")
        if not fr"{path_ficha_gcp}/{file_name}_{index}.png" in lista_blobs:
            upload_blob(file=fr"temp_data\{file_name}_{index}.png", folder_name=fr"ufv-demo/ficha-admision/{file_name}")
        
        base64_images.append(fr"gs://single-cirrus-435319-f1-bucket/ufv-demo/ficha-admision/{pdf_path.name.split('.')[0]}/{pdf_path.name.split('.')[0]}_{index}.png")
        #encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        #base64_images.append(encoded_image)
        buffer.close()

    doc.close()
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        print("Fichero temporal de imagenes eliminado")
    
    print(base64_images)
    return base64_images