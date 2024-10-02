import io
import langchain
import vertexai
import requests
from google.cloud import storage

from langchain.chains import (
    RetrievalQA,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


storage_client = storage.Client()
bucket_name = 'single-cirrus-435319-f1-bucket'
bucket = storage_client.bucket(bucket_name)

# LLM model
llm = VertexAI(
    model_name="gemini-1.5-flash-001",
    verbose=True,
    project="single-cirrus-435319-f1"
)

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

def main(query: str)->str:

    # Load GOOG's 10K annual report (92 pages).
    url = "https://abc.xyz/assets/investor/static/pdf/20230203_alphabet_10K.pdf"

    # Set up your GCS bucket name and destination file path
    destination_blob_name = 'test_data/20230203_alphabet_10K.pdf'

    # Download the PDF
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Create a temporary file to store the downloaded PDF
    temp_file_path = '/tmp/temp_pdf.pdf'

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(response.content)

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(temp_file_path)
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", ".", " "])
    docs = text_splitter.split_documents(documents)

    vertexai.init(project="single-cirrus-435319-f1")

    # Embedding
    embeddings = VertexAIEmbeddings("text-embedding-004")

    # Store docs in local VectorStore as index
    # it may take a while since API is rate limited
    db = Chroma.from_documents(docs, embeddings)

    # Expose index to the retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create chain to answer questions

    # Uses LLM to synthesize results from the search index.
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa({"query": query})
    return result['result']

def main_vision(query: str, uploaded_file: io.BytesIO):

    vertexai.init(project="single-cirrus-435319-f1")

    blob = bucket.blob(f'test_data/{uploaded_file.name}')

    # Create a temporary file to store the downloaded PDF
    temp_file_path = '/tmp/temp_pdf.pdf'

    # Write the contents of the uploaded file (BytesIO) to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())  # Read the BytesIO content and write to the file

    # Rewind the BytesIO object after reading
    uploaded_file.seek(0)

    blob.upload_from_filename(temp_file_path)

    # Upload the file from BytesIO
    extension = get_mime_type(uploaded_file.name)
    blob.upload_from_file(uploaded_file, rewind=True, content_type=extension)

    imagen = fr"gs://{bucket_name}/test_data/{uploaded_file.name}"

    llm = ChatVertexAI(
    model_name="gemini-1.5-flash-002",
    project="single-cirrus-435319-f1",
    verbose=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un asistente virtual y tu especialidad es utilizar tus capacidades de visión para responder a las preguntas del usuario.
                Tus respuestas deben ser precisas y deben estar basadas únicamente en lo que puedas observar en el documento enviado por el usuario
                """,
            ),
            
            ("human", "{input}"),
            (
                "human",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data}"},
                    }
                ],
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    for chunk in chain.stream({"input": query,"image_data": imagen}):
        yield chunk

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    print(f"LangChain version: {langchain.__version__}")
    print(f"vertexai version: {vertexai.__version__}")
    # Mocked PDF file data (minimal PDF structure in bytes)
    mock_pdf_content = b'%PDF-1.4\n%Mock PDF file content\n%%EOF'
    # Create a mock BytesIO object to simulate file upload
    mock_uploaded_file = io.BytesIO(mock_pdf_content)
    mock_uploaded_file.name = 'mock_file.pdf'  # Simulate the file name
    query = "Summarize the pdf content"
    response = main_vision(query, mock_uploaded_file)
    print(response)