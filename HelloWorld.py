import langchain
import vertexai
import requests
from google.cloud import storage

from langchain.chains import (
    RetrievalQA,
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

storage_client = storage.Client()

def main(query: str)->str:

    # Load GOOG's 10K annual report (92 pages).
    url = "https://abc.xyz/assets/investor/static/pdf/20230203_alphabet_10K.pdf"
    # Set up your GCS bucket name and destination file path
    bucket_name = 'single-cirrus-435319-f1-bucket'
    destination_blob_name = 'test_data/20230203_alphabet_10K.pdf'

    # Download the PDF
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Create a temporary file to store the downloaded PDF
    temp_file_path = '/tmp/temp_pdf.pdf'
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(response.content)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(temp_file_path)
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    vertexai.init(project="single-cirrus-435319-f1")

    # LLM model
    llm = VertexAI(
        model_name="gemini-1.5-flash-001",
        verbose=True,
        project="single-cirrus-435319-f1"
    )

    # Embedding
    embeddings = VertexAIEmbeddings("text-embedding-004")

    # Store docs in local VectorStore as index
    # it may take a while since API is rate limited
    db = Chroma.from_documents(docs, embeddings)

    # Expose index to the retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create chain to answer questions

    # Uses LLM to synthesize results from the search index.
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    
    result = qa({"query": query})
    return result['result']

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    print(f"LangChain version: {langchain.__version__}")
    print(f"vertexai version: {vertexai.__version__}")
    query = "What is the State or other jurisdiction of incorporation or organization of Alphabet Inc.?"
    response = main(query)
    print(response)