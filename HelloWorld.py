import langchain
import vertexai

from langchain.chains import (
    RetrievalQA,
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():

    # Load GOOG's 10K annual report (92 pages).
    url = "https://abc.xyz/assets/investor/static/pdf/20230203_alphabet_10K.pdf"
    loader = PyPDFLoader(url)
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

    query = "What is the State or other jurisdiction of incorporation or organization of Alphabet Inc.?"
    result = qa({"query": query})
    print(result['result'])

    query = "What is the I.R.S. Employer Identification Number of Alphabet?"
    result = qa({"query": query})
    print(result['result'])

    query = "What is the balance in at the end of the year 2022?"
    result = qa({"query": query})
    print(result['result'])

if __name__ == '__main__':

    print(f"LangChain version: {langchain.__version__}")
    print(f"vertexai version: {vertexai.__version__}")

    main()