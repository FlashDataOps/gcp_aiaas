import langchain
import vertexai

from langchain.chains import (
    ConversationChain,
    LLMChain,
    RetrievalQA,
    SimpleSequentialChain,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAI, VertexAIEmbeddings
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

if __name__ == '__main__':

    print(f"LangChain version: {langchain.__version__}")
    print(f"vertexai version: {vertexai.__version__}")

    main()