from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import TokenTextSplitter
import os

def load_and_store_documents(pdf_path, db_path):
    """
    Load a PDF, split into chunks, generate embeddings, and store them in a vector database.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into smaller chunks
    splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)
    chunks = splitter.split_documents(documents)

    # Generate embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Chroma vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    print(f"Embeddings saved to {db_path}")

# Example usage
if __name__ == "__main__":
    pdf_path = "data/Medical_Literature_CFTR.pdf"
    db_path = "embeddings/"
    load_and_store_documents(pdf_path, db_path)