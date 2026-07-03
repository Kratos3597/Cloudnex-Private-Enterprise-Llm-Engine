import os
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def run_ingestion():
    print("[Cloudnex AI] Starting multi-format data ingestion pipeline...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(base_dir, "knowledge_base")
    db_storage_dir = os.path.join(base_dir, "db_storage")
    
    if not os.path.exists(knowledge_base_dir) or not os.listdir(knowledge_base_dir):
        print(f"[!] Error: Knowledge base directory is empty.")
        return

    documents = []

    # Format Parsing Directives
    documents.extend(DirectoryLoader(knowledge_base_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True}).load())
    documents.extend(DirectoryLoader(knowledge_base_dir, glob="*.csv", loader_cls=CSVLoader, loader_kwargs={"autodetect_encoding": True}).load())
    documents.extend(DirectoryLoader(knowledge_base_dir, glob="*.pdf", loader_cls=PyPDFLoader).load())
    documents.extend(DirectoryLoader(knowledge_base_dir, glob="*.docx", loader_cls=Docx2txtLoader).load())
    
    try:
        documents.extend(DirectoryLoader(knowledge_base_dir, glob="*.xlsx", loader_cls=UnstructuredExcelLoader).load())
    except Exception:
        pass

    print(f"[+] Total raw document frames extracted: {len(documents)}")

    # Text Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)

    # Local Vector Database Compiling
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_storage_dir)
    print("[SUCCESS] Cloudnex AI Knowledge Base updated! Ready for local queries.")

if __name__ == "__main__":
    run_ingestion()
