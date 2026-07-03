import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

client = chromadb.PersistentClient(path='./mydb')

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name='documents',
    embedding_function=embedding_function
)

def pdf_loader(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return " ".join([page.page_content for page in pages])

def txt_loader(path):
    loader = TextLoader(path, encoding="utf-8")
    pages = loader.load()
    return " ".join([page.page_content for page in pages])

def create_document_chunks(text_splitter, documents_and_sources):
    all_chunks = []
    for doc_content, source_filename in documents_and_sources:
        chunks = text_splitter.create_documents(
            texts=[doc_content],
            metadatas=[{"source": source_filename}]
        )
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    print("Initializing vector database population...")
    
    try:
        client.delete_collection("documents")
        collection = client.get_or_create_collection(
            name='documents',
            embedding_function=embedding_function
        )
        print("Fresh 'documents' collection created.")
    except Exception as e:
        print(f"Note: Could not clear existing collection: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    documents_folder = 'data' 

    list_of_document_data = []

    if not os.path.exists(documents_folder):
        print(f"The folder '{documents_folder}' does not exist. Please create it and add your files.")
    else:
        for file_name in sorted(os.listdir(documents_folder)):
            full_path = os.path.join(documents_folder, file_name)
            doc_content = None
            try:
                if file_name.endswith('.pdf'):
                    print(f"Processing PDF file: {full_path}")
                    doc_content = pdf_loader(full_path)
                elif file_name.endswith('.txt'):
                    print(f"Processing TXT file: {full_path}")
                    doc_content = txt_loader(full_path)
                
                if doc_content:
                    list_of_document_data.append((doc_content, file_name))
            except Exception as e:
                print(f"⚠️ Failed to process {file_name}: {e}")

    if list_of_document_data:
        all_chunks = create_document_chunks(text_splitter, list_of_document_data)

        chroma_documents = [doc.page_content for doc in all_chunks]
        chroma_ids = [str(i) for i in range(len(chroma_documents))]
        chroma_metadatas = [doc_chunk.metadata for doc_chunk in all_chunks]

        collection.add(
            documents=chroma_documents,
            metadatas=chroma_metadatas,
            ids=chroma_ids
        )
        print(f"Successfully indexed {len(chroma_documents)} chunks into ChromaDB.")
    else:
        print("No documents were loaded. Database remains empty.")