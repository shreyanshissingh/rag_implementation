from src.data_ingestion import  DataIngestionManager
from src.embedding_manager import EmbeddingManager

## Example Usage to load PDF files
if __name__ == "__main__":
    ingestion_manager = DataIngestionManager()
    embedding_manager = EmbeddingManager()
    docs = ingestion_manager.process_all_pdfs("/Users/shreyanshsingh/Documents/PersonalProjects/Procore_P6_Automation/RAG KBs/rag_implementation/raw_docs/")
    print("=================================================================")
    print(f"Total documents loaded: {len(docs)}")

    
    ## Example Usage to load PDF files and split into chunks

    chunks = ingestion_manager.split_documents_in_chunks(docs)
    print("=================================================================")
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print(f"Sample chunk content:\n{chunks[0].page_content[:200]}")

    else:
        print("No chunks created!")

    ## Example Usage to create embeddings for the chunks
    chunk_embeddings_vector = embedding_manager.generate_embeddings(chunks)
    print("=================================================================")
    print(f"Total embeddings created: {len(chunk_embeddings_vector)}")
