"""
Data Ingestion Module

Handles PDF loading and document chunking for RAG system.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document


class DataIngestionManager:
    """Manages PDF document loading and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize DataIngestionManager
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_all_pdfs(self, path_pdf_directory: str) -> List[Document]:
        """Process all PDFs in the directory
        
        Args:
            path_pdf_directory: Path to directory containing PDFs
            
        Returns:
            List of loaded documents with metadata
        """
        all_documents = []
        pdf_dir = Path(path_pdf_directory)
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for each_file in pdf_files:
            print(f"Processing {each_file}")
            
            try:
                loader = PyPDFLoader(str(each_file))
                documents = loader.load()
                
                # Add additional metadata to documents
                for doc in documents:
                    doc.metadata['source_file'] = each_file.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_documents.extend(documents)
                print(f" ✔ Loaded {len(documents)} pages")
                
            except Exception as e:
                print(f" ⨯ Error : {e}")
        
        print(f"\n Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents_in_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        split_chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_chunks)} chunks")
        
        if split_chunks:
            print(f"Sample chunk --> \n")
            print(f"{split_chunks[0].page_content[:150]}")
            print(f"{split_chunks[0].metadata}")
        
        return split_chunks
