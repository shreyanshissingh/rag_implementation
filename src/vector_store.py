"""
Vector Store Module

Manages ChromaDB vector store for document embeddings.
"""

import os
import uuid
import numpy as np
import chromadb
from typing import List, Any, Dict
from langchain_core.documents import Document


class VectorStore:
    """Manages ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "construction_pdf_docs", 
                 persist_directory: str = "data/vector_store"):
        """Initialize Vector Store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Local folder where ChromaDB will be stored
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialise_store()
    
    def _initialise_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Doc embeddings for RAG"},
            )
            print(f"Initialized vector store. Collection Name  : {self.collection_name}")
            print(f" Collection Docs Count  : {self.collection.count()}")
        except Exception as e:
            print(f"Error in creating Vector Store : {e}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> List[str]:
        """Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents to add
            embeddings: numpy array of embeddings
            
        Returns:
            List of document ids
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must be the same")
        
        print(f"Adding {len(documents)} documents to the vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            return ids
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Doc embeddings for RAG"},
            )
            print(f"Collection reset. New count: {self.collection.count()}")
        except Exception as e:
            print(f"Error resetting collection: {e}")
            raise


def check_duplicate_documents(vectorstore: VectorStore) -> List[Dict[str, Any]]:
    """Check for duplicate documents in the vector store
    
    Args:
        vectorstore: VectorStore object to check
        
    Returns:
        List of duplicate document information
    """
    try:
        # Get all documents from collection
        all_docs = vectorstore.collection.get(include=["documents", "metadatas"])
        
        documents = all_docs['documents']
        metadatas = all_docs['metadatas']
        ids = all_docs['ids']
        
        # Create a dictionary to track content hashes
        content_hashes = {}
        duplicates = []
        
        for doc_id, content, metadata in zip(ids, documents, metadatas):
            # Create a hash of content
            content_hash = hash(content)
            
            if content_hash in content_hashes:
                duplicates.append({
                    "duplicate_id": doc_id,
                    "original_id": content_hashes[content_hash]["id"],
                    "source": metadata.get('source_file', 'unknown'),
                    "page": metadata.get('page', 'unknown')
                })
            else:
                content_hashes[content_hash] = {"id": doc_id, "content": content[:100]}
        
        print(f"Total documents: {len(documents)}")
        print(f"Unique documents: {len(content_hashes)}")
        print(f"Duplicate documents found: {len(duplicates)}")
        
        if duplicates:
            print("\nDuplicate details:")
            for dup in duplicates[:10]:  # Show first 10
                print(f"  Duplicate ID: {dup['duplicate_id']}, Original ID: {dup['original_id']}, Source: {dup['source']}")
        
        return duplicates
    except Exception as e:
        print(f"Error checking duplicates: {e}")
        raise
