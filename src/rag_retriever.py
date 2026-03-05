"""
RAG Retriever Module

Implements Retrieval-Augmented Generation using vector store.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager


class RAGRetriever:
    """Retrieves context from vector store using RAG"""
    
    def __init__(self, vectorstore: VectorStore, embedding_mgr: EmbeddingManager):
        """Initialize RAGRetriever
        
        Args:
            vectorstore: VectorStore object
            embedding_mgr: EmbeddingManager object
        """
        self.vectorstore = vectorstore
        self.embedding_mgr = embedding_mgr
    
    def retrieve_context(self, query: str, top_k: int = 5, 
                        score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Retrieve context from vector store
        
        Args:
            query: Query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score for results
            
        Returns:
            List of documents and their similarity scores
        """
        # Generate embeddings for query
        query_embedding = self.embedding_mgr.generate_embeddings([query])[0]
        
        try:
            # Query vector store
            results = self.vectorstore.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                doc_embeddings = results['embeddings'][0]
                ids = results['ids'][0]
                
                # Compute cosine similarities
                similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                
                for i, (doc, metadata, sim, doc_id) in enumerate(zip(
                    documents, metadatas, similarities, ids)):
                    
                    if sim > score_threshold:
                        retrieved_docs.append({
                            "content": doc,
                            "metadata": metadata,
                            "id": doc_id,
                            "similarity_score": sim
                        })
            else:
                print("No results found in Vector Store")
            
            return retrieved_docs
        except Exception as e:
            print(f"Error in retrieving context from Vector Store: {e}")
            raise


def rag_simple(query: str, rag_retriever: RAGRetriever, top_k: int = 5) -> str:
    """Simple RAG implementation to retrieve context for a query
    
    Args:
        query: Query string
        rag_retriever: RAGRetriever object
        top_k: Number of results to retrieve
        
    Returns:
        Concatenated context from retrieved documents
    """
    retrieved_docs = rag_retriever.retrieve_context(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else ""
    
    if not context:
        print("No relevant context found for the query.")
    return context


def rag_advanced(query: str, rag_retriever: RAGRetriever, top_k: int = 5) -> Dict[str, Any]:
    """Advanced RAG implementation that retrieves context and metadata
    
    Args:
        query: Query string
        rag_retriever: RAGRetriever object
        top_k: Number of results to retrieve
        
    Returns:
        Dictionary with context, sources and confidence scores
    """
    retrieved_docs = rag_retriever.retrieve_context(query, top_k=top_k)
    
    context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else ""
    
    # Extract source and confidence information
    sources_and_scores = []
    if retrieved_docs:
        for doc in retrieved_docs:
            sources_and_scores.append({
                "source": doc["metadata"].get("source_file", "unknown"),
                "page": doc["metadata"].get("page", "unknown"),
                "confidence_score": round(doc["similarity_score"], 4)
            })
    
    if not context:
        print("No relevant context found for the query.")
    
    return {
        "context": context,
        "sources": sources_and_scores,
        "num_documents_retrieved": len(retrieved_docs)
    }
