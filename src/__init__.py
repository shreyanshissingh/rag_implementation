"""
RAG Implementation Package

A modular Retrieval-Augmented Generation (RAG) system for processing and querying documents.
"""

from src.data_ingestion import DataIngestionManager
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore, check_duplicate_documents
from src.rag_retriever import RAGRetriever, rag_simple, rag_advanced
from src.llm_interface import LLMInterface, create_rag_prompt, format_rag_response
from src import config

__all__ = [
    "DataIngestionManager",
    "EmbeddingManager",
    "VectorStore",
    "check_duplicate_documents",
    "RAGRetriever",
    "rag_simple",
    "rag_advanced",
    "LLMInterface",
    "create_rag_prompt",
    "format_rag_response",
    "config"
]

__version__ = "1.0.0"
