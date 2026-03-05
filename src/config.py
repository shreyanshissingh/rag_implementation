"""
Configuration Module

Central configuration for the RAG system.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = PROJECT_ROOT / "raw_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Data Ingestion Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector Store Configuration
VECTOR_STORE_COLLECTION_NAME = "construction_pdf_docs"
VECTOR_STORE_PERSIST_DIR = str(VECTOR_STORE_DIR)

# RAG Retrieval Configuration
RAG_TOP_K = 5
RAG_SCORE_THRESHOLD = 0.5

# LLM Configuration
LLM_MODEL_NAME = "codegemma:7b"
LLM_TEMPERATURE = 0.2

# Default LLM Role and Instructions
LLM_ROLE = "helpful assistant who provides accurate and concise answers based on the provided context in Construction Project Management domain"
LLM_INSTRUCTIONS = "The response should be formatted as a Markdown list with clear headings and concise sentences."
