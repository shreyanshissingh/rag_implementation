"""
Embedding Manager Module

Handles text embedding generation using SentenceTransformer.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    """Generates embeddings from text using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Embedding Manager
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load SentenceTransformer model"""
        try:
            print(f"Loading model : {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} , loaded succesfully ✅")
        except Exception as e:
            print(f" Error in loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of text
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings ✅")
        return embeddings
