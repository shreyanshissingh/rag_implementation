"""
Main Module

Orchestrates the RAG system workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import DataIngestionManager
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore, check_duplicate_documents
from src.rag_retriever import RAGRetriever, rag_simple, rag_advanced
from src.llm_interface import LLMInterface, create_rag_prompt, format_rag_response
from src import config


class RAGSystem:
    """Main RAG System orchestrator"""
    
    def __init__(self):
        """Initialize the RAG system"""
        print("Initializing RAG System...")
        
        # Initialize components
        self.data_ingestion = DataIngestionManager(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.embedding_mgr = EmbeddingManager(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.vectorstore = VectorStore(
            collection_name=config.VECTOR_STORE_COLLECTION_NAME,
            persist_directory=config.VECTOR_STORE_PERSIST_DIR
        )
        
        self.rag_retriever = RAGRetriever(
            vectorstore=self.vectorstore,
            embedding_mgr=self.embedding_mgr
        )
        
        self.llm_interface = LLMInterface(
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE
        )
        
        print("RAG System initialized successfully ✅\n")
    
    def ingest_documents(self, pdf_directory: str):
        """Ingest documents from a directory
        
        Args:
            pdf_directory: Path to directory containing PDFs
        """
        print(f"Ingesting documents from {pdf_directory}...")
        
        # Process PDFs
        documents = self.data_ingestion.process_all_pdfs(pdf_directory)
        
        # Split into chunks
        chunks = self.data_ingestion.split_documents_in_chunks(documents)
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_mgr.generate_embeddings(texts)
        
        # Add to vector store
        self.vectorstore.add_documents(chunks, embeddings)
        
        print(f"Document ingestion completed! ✅\n")
        return chunks
    
    def check_duplicates(self):
        """Check for duplicate documents in vector store"""
        print("Checking for duplicate documents...")
        duplicates = check_duplicate_documents(self.vectorstore)
        print()
        return duplicates
    
    def reset_vectorstore(self):
        """Reset the vector store"""
        print("Resetting vector store...")
        self.vectorstore.reset_collection()
        print()
    
    def query_simple(self, query: str, top_k: int = 5) -> str:
        """Simple query using RAG
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Context retrieved from RAG
        """
        print(f"Processing query: {query}")
        context = rag_simple(query, self.rag_retriever, top_k=top_k)
        return context
    
    def query_advanced(self, query: str, top_k: int = 5) -> dict:
        """Advanced query using RAG with metadata
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with context, sources, and confidence scores
        """
        print(f"Processing query: {query}")
        result = rag_advanced(query, self.rag_retriever, top_k=top_k)
        return result
    
    def answer_question(self, question: str, use_advanced: bool = True) -> str:
        """Answer a question using RAG and LLM
        
        Args:
            question: User's question
            use_advanced: Whether to use advanced RAG retrieval
            
        Returns:
            Formatted answer with sources if advanced
        """
        # Retrieve context
        if use_advanced:
            rag_result = self.query_advanced(question)
            context = rag_result["context"]
            sources = rag_result.get("sources", [])
        else:
            context = self.query_simple(question)
            sources = []
        
        # Create prompt
        prompt = create_rag_prompt(
            role=config.LLM_ROLE,
            question=question,
            context=context,
            instructions=config.LLM_INSTRUCTIONS
        )
        
        # Generate response
        response = self.llm_interface.generate_response(prompt)
        
        # Format output
        if use_advanced:
            output = format_rag_response(question, response, rag_result)
        else:
            output = format_rag_response(question, response)
        
        return output


def main():
    """Main entry point"""
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Example: Ingest documents (uncomment to use)
        # rag_system.ingest_documents("raw_docs/")
        
        # Example: Simple query
        # result = rag_system.query_simple("RISK-FIN-001: Cost Overruns")
        # print(result)
        
        # Example: Advanced query
        # result = rag_system.query_advanced("RISK-FIN-001: Cost Overruns")
        # print(f"Context: {result['context']}")
        # print(f"Sources: {result['sources']}")
        
        # Example: Answer using LLM (interactive mode)
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            answer = rag_system.answer_question(question, use_advanced=True)
            print("\n" + answer)
    
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
