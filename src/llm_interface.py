"""
LLM Interface Module

Handles interaction with Large Language Models for RAG.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any


class LLMInterface:
    """Interface for interacting with LLM"""
    
    def __init__(self, model_name: str = "codegemma:7b", temperature: float = 0.2):
        """Initialize LLM Interface
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for model generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM"""
        try:
            print(f"Initializing LLM: {self.model_name}")
            self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)
            print(f"LLM initialized successfully ✅")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from LLM
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error generating response from LLM: {e}")
            raise


def create_rag_prompt(role: str, question: str, context: str, 
                     instructions: str) -> str:
    """Create a RAG prompt for the LLM
    
    Args:
        role: Role description for the LLM
        question: User's question
        context: Retrieved context from RAG
        instructions: Specific instructions for response format
        
    Returns:
        Formatted prompt string
    """
    prompt_template = PromptTemplate(
        input_variables=["role", "question", "rag_context", "instructions"],
        template="""
        {role}. Use the following context to answer the following question:

Context: 
{rag_context}

Question:
{question}

Instructions:
{instructions}

Answer:
"""
    )
    
    formatted_prompt = prompt_template.format(
        role=role,
        question=question,
        instructions=instructions,
        rag_context=context
    )
    
    return formatted_prompt


def format_rag_response(question: str, response: str, 
                       sources_info: Dict[str, Any] = None) -> str:
    """Format RAG response with sources and metadata
    
    Args:
        question: Original question
        response: LLM response
        sources_info: Information about sources used
        
    Returns:
        Formatted response string
    """
    output = f"Question: {question}\n\n"
    output += f"Answer:\n{response}\n"
    
    if sources_info and sources_info.get('sources'):
        output += "\n---\n"
        output += "Sources and Confidence Scores:\n"
        for source in sources_info['sources']:
            output += f"- {source['source']} (Page: {source['page']}) - Confidence: {source['confidence_score']}\n"
    
    return output
