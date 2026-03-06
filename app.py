from src.data_ingestion import  DataIngestionManager
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever

from src.rag_retriever import rag_advanced
from src.llm_interface import LLMInterface
from src.llm_interface import create_rag_prompt
from src.llm_interface import format_rag_response

## Example Usage to load PDF files
if __name__ == "__main__":
    ingestion_manager = DataIngestionManager()
    embedding_manager = EmbeddingManager()
    vector_store_manager = VectorStore(collection_name="construction_pdf_docs", persist_directory="data/vector_store")
    rag_retriever = RAGRetriever(vectorstore=vector_store_manager, embedding_mgr=embedding_manager)

    llm_manager = LLMInterface(model_name="codegemma:7b", temperature=0.2)

    # docs = ingestion_manager.process_all_pdfs("/Users/shreyanshsingh/Documents/PersonalProjects/Procore_P6_Automation/RAG KBs/rag_implementation/raw_docs/")
    # print("=================================================================")
    # print(f"Total documents loaded: {len(docs)}")

    
    # ## Example Usage to load PDF files and split into chunks

    # chunks = ingestion_manager.split_documents_in_chunks(docs)
    # print("=================================================================")
    # print(f"Total chunks created: {len(chunks)}")
    # if chunks:
    #     print(f"Sample chunk content:\n{chunks[0].page_content[:200]}")

    # else:
    #     print("No chunks created!")

    # ## Example Usage to create embeddings for the chunks
    # chunk_embeddings_vector = embedding_manager.generate_embeddings(chunks)
    # print("=================================================================")
    # print(f"Total embeddings created: {len(chunk_embeddings_vector)}")


    # ## Example Usage to add chunks and their embeddings to the vector store
    # try:        
    #     doc_ids = vector_store_manager.add_documents(chunks, chunk_embeddings_vector)
    #     print("=================================================================")
    #     print(f"Total documents added to vector store: {len(doc_ids)}")
    # except Exception as e:
    #     print(f"Error adding documents to vector store: {e}")

    question = input("Enter your question about construction project schedules and critical paths: ")
    role_description = "You are a helpful assistant that provides information about construction project schedules and critical paths based on the retrieved documents."
    instructions = "Provide a concise answer based on the retrieved context. If the information is not available, respond with 'Information not found in the provided context.'"


    #Step-1 - Retrieve relevant context from vector store using RAGRetriever
    retrieved_context = rag_advanced(question,rag_retriever, top_k=3)
    print("==================================================================================================================================")
    # print(f"Retrieved {len(retrieved_context)} relevant documents for the query.")
    context_text = retrieved_context.get('context')
    clean_context = ' '.join(context_text.split())
    # print(clean_context)
    # print(retrieved_context.get('sources'))
    # for i, doc in enumerate(retrieved_context):
    #     print(f"\nDocument {i+1} (Similarity Score: {doc['similarity_score']:.4f}):\n{doc['content'][:500]}...")

    #Step-2 - Create a RAG prompt for the LLM
    rag_prompt = create_rag_prompt(role_description, question, clean_context, instructions)

    #Step-3 - Generate response from LLM based on the RAG prompt
    llm_response = llm_manager.generate_response(rag_prompt)
    # print("=================================================================")
    # print(f"LLM Response:\n{llm_response}")

    #Step-4 - Format the final response to the user (if needed, based on instructions)
    final_response = format_rag_response(question, llm_response,retrieved_context)
    print("==================================================================================================================================")
    print(f"Role:\n{role_description}")
    print("==================================================================================================================================")
    print(f"Final Response to User:\n{final_response}")
    

    