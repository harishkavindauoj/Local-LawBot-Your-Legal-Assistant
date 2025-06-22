from typing import List, Dict, Any, Optional
from rag.retriever import DocumentRetriever
from rag.generator import GeminiGenerator
from embeddings.embedding_service import EmbeddingService
from embeddings.vector_store import create_vector_store
from data.data_loader import LegalDataLoader
from utils.logger import logger
from config import config


class RAGPipeline:
    """Complete RAG pipeline for legal question answering."""

    def __init__(self):
        self.embedding_service = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.is_initialized = False
        logger.info("RAG Pipeline created")

    def initialize(self):
        """Initialize all components of the RAG pipeline."""
        try:
            logger.info("Initializing RAG pipeline components...")

            # Initialize embedding service
            self.embedding_service = EmbeddingService()

            # Initialize vector store
            self.vector_store = create_vector_store()

            # Initialize retriever
            self.retriever = DocumentRetriever(self.vector_store, self.embedding_service)

            # Initialize generator
            self.generator = GeminiGenerator()

            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise

    def setup_knowledge_base(self, force_reload: bool = False):
        """Set up the knowledge base with legal documents."""
        if not self.is_initialized:
            self.initialize()

        try:
            # Check if we already have documents
            stats = self.vector_store.get_collection_stats()
            current_doc_count = stats.get("document_count", 0)

            logger.info(f"Current knowledge base contains {current_doc_count} documents")

            if current_doc_count > 0 and not force_reload:
                logger.info("Knowledge base already populated. Use force_reload=True to refresh.")
                return

            if force_reload and current_doc_count > 0:
                logger.info("Force reload requested - clearing existing documents")
                # Use the proper clear method
                self.vector_store.reset_collection()
                logger.info("Existing documents cleared")

            logger.info("Setting up legal knowledge base...")

            # Load and process legal documents
            data_loader = LegalDataLoader()

            # Validate configuration first
            if not data_loader.validate_config():
                logger.error("Configuration validation failed")
                return

            processed_documents = data_loader.get_processed_legal_data(force_reload=force_reload)

            if not processed_documents:
                logger.error("No legal documents loaded - check your dataset configuration")
                return

            # FOR QUICK TESTING - UNCOMMENT THESE LINES:
            # logger.info(f"TESTING MODE: Limiting to first 100 documents")
            # processed_documents = processed_documents[:100]

            # Log what we're about to process
            sources = {}
            for doc in processed_documents:
                source = doc.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1

            logger.info(f"Processing {len(processed_documents)} documents from sources: {sources}")

            # Generate embeddings for documents
            logger.info("Generating embeddings for legal documents...")
            embedded_documents = self.embedding_service.embed_documents(processed_documents)

            logger.info(f"Generated embeddings for {len(embedded_documents)} document chunks")

            # Add documents to vector store WITH BATCHING
            logger.info("Adding documents to vector store...")
            self.vector_store.add_documents(embedded_documents, batch_size=150)  # Specify batch size

            # Verify the addition
            final_stats = self.vector_store.get_collection_stats()
            final_count = final_stats.get("document_count", 0)

            logger.info(f"Knowledge base setup completed successfully")
            logger.info(f"Final document count: {final_count} (added {final_count - current_doc_count})")

        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def ask_legal_question(
            self,
            question: str,
            include_context: bool = True,
            custom_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a legal question through the complete RAG pipeline."""
        if not self.is_initialized:
            self.initialize()

        try:
            logger.info(f"Processing legal question: {question[:100]}...")

            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve_relevant_documents(
                question,
                k=custom_k or config.top_k_retrieval
            )

            if not relevant_docs:
                logger.warning("No relevant documents found for the question")
                return {
                    "question": question,
                    "answer": "I couldn't find relevant legal information to answer your question. Please try rephrasing your question or consult with a qualified attorney.",
                    "sources": [],
                    "confidence": "low",
                    "retrieved_documents_count": 0  # FIXED: Use the correct field name
                }

            # Generate response using retrieved context
            answer = self.generator.generate_legal_response(question, relevant_docs)

            # Prepare response with metadata
            response = {
                "question": question,
                "answer": answer,
                "sources": self._format_sources(relevant_docs),
                "confidence": self._assess_confidence(relevant_docs),
                "retrieved_documents_count": len(relevant_docs)  # FIXED: Use the correct field name
            }

            if include_context:
                response["context_documents"] = [
                    {
                        "content": doc["content"][:200] + "...",
                        "source": doc.get("metadata", {}).get("source", "Unknown"),
                        "similarity_score": doc.get("similarity_score", 0)
                    }
                    for doc in relevant_docs[:3]  # Include top 3 for context
                ]

            logger.info("Legal question processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing legal question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": "error",
                "retrieved_documents_count": 0  # FIXED: Use the correct field name
            }

    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information from retrieved documents."""
        sources = []
        seen_sources = set()

        for doc in documents:
            metadata = doc.get("metadata", {})
            source_info = {
                "title": metadata.get("title", "Unknown Title"),
                "source": metadata.get("source", "Unknown Source"),
                "type": metadata.get("document_type", "legal_document"),
                "similarity_score": round(doc.get("similarity_score", 0), 3)
            }

            # Avoid duplicate sources
            source_key = f"{source_info['title']}_{source_info['source']}"
            if source_key not in seen_sources:
                sources.append(source_info)
                seen_sources.add(source_key)

        return sources

    def _assess_confidence(self, documents: List[Dict[str, Any]]) -> str:
        """Assess confidence level based on retrieved documents."""
        if not documents:
            return "low"

        avg_similarity = sum(doc.get("similarity_score", 0) for doc in documents) / len(documents)

        if avg_similarity >= 0.8:
            return "high"
        elif avg_similarity >= 0.6:
            return "medium"
        else:
            return "low"

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG pipeline."""
        status = {
            "initialized": self.is_initialized,
            "components": {
                "embedding_service": self.embedding_service is not None,
                "vector_store": self.vector_store is not None,
                "retriever": self.retriever is not None,
                "generator": self.generator is not None
            }
        }

        if self.vector_store:
            status["knowledge_base"] = self.vector_store.get_collection_stats()

        return status

    def reset_knowledge_base(self):
        """Reset the knowledge base (useful for testing or updates)."""
        if self.vector_store:
            self.vector_store.reset_collection()
            logger.info("Knowledge base reset")