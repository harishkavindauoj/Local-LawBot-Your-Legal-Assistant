import numpy as np
from typing import List, Dict, Any
from embeddings.embedding_service import EmbeddingService
from embeddings.vector_store import ChromaVectorStore
from utils.logger import logger
from config import config


class DocumentRetriever:
    """Handles document retrieval for RAG pipeline."""

    def __init__(self, vector_store: ChromaVectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        logger.info("Document retriever initialized")

    def retrieve_relevant_documents(
            self,
            query: str,
            k: int = None,
            similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant documents for a given query."""
        k = k or config.top_k_retrieval

        try:
            # Generate embedding for the query
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            query_embedding = self.embedding_service.embed_text(query)

            # Search for similar documents
            similar_docs = self.vector_store.similarity_search(query_embedding, k=k)

            # Filter by similarity threshold
            filtered_docs = [
                doc for doc in similar_docs
                if doc.get("similarity_score", 0) >= similarity_threshold
            ]

            logger.info(f"Retrieved {len(filtered_docs)} relevant documents (threshold: {similarity_threshold})")

            # Sort by similarity score (highest first)
            filtered_docs.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            return filtered_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def retrieve_by_metadata(
            self,
            metadata_filters: Dict[str, Any],
            k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve documents based on metadata filters (if supported by vector store)."""
        try:
            # This is a simplified implementation
            # In a full implementation, you'd query the vector store with metadata filters
            logger.info(f"Retrieving documents with metadata filters: {metadata_filters}")

            # For now, we'll retrieve more documents and filter them
            all_docs = self.vector_store.similarity_search(
                np.random.rand(384),  # Random embedding for broad search
                k=k * 2
            )

            # Filter by metadata
            filtered_docs = []
            for doc in all_docs:
                doc_metadata = doc.get("metadata", {})
                match = True

                for key, value in metadata_filters.items():
                    if key not in doc_metadata or doc_metadata[key] != value:
                        match = False
                        break

                if match:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= k:
                        break

            logger.info(f"Retrieved {len(filtered_docs)} documents matching metadata filters")
            return filtered_docs

        except Exception as e:
            logger.error(f"Error retrieving documents by metadata: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            return self.vector_store.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}