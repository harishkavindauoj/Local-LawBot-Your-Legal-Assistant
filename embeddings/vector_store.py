import uuid
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
from utils.logger import logger
from config import config


class ChromaVectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(self, collection_name: str = "legal_documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info("Initializing ChromaDB client")

            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(config.chromadb_persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents for RAG"}
            )

            logger.info(f"ChromaDB collection '{self.collection_name}' ready")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 150):
        """Add documents with embeddings to the vector store in batches."""
        if not documents:
            logger.warning("No documents to add")
            return

        try:
            logger.info(f"Adding {len(documents)} documents to ChromaDB in batches of {batch_size}")

            # Process documents in batches
            total_added = 0
            total_batches = (len(documents) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(documents), batch_size):
                batch_documents = documents[batch_idx:batch_idx + batch_size]
                current_batch_num = (batch_idx // batch_size) + 1

                logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_documents)} documents)")

                # Prepare data for this batch
                ids = []
                embeddings = []
                documents_content = []
                metadatas = []

                for doc in batch_documents:
                    doc_id = doc.get("id") or str(uuid.uuid4())

                    # Handle embedding
                    embedding = doc.get("embedding")
                    if embedding is None:
                        logger.warning(f"Document {doc_id} has no embedding, skipping")
                        continue

                    # Convert numpy array to list if needed
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    ids.append(doc_id)
                    embeddings.append(embedding)
                    documents_content.append(doc["content"])

                    # Prepare metadata (ChromaDB doesn't support nested dicts)
                    metadata = doc.get("metadata", {})
                    flat_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            flat_metadata[key] = value
                        else:
                            flat_metadata[key] = str(value)

                    metadatas.append(flat_metadata)

                # Add this batch to collection
                if ids:  # Only add if we have valid documents
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents_content,
                        metadatas=metadatas
                    )

                    total_added += len(ids)
                    logger.info(f"Batch {current_batch_num}/{total_batches} completed - added {len(ids)} documents")

            logger.info(f"Successfully added {total_added} documents to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            documents = []
            for i in range(len(results["ids"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} similar documents")
            return documents

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "ready"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"status": "error", "error": str(e)}

    def reset_collection(self):
        """Reset/clear the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents for RAG"}
            )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


class FAISSVectorStore:
    """FAISS-based vector store implementation (alternative to ChromaDB)."""

    def __init__(self, dimension: int = 384):
        """Initialize FAISS vector store."""
        import faiss

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents = []  # Store document metadata
        self.doc_ids = []

        logger.info(f"Initialized FAISS index with dimension {dimension}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to FAISS index."""
        embeddings = []

        for doc in documents:
            embedding = doc.get("embedding")
            if embedding is None:
                continue

            if isinstance(embedding, list):
                embedding = np.array(embedding)

            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

            # Store document metadata
            self.documents.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            })
            self.doc_ids.append(doc.get("id", str(len(self.doc_ids))))

        if embeddings:
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Added {len(embeddings)} documents to FAISS index")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["id"] = self.doc_ids[idx]
                doc["similarity_score"] = float(score)
                results.append(doc)

        return results


def create_vector_store(store_type: str = None) -> ChromaVectorStore:
    """Factory function to create vector store based on configuration."""
    store_type = store_type or config.vector_db_type

    if store_type.lower() == "chromadb":
        return ChromaVectorStore()
    elif store_type.lower() == "faiss":
        return FAISSVectorStore()
    else:
        logger.warning(f"Unknown vector store type: {store_type}, defaulting to ChromaDB")
        return ChromaVectorStore()