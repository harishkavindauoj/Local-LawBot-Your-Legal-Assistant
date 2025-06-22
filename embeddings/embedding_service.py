import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from utils.logger import logger
from config import config


class EmbeddingService:
    """Handles text embedding generation using SentenceTransformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        if not texts:
            return []

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            logger.info("Embeddings generated successfully")
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for documents and attach them."""
        texts = [doc["content"] for doc in documents]
        embeddings = self.embed_texts(texts)

        embedded_docs = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc["embedding"] = embedding
            embedded_docs.append(embedded_doc)

        return embedded_docs

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0