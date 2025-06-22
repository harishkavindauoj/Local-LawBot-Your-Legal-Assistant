import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import logger


class TextProcessor:
    """Handles text preprocessing and chunking for legal documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'§]', '', text)

        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)

        return text.strip()

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text:
            return []

        cleaned_text = self.clean_text(text)
        chunks = self.text_splitter.split_text(cleaned_text)

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue

            chunk_metadata = {
                "chunk_id": i,
                "chunk_size": len(chunk),
                "source": metadata.get("source", "unknown") if metadata else "unknown"
            }

            if metadata:
                chunk_metadata.update(metadata)

            chunked_docs.append({
                "content": chunk,
                "metadata": chunk_metadata
            })

        logger.info(f"Created {len(chunked_docs)} chunks from text of length {len(text)}")
        return chunked_docs

    def preprocess_legal_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Preprocess a legal document with domain-specific handling."""
        text = document.get("text", "")

        # Extract metadata
        metadata = {
            "title": document.get("title", ""),
            "source": document.get("source", ""),
            "jurisdiction": document.get("jurisdiction", ""),
            "document_type": document.get("type", "legal_document"),
            "date": document.get("date", "")
        }

        return self.chunk_text(text, metadata)