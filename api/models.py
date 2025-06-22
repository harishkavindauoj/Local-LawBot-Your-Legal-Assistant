from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class LegalQuestion(BaseModel):
    """Model for legal question requests."""
    question: str = Field(..., min_length=10, max_length=1000, description="The legal question to ask")
    include_context: bool = Field(default=True, description="Whether to include context documents in response")
    max_documents: Optional[int] = Field(default=None, ge=1, le=10,
                                         description="Maximum number of documents to retrieve")


class SourceInfo(BaseModel):
    """Model for source information."""
    title: str
    source: str
    type: str
    similarity_score: float


class ContextDocument(BaseModel):
    """Model for context document information."""
    content: str
    source: str
    similarity_score: float


class LegalResponse(BaseModel):
    """Model for legal question responses."""
    question: str
    answer: str
    sources: List[SourceInfo]
    confidence: str
    # FIXED: Changed to int instead of List[str] since you're returning the count
    retrieved_documents_count: int = Field(default=0, description="Number of documents retrieved")
    # OR if you want to keep the original field name, use:
    # retrieved_documents: int = Field(default=0, description="Number of documents retrieved")
    context_documents: Optional[List[ContextDocument]] = None


class PipelineStatus(BaseModel):
    """Model for pipeline status information."""
    initialized: bool
    components: Dict[str, bool]
    knowledge_base: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    detail: Optional[str] = None
    status_code: int