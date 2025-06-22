from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
from api.models import LegalQuestion, LegalResponse, PipelineStatus
from rag.pipeline import RAGPipeline
from utils.logger import logger
from config import config

# Global pipeline instance
pipeline = RAGPipeline()

# Create router
router = APIRouter()


def get_pipeline() -> RAGPipeline:
    """Dependency to get the RAG pipeline."""
    if not pipeline.is_initialized:
        try:
            pipeline.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise HTTPException(status_code=500, detail="RAG pipeline initialization failed")
    return pipeline


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Local LawBot API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Legal Assistant"
    }


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "lawbot-api"}


@router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Get the current status of the RAG pipeline."""
    try:
        status = pipeline.get_pipeline_status()
        return PipelineStatus(**status)
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline status")


@router.post("/setup", response_model=Dict[str, str])
async def setup_knowledge_base(
        background_tasks: BackgroundTasks,
        force_reload: bool = False,
        pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Set up the knowledge base with legal documents."""
    try:
        background_tasks.add_task(pipeline.setup_knowledge_base, force_reload)
        return {
            "message": "Knowledge base setup started",
            "status": "processing",
            "force_reload": str(force_reload)
        }
    except Exception as e:
        logger.error(f"Error starting knowledge base setup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start knowledge base setup")


@router.post("/ask", response_model=LegalResponse)
async def ask_legal_question(
        question_data: LegalQuestion,
        pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Ask a legal question and get an AI-generated response."""
    try:
        # Ensure knowledge base is set up
        stats = pipeline.get_pipeline_status()
        kb_stats = stats.get("knowledge_base", {})
        if kb_stats.get("document_count", 0) == 0:
            try:
                pipeline.setup_knowledge_base()
            except Exception as setup_error:
                logger.warning(f"Knowledge base setup failed: {setup_error}")

        response = pipeline.ask_legal_question(
            question=question_data.question,
            include_context=question_data.include_context,
            custom_k=question_data.max_documents
        )

        return LegalResponse(**response)

    except Exception as e:
        logger.error(f"Error processing legal question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process legal question: {str(e)}"
        )


@router.post("/reset", response_model=Dict[str, str])
async def reset_knowledge_base(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Reset the knowledge base (useful for testing)."""
    try:
        pipeline.reset_knowledge_base()
        return {"message": "Knowledge base reset successfully", "status": "completed"}
    except Exception as e:
        logger.error(f"Error resetting knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset knowledge base")
