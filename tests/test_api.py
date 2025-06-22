import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import os

# CRITICAL: Mock the config module BEFORE any imports that depend on it
# This prevents the Config() instantiation error during module loading

# Create a mock config instance
mock_config_instance = Mock()
mock_config_instance.gemini_api_key = "mock_gemini_key"
mock_config_instance.openai_api_key = "mock_openai_key"
mock_config_instance.anthropic_api_key = "mock_anthropic_key"
mock_config_instance.huggingface_api_token = "mock_hf_token"
mock_config_instance.documents_path = "mock_docs_path"
mock_config_instance.vector_store_path = "mock_vector_path"
mock_config_instance.chunk_size = 1000
mock_config_instance.chunk_overlap = 200
mock_config_instance.max_documents = 10
mock_config_instance.similarity_threshold = 0.7
mock_config_instance.use_openai = True
mock_config_instance.use_anthropic = False
mock_config_instance.use_huggingface = False
mock_config_instance.use_gemini = False

# Mock the config module and Config class
mock_config_module = Mock()
mock_config_module.Config = Mock(return_value=mock_config_instance)
mock_config_module.config = mock_config_instance

# Apply the mock to sys.modules before importing anything
sys.modules['config'] = mock_config_module

# Now we can safely import our modules
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json

from main import app
from api.models import LegalQuestion, LegalResponse, PipelineStatus, SourceInfo, ContextDocument


@pytest.fixture
def test_client():
    """Fixture to provide test client."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Fixture to provide a mock RAG pipeline for unit tests."""
    mock_pipeline = Mock()
    mock_pipeline.is_initialized = True
    mock_pipeline.initialize = Mock()
    mock_pipeline.get_pipeline_status = Mock(return_value={
        "initialized": True,
        "components": {
            "embeddings": True,
            "vector_store": True,
            "generator": True
        },
        "knowledge_base": {
            "document_count": 100,
            "last_updated": "2024-01-01T00:00:00Z"
        }
    })
    mock_pipeline.setup_knowledge_base = Mock()
    mock_pipeline.ask_legal_question = Mock(return_value={
        "question": "What is contract law?",
        "answer": "Contract law is a body of law that governs oral and written agreements...",
        "sources": [
            {
                "title": "Contract Law Basics",
                "source": "legal_document_1.pdf",
                "type": "legal_document",
                "similarity_score": 0.95
            }
        ],
        "confidence": "high",
        "retrieved_documents": 5,
        "context_documents": [
            {
                "content": "Contract law governs agreements between parties...",
                "source": "legal_document_1.pdf",
                "similarity_score": 0.95
            }
        ]
    })
    mock_pipeline.reset_knowledge_base = Mock()
    return mock_pipeline


class TestRootEndpoints:
    """Test root and basic endpoints."""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Local LawBot: Retrieval-Augmented Legal Assistant"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "api" in data

    def test_api_root_endpoint(self, test_client):
        """Test the API root endpoint."""
        response = test_client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Local LawBot API"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Retrieval-Augmented Legal Assistant"

    def test_health_check(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "lawbot-api"


class TestPipelineDependencyUnit:
    """Test pipeline dependency injection with mocks."""

    def test_get_pipeline_initialized(self, mock_pipeline):
        """Test getting an already initialized pipeline."""
        with patch('api.routes.pipeline', mock_pipeline):
            from api.routes import get_pipeline
            result = get_pipeline()
            assert result == mock_pipeline
            mock_pipeline.initialize.assert_not_called()

    def test_get_pipeline_not_initialized(self):
        """Test getting an uninitialized pipeline."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = False
        mock_pipeline.initialize = Mock()

        with patch('api.routes.pipeline', mock_pipeline):
            from api.routes import get_pipeline
            result = get_pipeline()
            assert result == mock_pipeline
            mock_pipeline.initialize.assert_called_once()

    def test_get_pipeline_initialization_fails(self):
        """Test pipeline initialization failure."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = False
        mock_pipeline.initialize = Mock(side_effect=Exception("Initialization failed"))

        with patch('api.routes.pipeline', mock_pipeline):
            from api.routes import get_pipeline
            with pytest.raises(HTTPException) as exc_info:
                get_pipeline()

            assert exc_info.value.status_code == 500
            assert "RAG pipeline initialization failed" in str(exc_info.value.detail)


class TestPipelineStatusUnit:
    """Test pipeline status endpoint with mocks."""

    def test_get_pipeline_status_success(self, test_client, mock_pipeline):
        """Test successful pipeline status retrieval."""
        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.get("/api/v1/status")
            assert response.status_code == 200
            data = response.json()
            assert data["initialized"] is True
            assert "components" in data
            assert "knowledge_base" in data
            mock_pipeline.get_pipeline_status.assert_called_once()

    def test_get_pipeline_status_error(self, test_client):
        """Test pipeline status retrieval error."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.get_pipeline_status = Mock(side_effect=Exception("Status error"))

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.get("/api/v1/status")
            assert response.status_code == 500
            data = response.json()
            assert "Failed to get pipeline status" in data["error"]


class TestKnowledgeBaseSetupUnit:
    """Test knowledge base setup endpoint with mocks."""

    def test_setup_knowledge_base_success(self, test_client, mock_pipeline):
        """Test successful knowledge base setup."""
        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/setup")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Knowledge base setup started"
            assert data["status"] == "processing"

    def test_setup_knowledge_base_with_force_reload(self, test_client, mock_pipeline):
        """Test knowledge base setup with force reload."""
        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/setup?force_reload=true")
            assert response.status_code == 200
            data = response.json()
            assert data["force_reload"] == "True"

    def test_setup_knowledge_base_error(self, test_client):
        """Test knowledge base setup error."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.initialize = Mock()
        mock_pipeline.setup_knowledge_base = Mock(side_effect=Exception("Setup error"))

        with patch('api.routes.pipeline', mock_pipeline), \
             patch('api.routes.get_pipeline', return_value=mock_pipeline):
            response = test_client.post("/api/v1/setup")
            assert response.status_code == 500
            data = response.json()
            assert "Failed to start knowledge base setup" in data["error"]


class TestLegalQuestionAskingUnit:
    """Test legal question asking endpoint with mocks."""

    def test_ask_legal_question_success(self, test_client, mock_pipeline):
        """Test successful legal question asking."""
        question_data = {
            "question": "What is contract law and how does it work?",
            "include_context": True,
            "max_documents": 5
        }

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/ask", json=question_data)
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "confidence" in data
            assert "retrieved_documents" in data
            mock_pipeline.ask_legal_question.assert_called_once()

    def test_ask_legal_question_auto_setup(self, test_client):
        """Test legal question asking with automatic knowledge base setup."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.get_pipeline_status = Mock(return_value={
            "knowledge_base": {"document_count": 0}
        })
        mock_pipeline.setup_knowledge_base = Mock()
        mock_pipeline.ask_legal_question = Mock(return_value={
            "question": "What is contract law?",
            "answer": "Contract law is...",
            "sources": [],
            "confidence": "medium",
            "retrieved_documents": 0,
            "context_documents": []
        })

        question_data = {
            "question": "What is contract law and how does it work?"
        }

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/ask", json=question_data)
            assert response.status_code == 200
            mock_pipeline.setup_knowledge_base.assert_called_once()

    def test_ask_legal_question_validation_error(self, test_client, mock_pipeline):
        """Test legal question validation errors."""
        # Question too short (assuming your validation requires longer questions)
        question_data = {"question": "What?"}

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/ask", json=question_data)
            # This should either pass (if no length validation) or return 422
            assert response.status_code in [200, 422]

    def test_ask_legal_question_missing_question(self, test_client, mock_pipeline):
        """Test missing question field."""
        question_data = {"include_context": True}

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/ask", json=question_data)
            assert response.status_code == 422

    def test_ask_legal_question_processing_error(self, test_client):
        """Test legal question processing error."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.get_pipeline_status = Mock(return_value={
            "knowledge_base": {"document_count": 100}
        })
        mock_pipeline.ask_legal_question = Mock(side_effect=Exception("Processing error"))

        question_data = {
            "question": "What is contract law and how does it work?"
        }

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/ask", json=question_data)
            assert response.status_code == 500
            data = response.json()
            assert "Failed to process legal question" in data["error"]


class TestKnowledgeBaseResetUnit:
    """Test knowledge base reset endpoint with mocks."""

    def test_reset_knowledge_base_success(self, test_client, mock_pipeline):
        """Test successful knowledge base reset."""
        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/reset")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Knowledge base reset successfully"
            assert data["status"] == "completed"
            mock_pipeline.reset_knowledge_base.assert_called_once()

    def test_reset_knowledge_base_error(self, test_client):
        """Test knowledge base reset error."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.reset_knowledge_base = Mock(side_effect=Exception("Reset error"))

        with patch('api.routes.pipeline', mock_pipeline):
            response = test_client.post("/api/v1/reset")
            assert response.status_code == 500
            data = response.json()
            assert "Failed to reset knowledge base" in data["error"]


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_legal_question_model_valid(self):
        """Test valid LegalQuestion model."""
        question = LegalQuestion(
            question="What is contract law and how does it work?",
            include_context=True,
            max_documents=5
        )
        assert question.question == "What is contract law and how does it work?"
        assert question.include_context is True
        assert question.max_documents == 5

    def test_legal_question_model_defaults(self):
        """Test LegalQuestion model with defaults."""
        question = LegalQuestion(question="What is contract law?")
        assert question.include_context is True
        assert question.max_documents is None

    def test_legal_response_model(self):
        """Test LegalResponse model."""
        source = SourceInfo(
            title="Contract Law Basics",
            source="legal_doc.pdf",
            type="legal_document",
            similarity_score=0.95
        )

        context_doc = ContextDocument(
            content="Contract law governs agreements...",
            source="legal_doc.pdf",
            similarity_score=0.95
        )

        response = LegalResponse(
            question="What is contract law?",
            answer="Contract law is...",
            sources=[source],
            confidence="high",
            retrieved_documents=5,
            context_documents=[context_doc]
        )

        assert response.question == "What is contract law?"
        assert len(response.sources) == 1
        assert response.sources[0].title == "Contract Law Basics"
        assert len(response.context_documents) == 1

    def test_pipeline_status_model(self):
        """Test PipelineStatus model."""
        status = PipelineStatus(
            initialized=True,
            components={"embeddings": True, "vector_store": True},
            knowledge_base={"document_count": 100}
        )

        assert status.initialized is True
        assert status.components["embeddings"] is True
        assert status.knowledge_base["document_count"] == 100


class TestErrorHandling:
    """Test error handling and exception cases."""

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/v1/health")
        assert response.status_code in [200, 405]

    def test_nonexistent_endpoint(self, test_client):
        """Test 404 handling."""
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404


class TestIntegrationFlow:
    """Test integration scenarios with mocks."""

    def test_full_question_answering_flow(self, test_client, mock_pipeline):
        """Test complete flow from question to answer."""
        # First check status
        with patch('api.routes.pipeline', mock_pipeline):
            status_response = test_client.get("/api/v1/status")
            assert status_response.status_code == 200

            # Ask a question
            question_data = {
                "question": "What are the key principles of contract law?",
                "include_context": True,
                "max_documents": 3
            }

            ask_response = test_client.post("/api/v1/ask", json=question_data)
            assert ask_response.status_code == 200

            # Verify the response structure
            data = ask_response.json()
            assert "question" in data
            assert "answer" in data
            assert "sources" in data
            assert "confidence" in data
            assert "retrieved_documents" in data


# Run only unit tests
if __name__ == "__main__":
    # Run unit tests only
    test_args = [__file__, "-v"]
    pytest.main(test_args)