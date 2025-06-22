# Local LawBot 🤖⚖️

A **Retrieval-Augmented Generation (RAG)** legal assistant that provides intelligent answers to legal questions using Google's Gemini Pro and ChromaDB. This system processes legal documents, creates embeddings, and generates contextually relevant responses to help users understand legal concepts and principles.

## 🌟 Features

- **Intelligent Legal Q&A**: Ask questions about federal regulations, statutes, and legal procedures
- **Multiple Legal Datasets**: Access to Federal Register, CFR, US Code, and more
- **Advanced Document Processing**: Intelligent chunking with configurable overlap
- **Vector Similarity Search**: ChromaDB-powered document retrieval
- **Context-Aware Responses**: Gemini 1.5 Pro generates answers based on retrieved legal documents
- **Source Attribution**: Provides citations and sources for all responses
- **Confidence Scoring**: Indicates reliability of answers based on document relevance
- **Intelligent Caching**: Speeds up processing with document caching
- **Web API**: RESTful API for programmatic access
- **Interactive UI**: User-friendly Streamlit interface
- **Extensible Architecture**: Modular design for easy enhancement

## 🏗️ Architecture

The system follows a modular RAG pipeline architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Legal Docs    │ -> │  Vector Store    │ -> │   Retriever     │
│                 │    │   (ChromaDB)     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Frontend     │ <- │    FastAPI       │ <- │   Generator     │
│  (Streamlit)    │    │      API         │    │  (Gemini Pro)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Data Processing Layer**: Loads and preprocesses legal documents
2. **Embedding Service**: Generates vector embeddings using sentence transformers
3. **Vector Store**: ChromaDB for efficient similarity search
4. **RAG Pipeline**: Orchestrates retrieval and generation
5. **API Layer**: FastAPI with Pydantic models
6. **Frontend**: Streamlit web interface

## 📋 Prerequisites

- Python 3.8+
- Google AI Studio API Key (for Gemini 1.5 Pro)
- HuggingFace Account (optional, for faster dataset downloads)
- 4GB+ RAM (for embedding models and document processing)
- 5GB+ disk space (for legal datasets and vector storage)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/harishkavindauoj/Local-LawBot-Your-Legal-Assistant.git
cd Local-LawBot-Your-Legal-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro

# HuggingFace Configuration (for datasets)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMADB_PERSIST_DIR=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Legal Dataset Configuration
LEGAL_DATASET=pile-of-law/pile-of-law
LEGAL_DATASET_CONFIG=federal_register

# Document Processing Configuration
CHUNK_SIZE=2048
CHUNK_OVERLAP=200
MAX_DOCUMENTS=10

# RAG Configuration
TOP_K_RETRIEVAL=5
MAX_CONTEXT_LENGTH=2048
TEMPERATURE=0.3

# Multiple dataset configs (comma-separated)
USE_MULTIPLE_CONFIGS=true
MULTIPLE_CONFIGS=federal_register,cfr,uscode

# Cache Settings
ENABLE_CACHING=true
CACHE_DIRECTORY=./data/processed

# Logging
LOG_LEVEL=INFO
```

### 4. Prepare Legal Documents

The system automatically downloads and processes legal documents from the Pile of Law dataset on HuggingFace:

- **Federal Register**: Government regulations and notices
- **Code of Federal Regulations (CFR)**: Administrative rules
- **US Code**: Federal statutes

No manual document preparation needed - the system will download and process these automatically.

### 5. Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Launch the Web Interface

In a new terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Access the interface at `http://localhost:8501`

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### Health Check
```http
GET /health
```
Returns API health status.

#### Pipeline Status
```http
GET /status
```
Returns current pipeline initialization status and knowledge base statistics.

#### Setup Knowledge Base
```http
POST /setup?force_reload=false
```
Initializes the knowledge base with legal documents. Use `force_reload=true` to rebuild from scratch.

#### Ask Legal Question
```http
POST /ask
Content-Type: application/json

{
  "question": "What are tenant rights regarding repairs?",
  "include_context": true,
  "max_documents": 5
}
```

**Response:**
```json
{
  "question": "What are tenant rights regarding repairs?",
  "answer": "Based on the legal documents...",
  "sources": [
    {
      "title": "Tenant Rights Guide",
      "source": "tenant_rights.txt",
      "type": "legal_document",
      "similarity_score": 0.892
    }
  ],
  "confidence": "high",
  "retrieved_documents": 3,
  "context_documents": [...]
}
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-pro` |
| `HUGGINGFACE_TOKEN` | HuggingFace API token for datasets | Optional |
| `VECTOR_DB_TYPE` | Vector database type | `chromadb` |
| `CHROMADB_PERSIST_DIR` | ChromaDB storage path | `./data/chromadb` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `API_RELOAD` | Enable auto-reload in development | `true` |
| `LEGAL_DATASET` | HuggingFace dataset name | `pile-of-law/pile-of-law` |
| `LEGAL_DATASET_CONFIG` | Default dataset configuration | `federal_register` |
| `CHUNK_SIZE` | Document chunk size for processing | `2048` |
| `CHUNK_OVERLAP` | Overlap between document chunks | `200` |
| `MAX_DOCUMENTS` | Maximum documents to process | `10` |
| `TOP_K_RETRIEVAL` | Documents to retrieve for RAG | `5` |
| `MAX_CONTEXT_LENGTH` | Maximum context length for generation | `2048` |
| `TEMPERATURE` | Generation temperature | `0.3` |
| `USE_MULTIPLE_CONFIGS` | Enable multiple dataset configs | `true` |
| `MULTIPLE_CONFIGS` | Comma-separated dataset configs | `federal_register,cfr,uscode` |
| `ENABLE_CACHING` | Enable document processing cache | `true` |
| `CACHE_DIRECTORY` | Cache storage directory | `./data/processed` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Advanced Configuration

Edit `config.py` to customize:
- Model parameters
- Retrieval settings  
- API configurations
- File paths

## 📖 Usage Examples

### Web Interface

1. **Start the System**: Launch both API and Streamlit interface
2. **Setup Knowledge Base**: Click "Setup Knowledge Base" in sidebar
3. **Ask Questions**: Type legal questions in the main interface
4. **Review Responses**: Get answers with sources and confidence scores

### API Usage

```python
import requests

# Ask a question about federal regulations
response = requests.post("http://localhost:8000/api/v1/ask", json={
    "question": "What are the requirements for public notice in federal rulemaking?",
    "include_context": True,
    "max_documents": 3
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {[source['title'] for source in result['sources']]}")
```

### Python Integration

```python
from rag.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()
pipeline.initialize()
pipeline.setup_knowledge_base()

# Ask a question
response = pipeline.ask_legal_question(
    "What are my rights as a tenant?",
    include_context=True
)

print(response['answer'])
```

## 🗂️ Project Structure

```
local-lawbot/
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables
├── README.md                 # This file
│
├── api/                      # API layer
│   ├── __init__.py
│   ├── models.py            # Pydantic models
│   └── routes.py            # API routes
│
├── rag/                     # RAG pipeline
│   ├── __init__.py
│   ├── pipeline.py          # Main RAG orchestrator
│   ├── retriever.py         # Document retrieval
│   └── generator.py         # Response generation
│
├── embeddings/              # Embedding services
│   ├── __init__.py
│   ├── embedding_service.py # Text embedding
│   └── vector_store.py      # ChromaDB interface
│
├── data/                    # Data management
│   ├── __init__.py
│   ├── data_loader.py       # Document loading (HuggingFace)
│   ├── chromadb/           # Vector database
│   └── processed/          # Cached processed documents
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── text_processing.py  # Text utilities
│
└── frontend/               # User interface
    └── streamlit_app.py    # Streamlit web app
```

## 🛠️ Development

### Adding New Document Types

1. **Update Data Loader**: Modify `data/data_loader.py` to handle new formats
2. **Document Processing**: Add preprocessing logic for specific document types
3. **Metadata Enhancement**: Include relevant metadata for better retrieval

### Customizing the Pipeline

1. **Retrieval Strategy**: Modify `rag/retriever.py` for different search algorithms
2. **Response Generation**: Update `rag/generator.py` for custom prompting
3. **Embedding Models**: Change embedding service in `embeddings/embedding_service.py`

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-asyncio

# Run API tests
pytest test_api.py -v
```

## 📊 Performance Optimization

### Memory Usage
- Use smaller embedding models for limited RAM
- Implement document chunking for large files
- Configure ChromaDB memory settings

### Response Speed
- Pre-generate embeddings for static documents
- Implement caching for frequent queries
- Use async processing for concurrent requests

### Accuracy Improvements
- Fine-tune similarity thresholds
- Implement query expansion
- Add domain-specific preprocessing

## 🔒 Security Considerations

- **API Key Management**: Store API keys securely, never commit to version control
- **Input Validation**: All user inputs are validated using Pydantic models
- **Rate Limiting**: Implement rate limiting for production deployment
- **Access Control**: Add authentication for sensitive legal information

## 🤝 Contributing

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for classes and methods
- Write tests for new functionality
- Update documentation as needed

## 📝 Legal Disclaimer

**Important**: This application provides general legal information, not legal advice. The responses are generated based on the provided legal documents and should not be considered as professional legal counsel. Always consult with qualified attorneys for specific legal matters.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI** for the Gemini Pro API
- **ChromaDB** for vector storage capabilities
- **Sentence Transformers** for embedding models
- **FastAPI** for the robust API framework
- **Streamlit** for the intuitive web interface

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/local-lawbot/issues)
- **Documentation**: Check the `/docs` endpoint when running the API
- **Community**: Join our discussions in GitHub Discussions

---

**Built with ❤️ for the legal community**
