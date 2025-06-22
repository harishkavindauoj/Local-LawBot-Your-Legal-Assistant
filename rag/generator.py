import google.generativeai as genai
from typing import List, Dict, Any, Optional
from utils.logger import logger
from config import config


class GeminiGenerator:
    """Handles text generation using Google's Gemini API."""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or config.gemini_api_key
        self.model_name = model_name or config.gemini_model
        self.model = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini API client."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise

    def generate_legal_response(
            self,
            question: str,
            context_documents: List[Dict[str, Any]],
            max_tokens: Optional[int] = None
    ) -> str:
        """Generate a legal response based on question and retrieved context."""

        try:
            # Build context from retrieved documents
            context = self._build_context(context_documents)

            # Create the prompt
            prompt = self._create_legal_prompt(question, context)

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=max_tokens or 1024,
                top_p=0.8,
                top_k=40
            )

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract text from response
            if response.candidates:
                generated_text = response.candidates[0].content.parts[0].text
                logger.info("Successfully generated legal response")
                return generated_text.strip()
            else:
                logger.warning("No response generated from Gemini")
                return "I apologize, but I couldn't generate a response to your legal question at this time."

        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return f"I encountered an error while processing your legal question: {str(e)}"

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant legal documents found."

        context_parts = []
        for i, doc in enumerate(documents[:config.top_k_retrieval]):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            title = metadata.get("title", f"Document {i + 1}")

            context_part = f"Document {i + 1} - {title} (Source: {source}):\n{doc['content']}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _create_legal_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for legal question answering."""

        prompt = f"""You are a knowledgeable legal assistant designed to help users understand legal concepts and principles. Your role is to provide accurate, helpful, and accessible legal information based on the provided context.

IMPORTANT DISCLAIMERS:
- You provide general legal information, not legal advice
- Users should consult with qualified attorneys for specific legal situations
- Laws vary by jurisdiction and can change over time
- Your responses are based on the provided context and general legal principles

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question based primarily on the provided context documents
2. If the context doesn't fully address the question, clearly state this limitation
3. Provide clear, practical information that a non-lawyer can understand
4. Include relevant disclaimers about seeking professional legal advice when appropriate
5. Structure your response with clear sections if the topic is complex
6. Cite specific documents or sources when referencing particular legal principles

RESPONSE:"""

        return prompt

    def generate_simple_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a simple response for non-legal queries."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=max_tokens,
                top_p=0.9
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            if response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "I couldn't generate a response at this time."

        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return f"Error generating response: {str(e)}"






