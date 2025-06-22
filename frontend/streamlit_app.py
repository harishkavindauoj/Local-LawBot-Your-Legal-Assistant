import streamlit as st
import requests
import json
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Local LawBot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000/api/v1"


class LawBotUI:
    """Streamlit UI for the Local LawBot application."""

    def __init__(self):
        self.api_url = API_BASE_URL

    def check_api_health(self) -> bool:
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the pipeline status from the API."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def setup_knowledge_base(self, force_reload: bool = False) -> Dict[str, Any]:
        """Trigger knowledge base setup."""
        try:
            response = requests.post(
                f"{self.api_url}/setup",
                params={"force_reload": force_reload},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Setup failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def ask_question(self, question: str, include_context: bool = True, max_docs: int = 5) -> Dict[str, Any]:
        """Ask a legal question."""
        try:
            payload = {
                "question": question,
                "include_context": include_context,
                "max_documents": max_docs
            }

            response = requests.post(
                f"{self.api_url}/ask",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Question processing failed with status code {response.status_code}",
                    "detail": response.text
                }
        except Exception as e:
            return {"error": str(e)}

    def render_sidebar(self):
        """Render the sidebar with system controls."""
        st.sidebar.title("⚖️ Local LawBot")
        st.sidebar.markdown("---")

        # API Health Check
        st.sidebar.subheader("🔧 System Status")

        if st.sidebar.button("Check API Health"):
            with st.sidebar:
                with st.spinner("Checking API..."):
                    is_healthy = self.check_api_health()

                if is_healthy:
                    st.success("✅ API is running")
                else:
                    st.error("❌ API is not responding")

        # Pipeline Status
        if st.sidebar.button("Get Pipeline Status"):
            with st.sidebar:
                with st.spinner("Getting status..."):
                    status = self.get_pipeline_status()

                if "error" in status:
                    st.error(f"Error: {status['error']}")
                else:
                    st.json(status)

        st.sidebar.markdown("---")

        # Knowledge Base Setup
        st.sidebar.subheader("📚 Knowledge Base")

        force_reload = st.sidebar.checkbox("Force Reload", help="Force reload of all documents")

        if st.sidebar.button("Setup Knowledge Base"):
            with st.sidebar:
                with st.spinner("Setting up knowledge base..."):
                    result = self.setup_knowledge_base(force_reload)

                if "error" in result:
                    st.error(f"Setup failed: {result['error']}")
                else:
                    st.success("✅ Knowledge base setup started")
                    st.info("This may take a few minutes. Check the logs for progress.")

        st.sidebar.markdown("---")

        # Settings
        st.sidebar.subheader("⚙️ Settings")
        max_docs = st.sidebar.slider("Max Documents to Retrieve", 1, 10, 5)
        include_context = st.sidebar.checkbox("Include Context in Response", value=True)

        return max_docs, include_context

    def render_main_interface(self, max_docs: int, include_context: bool):
        """Render the main chat interface."""
        st.title("⚖️ Local LawBot: Your Legal Assistant")
        st.markdown("Ask me questions about tenant rights, consumer law, and other legal topics!")

        # Example questions
        with st.expander("💡 Example Questions"):
            examples = [
                "What is the proposed rule regarding seat belts in large school buses versus small school buses?",
                "How does the statute define the term ""appropriate congressional committees,"" and why is their role critical in the implementation and oversight of nuclear security reforms?",
                "Which Federal Motor Vehicle Safety Standards (FMVSS) are directly referenced or proposed to be "
                "amended in this NPRM?",
                "How does the NPRM address FMVSS No. 209’s distinction between Type 1 and Type 2 seat belt assemblies?",
                "What are the specific responsibilities of the Implementation Assessment Panel established under subsection (c) in relation to the nuclear security enterprise reform plan?"
            ]

            for example in examples:
                if st.button(example, key=f"example_{hash(example)}"):
                    st.session_state["question_input"] = example

        # Question input
        question = st.text_area(
            "Ask your legal question:",
            height=100,
            placeholder="Type your legal question here...",
            key="question_input"
        )

        # Ask button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            ask_button = st.button("🤔 Ask Question", type="primary", use_container_width=True)

        # Process question
        if ask_button and question.strip():
            self.process_question(question, max_docs, include_context)
        elif ask_button:
            st.warning("Please enter a question first!")

    def process_question(self, question: str, max_docs: int, include_context: bool):
        """Process and display the response to a question."""
        with st.spinner("🔍 Searching legal documents and generating response..."):
            response = self.ask_question(question, include_context, max_docs)

        if "error" in response:
            st.error(f"❌ Error: {response['error']}")
            if "detail" in response:
                st.error(f"Details: {response['detail']}")
            return

        # Display response
        st.markdown("---")
        st.subheader("📝 Response")

        # Answer
        st.markdown("### Answer")
        st.write(response.get("answer", "No answer provided"))

        # Confidence and metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = response.get("confidence", "unknown")
            confidence_color = {
                "high": "🟢",
                "medium": "🟡",
                "low": "🔴",
                "error": "❌"
            }.get(confidence, "⚪")
            st.metric("Confidence", f"{confidence_color} {confidence.title()}")

        with col2:
            st.metric("Documents Retrieved", response.get("retrieved_documents", 0))

        with col3:
            st.metric("Sources Found", len(response.get("sources", [])))

        # Sources
        sources = response.get("sources", [])
        if sources:
            st.markdown("### 📚 Sources")
            for i, source in enumerate(sources, 1):
                with st.expander(f"Source {i}: {source.get('title', 'Unknown Title')}"):
                    st.write(f"**Type:** {source.get('type', 'Unknown')}")
                    st.write(f"**Source:** {source.get('source', 'Unknown')}")
                    st.write(f"**Similarity Score:** {source.get('similarity_score', 0):.3f}")

        # Context documents (if included)
        context_docs = response.get("context_documents", [])
        if context_docs and include_context:
            st.markdown("### 🔍 Context Documents")
            for i, doc in enumerate(context_docs, 1):
                with st.expander(f"Context {i} (Score: {doc.get('similarity_score', 0):.3f})"):
                    st.write(f"**Source:** {doc.get('source', 'Unknown')}")
                    st.write(f"**Content Preview:** {doc.get('content', 'No content')}")

        # Disclaimer
        st.markdown("---")
        st.info("""
        ⚠️ **Legal Disclaimer**: This response provides general legal information, not legal advice. 
        Laws vary by jurisdiction and situation. For specific legal matters, please consult with a qualified attorney.
        """)

    def run(self):
        """Run the Streamlit application."""
        # Render sidebar
        max_docs, include_context = self.render_sidebar()

        # Render main interface
        self.render_main_interface(max_docs, include_context)

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Local LawBot v1.0.0 | Powered by Gemini Pro & ChromaDB"
            "</div>",
            unsafe_allow_html=True
        )


# Run the application
if __name__ == "__main__":
    app = LawBotUI()
    app.run()