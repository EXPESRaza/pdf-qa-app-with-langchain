"""
Main Streamlit application for PDF Q&A system.
"""
import os
import tempfile
import streamlit as st
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from app.core.document import PDFProcessor
from app.core.vectorstore import VectorStoreManager
from app.tools.exact_match import ExactMatchTool
from app.tools.semantic_qa import SemanticQATool
from app.core.agent import QueryRouter
from app.components.chat import (
    init_session_state,
    display_chat_history,
    add_message,
    display_exact_match_result,
    display_qa_result
)

# Load custom CSS
def load_custom_css():
    with open("app/static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Configure page
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Initialize session state
init_session_state()

def get_llm(model_name: str):
    """Get LLM instance based on selection."""
    if model_name.startswith("gpt"):
        return ChatOpenAI(
            model_name=model_name,
            temperature=0,
            streaming=True
        )
    else:  # Ollama
        return OllamaLLM(model=model_name.lower())

def init_tools(pdf_path: str, model_name: str):
    """Initialize Q&A tools."""
    # Initialize components
    pdf_processor = PDFProcessor()
    vector_store = VectorStoreManager()
    
    # Process PDF and create vector store
    with st.spinner("Processing PDF and creating vector store..."):
        documents = pdf_processor.process_pdf(pdf_path)
        vector_store.create_vector_store(documents)
    
    # Initialize tools
    exact_match_tool = ExactMatchTool(pdf_processor)
    semantic_qa_tool = SemanticQATool(
        vector_store=vector_store,
        llm=get_llm(model_name)
    )
    
    # Create router
    return QueryRouter(
        exact_match_tool=exact_match_tool,
        semantic_qa_tool=semantic_qa_tool,
        llm=get_llm(model_name)
    )

def handle_query(
    router: QueryRouter,
    query: str,
    pdf_path: str,
    streaming: bool
):
    """Handle user query."""
    # Add user message
    add_message("user", query)
    
    # Create placeholder for streaming
    placeholder = st.empty() if streaming else None
    
    def streaming_callback(token: str):
        """Handle streaming tokens."""
        if streaming:
            st.session_state.current_response += token
            placeholder.markdown(st.session_state.current_response)
    
    # Process query
    with st.spinner("Processing your question..."):
        result = router.process_query(
            question=query,
            file_path=pdf_path,
            streaming_callback=streaming_callback if streaming else None
        )
    
    # Display results
    if result.tool_used == "EXACT_MATCH":
        display_exact_match_result(result.result)
        content = f"Found {result.result.count} matches"
        sources = [
            {"page_number": m["page_number"], "snippet": s}
            for m in result.result.matches
            for s in m["snippets"]
        ]
    else:  # SEMANTIC_QA
        display_qa_result(
            result.result,
            streaming=streaming,
            placeholder=placeholder
        )
        content = result.result.answer
        sources = [
            {
                "page_number": doc.metadata["page_number"],
                "content": doc.page_content
            }
            for doc in result.result.source_documents
        ]
    
    # Add assistant message
    add_message("assistant", content, sources)
    
    # Force a rerun to update the chat display
    st.rerun()

def main():
    """Main application."""
    # Add CSS styles at the beginning
    st.markdown(
        """
        <style>
        /* Reduce Streamlit's default padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        
        /* Hide Streamlit's default header decoration */
        header {
            visibility: hidden;
        }
        
        /* Adjust main content padding */
        .main > div:first-child {
            padding-top: 0rem !important;
        }
        
        /* Adjust Streamlit element container height */
        .element-container {
            min-height: 0 !important;
            height: auto !important;
        }
        
        /* Messages container */
        .messages-container {
            height: 350px;
            overflow-y: auto;
            padding: 1rem;
            margin-bottom: 60px;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            background: white;
        }
        
        /* Empty chat state */
        .empty-chat {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }
        
        /* Chat input container */
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 0.5rem 1rem;
            border-top: 1px solid #e6e6e6;
            z-index: 100;
        }
        
        /* Chat input styling */
        .stChatInput {
            width: 100% !important;
            margin: 0 !important;
            padding: 0.5rem !important;
        }
        
        /* Chat message styling */
        .stChatMessage {
            margin-bottom: 0.75rem;
            padding: 0.5rem !important;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Source document styling */
        .source-document {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #3c7dff;
        }
        
        /* Welcome page styles */
        .welcome-container {
            max-width: 800px;
            margin: 0.5rem auto;
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .welcome-icon {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            animation: float 3s ease-in-out infinite;
        }
        
        .welcome-description {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        
        .features-grid {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 3rem auto 0;
            max-width: 800px;
            padding: 0 1rem;
        }
        
        .feature-card {
            flex: 1;
            max-width: 300px;
            background: linear-gradient(145deg, #ffffff, #f5f7fa);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.8);
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #6b9fff, #3c7dff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }
        
        .feature-card h3 {
            color: #2c3e50;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .feature-card p {
            color: #666;
            font-size: 1.1rem;
            line-height: 1.5;
            margin: 0;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Chat interface container */
        .chat-interface {
            display: flex;
            flex-direction: column;
            height: 450px;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            background: white;
            position: relative;
            overflow: hidden;
        }
        
        /* Chat input wrapper */
        .chat-input-wrapper {
            position: relative;
            background: white;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #e6e6e6;
            z-index: 100;
        }
        
        /* Messages wrapper */
        .messages-wrapper {
            min-height: 0;
            height: auto;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column-reverse;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            background: white;
        }
        
        /* Chat input styling */
        .stChatInput {
            width: 100% !important;
            margin: 0 !important;
            padding: 0.5rem !important;
        }
        
        /* Empty chat state */
        .empty-chat {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }
        
        /* Chat message styling */
        .stChatMessage {
            margin-bottom: 0.75rem;
            padding: 0.5rem !important;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <h2>⚙️ Settings</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Model selection
        model_options = [
            "gpt-3.5-turbo",
            "gpt-4",
            "llama3.2",
            "mistral"
        ]
        model_name = st.selectbox(
            "🤖 Select Language Model",
            model_options,
            index=0
        )
        
        # Streaming toggle
        streaming = st.toggle("🔄 Enable Streaming", value=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # File uploader
        st.markdown(
            """
            <div class="upload-section">
                <h3>📄 Upload Document</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to analyze"
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_response = ""
            st.success("Chat history cleared!")
    
    # Main content
    if uploaded_file:
        # Header for active session
        st.markdown(
            f"""
            <div class="session-header">
                <h1>📚 PDF Q&A System</h1>
                <div class="session-info">
                    <div class="info-badge">
                        <span class="label">Model:</span>
                        <span class="value">{model_name}</span>
                    </div>
                    <div class="info-badge">
                        <span class="label">File:</span>
                        <span class="value">{uploaded_file.name}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Initialize tools
            if "router" not in st.session_state:
                st.session_state.router = init_tools(pdf_path, model_name)
            
            # Chat interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Chat container
                st.markdown("### 💬 Chat Interface")
                
                # Chat input container
                st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
                query = st.chat_input(
                    "Ask a question about the PDF...",
                    key="chat_input"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Messages container
                st.markdown('<div class="messages-wrapper">', unsafe_allow_html=True)
                display_chat_history()
                st.markdown('</div>', unsafe_allow_html=True)
                
                if query:
                    handle_query(st.session_state.router, query, pdf_path, streaming)
            
            with col2:
                # Information panel
                st.markdown("### ℹ️ Quick Tips")
                st.markdown("""
                - 🔍 Use quotes for exact matching
                - 💡 Ask specific questions
                - 📄 Reference page numbers
                - 🔄 Toggle streaming for faster responses
                """)
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            
        finally:
            # Cleanup
            try:
                os.unlink(pdf_path)
            except:
                pass
    else:
        # Welcome page header
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-icon">📚</div>
                <h1>Welcome to PDF Q&A System</h1>
                <p class="welcome-description">
                    Upload a PDF and start an intelligent conversation about its content.
                    Get instant answers, find exact matches, and explore documents effortlessly.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Features grid
        st.markdown(
            """
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h3>Smart Search</h3>
                    <p>Instantly find exact phrases or get semantic understanding of your document's content with advanced AI capabilities.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h3>AI-Powered</h3>
                    <p>Leverage state-of-the-art language models for accurate, context-aware responses to your questions.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main() 