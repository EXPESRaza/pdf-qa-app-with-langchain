"""
Chat component for displaying and managing chat messages.
"""
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime

from app.tools.exact_match import MatchResult
from app.tools.semantic_qa import QAResult

def display_chat_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None
):
    """
    Display a chat message with optional sources.
    
    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
        sources: Optional source documents
    """
    # Create a container for the message
    message_container = st.container()
    
    with message_container:
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(content)
            
            if sources:
                with st.expander("📄 View Sources", expanded=False):
                    for source in sources:
                        st.markdown(
                            f"""
                            <div class="source-document">
                                <strong>Page {source.get('page_number', 'N/A')}</strong>
                                <p>{source.get('content', source.get('snippet', 'No content available'))}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

def display_exact_match_result(result: MatchResult):
    """
    Display exact match results.
    
    Args:
        result: MatchResult instance
    """
    st.markdown(f"🔍 Found {result.count} matches")
    
    if result.matches:
        for match in result.matches:
            with st.expander(
                f"📄 Page {match['page_number']} ({match['count']} matches)",
                expanded=False
            ):
                for snippet in match['snippets']:
                    st.markdown(
                        f"""
                        <div class="source-document">
                            <p>{snippet}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

def display_qa_result(
    result: QAResult,
    streaming: bool = False,
    placeholder: Optional[Any] = None
):
    """
    Display Q&A results.
    
    Args:
        result: QAResult instance
        streaming: Whether to use streaming output
        placeholder: Optional streamlit placeholder for streaming
    """
    if streaming and placeholder:
        placeholder.markdown(result.answer)
    else:
        st.markdown(result.answer)
        
    if result.source_documents:
        with st.expander("📄 View Sources", expanded=False):
            for doc in result.source_documents:
                st.markdown(
                    f"""
                    <div class="source-document">
                        <strong>Page {doc.metadata.get('page_number', 'N/A')}</strong>
                        <p>{doc.page_content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

def display_chat_history():
    """Display the chat history."""
    messages = st.session_state.get("messages", [])
    
    if not messages:
        st.markdown(
            """
            <div class="empty-chat">
                <p>No messages yet. Start a conversation by asking a question about the PDF!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Display messages in reverse order but keep prompt-response pairs together
    for i in range(len(messages) - 1, -1, -2):
        if i > 0:  # If we have both prompt and response
            # Display prompt
            with st.container():
                display_chat_message(
                    role=messages[i - 1]["role"],
                    content=messages[i - 1]["content"],
                    sources=messages[i - 1].get("sources")
                )
            # Display response
            with st.container():
                display_chat_message(
                    role=messages[i]["role"],
                    content=messages[i]["content"],
                    sources=messages[i].get("sources")
                )
        else:  # If we only have a prompt (shouldn't happen in normal flow)
            with st.container():
                display_chat_message(
                    role=messages[i]["role"],
                    content=messages[i]["content"],
                    sources=messages[i].get("sources")
                )

def add_message(role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
    """Add a message to the chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": datetime.now().isoformat()
    }) 