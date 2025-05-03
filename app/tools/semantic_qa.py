"""
Semantic Q&A tool for answering questions about PDF content.
"""
from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

from app.core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Container for Q&A results."""
    answer: str
    source_documents: List[Dict[str, Any]]

class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    def __init__(self, callback):
        """Initialize with streaming callback."""
        super().__init__()
        self.callback = callback
        self._buffer = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process streaming token."""
        self._buffer.append(token)
        self.callback(token)
    
    def get_buffer(self) -> str:
        """Get the complete response."""
        return "".join(self._buffer)

class SemanticQATool:
    """Tool for semantic question answering over PDF content."""
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm: BaseLLM
    ):
        """
        Initialize the semantic QA tool.
        
        Args:
            vector_store: Vector store manager
            llm: Language model for Q&A
        """
        self.vector_store = vector_store
        self.llm = llm
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )
        
        # Create the prompt template
        template = """You are a helpful AI assistant that answers questions about PDF documents.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        Assistant: Let me help you with that."""
        
        self.prompt = ChatPromptTemplate.from_template(template)
    
    def ask(
        self,
        question: str,
        streaming_callback: Optional[callable] = None
    ) -> QAResult:
        """
        Ask a question about the PDF content.
        
        Args:
            question: User's question
            streaming_callback: Optional callback for streaming responses
            
        Returns:
            QAResult containing answer and source documents
        """
        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(question)
            
            # Format messages
            messages = self.prompt.format_messages(
                context=docs,
                chat_history=self.memory.load_memory_variables({})["chat_history"],
                question=question
            )
            
            # Generate answer
            if streaming_callback:
                # Create streaming handler
                streaming_handler = StreamingCallbackHandler(streaming_callback)
                
                # Generate response with streaming
                response = ""
                for chunk in self.llm.stream(messages):
                    if hasattr(chunk, 'content'):
                        response += chunk.content
                    else:
                        response += str(chunk)
                
                answer = response
            else:
                # For non-streaming, use direct invoke
                response = self.llm.invoke(messages)
                answer = response.content
            
            # Save to memory
            self.memory.save_context(
                {"input": question},
                {"output": answer}
            )
            
            return QAResult(
                answer=answer,
                source_documents=docs
            )
            
        except Exception as e:
            logger.error(f"Error in semantic QA: {e}")
            raise
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history."""
        return self.memory.chat_memory.messages 