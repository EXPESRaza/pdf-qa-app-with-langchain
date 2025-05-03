"""
Vector store module for document embeddings and similarity search using FAISS.
"""
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages document embeddings and similarity search using FAISS."""
    
    def __init__(
        self,
        embedding_model: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: LangChain embeddings model (defaults to OpenAI)
            persist_directory: Directory to persist the vector store
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vector_store = None
    
    def create_vector_store(
        self,
        documents: List[Document],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to embed
            metadatas: Optional metadata for each document
            
        Returns:
            FAISS vector store instance
        """
        try:
            # Extract text and metadata from documents
            texts = [doc.page_content for doc in documents]
            if not metadatas:
                metadatas = [doc.metadata for doc in documents]
            
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
            
            if self.persist_directory:
                self.save_vector_store()
                
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, directory: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Optional directory to save to (overrides persist_directory)
        """
        if not self.vector_store:
            raise ValueError("No vector store to save")
            
        save_path = directory or self.persist_directory
        if not save_path:
            raise ValueError("No persist directory specified")
            
        try:
            self.vector_store.save_local(save_path)
            logger.info(f"Vector store saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, directory: Optional[str] = None) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            directory: Optional directory to load from (overrides persist_directory)
            
        Returns:
            Loaded FAISS vector store
        """
        load_path = directory or self.persist_directory
        if not load_path:
            raise ValueError("No persist directory specified")
            
        try:
            self.vector_store = FAISS.load_local(
                load_path,
                self.embedding_model
            )
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        if not self.vector_store:
            raise ValueError("No vector store available")
            
        try:
            return self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add new documents to an existing vector store.
        
        Args:
            documents: List of documents to add
            metadatas: Optional metadata for each document
        """
        if not self.vector_store:
            self.create_vector_store(documents)
            return
            
        try:
            # Extract text and metadata from documents
            texts = [doc.page_content for doc in documents]
            if not metadatas:
                metadatas = [doc.metadata for doc in documents]
                
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
            if self.persist_directory:
                self.save_vector_store()
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise 