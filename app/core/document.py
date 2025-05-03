"""
Document processing module for PDF handling and text chunking.
"""
from typing import List, Dict, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF document loading, parsing, and chunking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", " ", ""]
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            separators: List of separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def load_pdf(self, file_path: str | Path) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed document chunks with metadata
        """
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Add page numbers to metadata
            for i, page in enumerate(pages):
                page.metadata["page_number"] = i + 1
            
            return pages
        except Exception as e:
            logger.error(f"Error loading PDF file: {e}")
            raise
    
    def process_pdf(
        self,
        file_path: str | Path,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load and chunk a PDF file with metadata.
        
        Args:
            file_path: Path to the PDF file
            metadata: Additional metadata to add to each chunk
            
        Returns:
            List of processed document chunks
        """
        pages = self.load_pdf(file_path)
        
        # Split pages into chunks
        chunks = []
        for page in pages:
            page_chunks = self.text_splitter.split_documents([page])
            
            # Preserve page numbers and add custom metadata
            for chunk in page_chunks:
                if metadata:
                    chunk.metadata.update(metadata)
                chunks.append(chunk)
        
        return chunks
    
    def get_page_content(self, file_path: str | Path, page_number: int) -> str:
        """
        Get the raw content of a specific page.
        
        Args:
            file_path: Path to the PDF file
            page_number: Page number (1-based)
            
        Returns:
            Raw text content of the page
        """
        pages = self.load_pdf(file_path)
        if 1 <= page_number <= len(pages):
            return pages[page_number - 1].page_content
        raise ValueError(f"Page number {page_number} out of range") 