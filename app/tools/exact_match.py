"""
Tool for exact text matching and counting in PDF documents.
"""
from typing import List, Dict, Any
import re
import logging
from dataclasses import dataclass

from app.core.document import PDFProcessor

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Container for text match results."""
    count: int
    matches: List[Dict[str, Any]]
    
class ExactMatchTool:
    """Tool for finding exact text matches in documents."""
    
    def __init__(self, pdf_processor: PDFProcessor):
        """
        Initialize the exact match tool.
        
        Args:
            pdf_processor: PDFProcessor instance for document handling
        """
        self.pdf_processor = pdf_processor
    
    def count_matches(
        self,
        file_path: str,
        query: str,
        case_sensitive: bool = False,
        whole_word: bool = False
    ) -> MatchResult:
        """
        Count exact matches of a query in a PDF document.
        
        Args:
            file_path: Path to the PDF file
            query: Text to search for
            case_sensitive: Whether to perform case-sensitive matching
            whole_word: Whether to match whole words only
            
        Returns:
            MatchResult containing count and match details
        """
        try:
            pages = self.pdf_processor.load_pdf(file_path)
            
            # Prepare regex pattern
            if whole_word:
                pattern = fr'\b{re.escape(query)}\b'
            else:
                pattern = re.escape(query)
                
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            total_count = 0
            matches = []
            
            for page in pages:
                text = page.page_content
                page_matches = list(regex.finditer(text))
                page_count = len(page_matches)
                
                if page_count > 0:
                    matches.append({
                        'page_number': page.metadata['page_number'],
                        'count': page_count,
                        'snippets': [
                            self._get_context_snippet(text, m.start(), m.end())
                            for m in page_matches
                        ]
                    })
                    total_count += page_count
            
            return MatchResult(
                count=total_count,
                matches=sorted(matches, key=lambda x: x['page_number'])
            )
            
        except Exception as e:
            logger.error(f"Error counting matches: {e}")
            raise
    
    def _get_context_snippet(
        self,
        text: str,
        start: int,
        end: int,
        context_chars: int = 50
    ) -> str:
        """
        Get a text snippet around a match with context.
        
        Args:
            text: Full text to extract from
            start: Start index of match
            end: End index of match
            context_chars: Number of context characters on each side
            
        Returns:
            Text snippet with context
        """
        snippet_start = max(0, start - context_chars)
        snippet_end = min(len(text), end + context_chars)
        
        prefix = '...' if snippet_start > 0 else ''
        suffix = '...' if snippet_end < len(text) else ''
        
        return f"{prefix}{text[snippet_start:snippet_end].strip()}{suffix}" 