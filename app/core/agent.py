"""
Agent module for intelligent query routing between tools.
"""
from typing import Dict, Any, Optional, Union
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

from app.tools.exact_match import ExactMatchTool, MatchResult
from app.tools.semantic_qa import SemanticQATool, QAResult

logger = logging.getLogger(__name__)

ROUTING_PROMPT = """You are a query routing agent that determines whether a user's question requires exact text matching or semantic question answering.

Question: {question}

Determine if this question requires:
1. EXACT_MATCH - for questions about:
   - Counting occurrences (e.g., "how many times", "count of", "number of occurrences")
   - Finding specific phrases (e.g., "find the exact phrase", "where does X appear")
   - Checking presence (e.g., "does X appear", "is X mentioned", "occurs", "appears")
   - Frequency analysis (e.g., "how frequently", "how often")

2. SEMANTIC_QA - for questions requiring:
   - Understanding and explanation
   - Summarization or analysis
   - Complex reasoning
   - Multiple document context
   - Answering "why" or "how" questions (unless specifically counting)

Examples:
- "How many times is 'neural network' mentioned?" -> EXACT_MATCH
- "Where does the phrase 'machine learning' appear?" -> EXACT_MATCH
- "Count the occurrences of 'AI' in the document" -> EXACT_MATCH
- "What are the main benefits of AI?" -> SEMANTIC_QA
- "Explain how neural networks work" -> SEMANTIC_QA

Output only one word: EXACT_MATCH or SEMANTIC_QA
"""

# Regular expressions for identifying exact match queries
EXACT_MATCH_PATTERNS = [
    r"how many times",
    r"how many occurrences",
    r"how many instances",
    r"how often",
    r"how frequently",
    r"count of",
    r"number of",
    r"occurrences of",
    r"instances of",
    r"appears",
    r"appear",
    r"occurring",
    r"occurs",
    r"mentioned",
    r"mentions",
    r"find.*exact",
    r"locate.*phrase",
    r"search for.*exact",
]

@dataclass
class QueryResult:
    """Container for query results."""
    tool_used: str
    result: Union[MatchResult, QAResult]

class QueryRouter:
    """Routes queries to appropriate tools based on intent."""
    
    def __init__(
        self,
        exact_match_tool: ExactMatchTool,
        semantic_qa_tool: SemanticQATool,
        llm: Optional[BaseLLM] = None
    ):
        """
        Initialize the query router.
        
        Args:
            exact_match_tool: Tool for exact matching
            semantic_qa_tool: Tool for semantic Q&A
            llm: Language model for routing (defaults to OpenAI)
        """
        self.exact_match_tool = exact_match_tool
        self.semantic_qa_tool = semantic_qa_tool
        self.llm = llm or ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.exact_match_pattern = re.compile(
            "|".join(EXACT_MATCH_PATTERNS),
            re.IGNORECASE
        )
        self.routing_prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
    
    def _determine_tool(self, question: str) -> str:
        """
        Determine which tool to use for a query.
        
        Args:
            question: User's question
            
        Returns:
            Tool identifier ('EXACT_MATCH' or 'SEMANTIC_QA')
        """
        try:
            # First check for exact match patterns
            if self.exact_match_pattern.search(question):
                return 'EXACT_MATCH'
            
            # If no pattern match, use LLM for more nuanced decision
            response = self.llm.invoke(
                self.routing_prompt.format_messages(question=question)
            ).content.strip()
            
            if response not in ['EXACT_MATCH', 'SEMANTIC_QA']:
                logger.warning(f"Invalid routing response: {response}")
                return 'SEMANTIC_QA'  # Default to semantic QA
                
            return response
            
        except Exception as e:
            logger.error(f"Error in tool determination: {e}")
            return 'SEMANTIC_QA'  # Default to semantic QA on error
    
    def _extract_search_term(self, question: str) -> str:
        """
        Extract search term from question for exact matching.
        
        Args:
            question: User's question
            
        Returns:
            Extracted search term
        """
        # Use LLM to extract the search term
        prompt = ChatPromptTemplate.from_template("""Extract the exact text or phrase to search for from this question.
        Look for text in quotes first, if not found, identify the key phrase to search for.
        Output only the text to search for, nothing else.
        
        Examples:
        - Question: "How many times does 'neural network' appear?"
          Output: neural network
        - Question: "Count occurrences of artificial intelligence in the text"
          Output: artificial intelligence
        - Question: "Find all mentions of 'deep learning' in the document"
          Output: deep learning
        
        Question: {question}
        """)
        
        try:
            # First try to find quoted text
            quoted = re.findall(r'["\']([^"\']*)["\']', question)
            if quoted:
                # Return the first quoted string found
                return quoted[0]
            
            # If no quotes, use LLM
            return self.llm.invoke(
                prompt.format_messages(question=question)
            ).content.strip().strip('"\'')
        except Exception as e:
            logger.error(f"Error extracting search term: {e}")
            return question
    
    def process_query(
        self,
        question: str,
        file_path: str,
        streaming_callback: Optional[callable] = None,
        force_tool: Optional[str] = None
    ) -> QueryResult:
        """
        Process a query using the appropriate tool.
        
        Args:
            question: User's question
            file_path: Path to the PDF file
            streaming_callback: Optional callback for streaming responses
            force_tool: Optional tool to use ('EXACT_MATCH' or 'SEMANTIC_QA')
            
        Returns:
            QueryResult containing tool used and result
        """
        try:
            # Determine which tool to use
            tool = force_tool or self._determine_tool(question)
            
            if tool == 'EXACT_MATCH':
                search_term = self._extract_search_term(question)
                result = self.exact_match_tool.count_matches(
                    file_path=file_path,
                    query=search_term
                )
            else:  # SEMANTIC_QA
                result = self.semantic_qa_tool.ask(
                    question=question,
                    streaming_callback=streaming_callback
                )
            
            return QueryResult(tool_used=tool, result=result)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def clear_chat_history(self) -> None:
        """Clear the semantic QA chat history."""
        self.semantic_qa_tool.clear_memory()
    
    def get_chat_history(self) -> list:
        """Get the semantic QA chat history."""
        return self.semantic_qa_tool.get_chat_history() 