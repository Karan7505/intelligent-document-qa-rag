# ============================================
# RESPONSIBLE AI MODULE
# Adds safety and transparency features
# ============================================

from typing import List, Dict

class ResponsibleAI:
    """
    Adds responsible AI features:
    1. Confidence scoring
    2. Source citation
    3. Hallucination detection
    """
    
    # ============================================
    # CONFIDENCE SCORING
    # ============================================
    
    @staticmethod
    def calculate_confidence(source_documents: List) -> float:
        """
        Calculate confidence score based on number of relevant sources
        
        Logic:
        - More sources = more confident
        - If 5 sources found = 100% confident
        - If 1 source found = 60% confident
        - If 0 sources = 20% confident (low confidence)
        
        Input: List of source documents
        Output: Confidence score (0.0 to 1.0)
        """
        if not source_documents:
            return 0.2  # Very low confidence if no sources
        
        # Calculate based on number of sources
        num_sources = len(source_documents)
        
        if num_sources >= 5:
            confidence = 0.95  # Very confident
        elif num_sources >= 3:
            confidence = 0.80  # Confident
        elif num_sources >= 1:
            confidence = 0.60  # Somewhat confident
        else:
            confidence = 0.20  # Not confident
        
        return confidence
    
    # ============================================
    # SOURCE CITATION
    # ============================================
    
    @staticmethod
    def format_sources(source_documents: List) -> str:
        """
        Format source documents nicely for display
        
        Input: List of document chunks
        Output: Formatted string showing sources
        
        Example output:
        📚 Sources:
        1. Page 1 - Climate change causes
        2. Page 2 - Global warming effects
        """
        if not source_documents:
            return "❌ No sources found for this answer"
        
        formatted = "📚 Sources:\n"
        for i, doc in enumerate(source_documents, 1):
            page_num = doc.metadata.get('page', 'Unknown')
            formatted += f"  {i}. Page {page_num}\n"
        
        return formatted
    
    # ============================================
    # HALLUCINATION DETECTION
    # ============================================
    
    @staticmethod
    def detect_potential_hallucination(answer: str, source_text: str) -> bool:
        """
        Detect if answer might contain made-up information
        
        Simple check: Does the answer use specific numbers/facts
        that might not be in sources?
        
        Input: answer (AI's response), source_text (from documents)
        Output: True if potential hallucination detected
        
        Note: This is a simple check. Real hallucination detection
        requires more advanced techniques.
        """
        # Check for extremely specific numbers that might be hallucinated
        suspicious_patterns = ["according to", "studies show", "research indicates"]
        
        # If answer contains these but sources don't mention them
        for pattern in suspicious_patterns:
            if pattern in answer.lower() and pattern not in source_text.lower():
                return True
        
        return False
    
    # ============================================
    # QUERY VALIDATION
    # ============================================
    
    @staticmethod
    def validate_query(query: str) -> Dict[str, any]:
        """
        Check if query is safe and appropriate
        
        Returns:
        {
            "is_valid": True/False,
            "reason": "explanation if invalid"
        }
        """
        # List of sensitive keywords we should warn about
        sensitive_keywords = [
            "illegal",
            "harmful",
            "dangerous",
            "hate",
            "violence"
        ]
        
        # Check if any sensitive keywords are in the query
        query_lower = query.lower()
        for keyword in sensitive_keywords:
            if keyword in query_lower:
                return {
                    "is_valid": False,
                    "reason": f"Query contains sensitive keyword: {keyword}"
                }
        
        # Check if query is too short (might be testing)
        if len(query.strip()) < 5:
            return {
                "is_valid": False,
                "reason": "Query too short. Please ask a complete question."
            }
        
        # Query is valid
        return {
            "is_valid": True,
            "reason": "Query is valid"
        }
    
    # ============================================
    # COMPREHENSIVE SAFETY CHECK
    # ============================================
    
    @staticmethod
    def comprehensive_safety_check(
        query: str,
        answer: str,
        source_documents: List,
        confidence: float
    ) -> Dict:
        """
        Run all safety checks and return a report
        
        Returns a dict with all safety information
        """
        validation = ResponsibleAI.validate_query(query)
        hallucination = ResponsibleAI.detect_potential_hallucination(
            answer,
            " ".join([doc.page_content for doc in source_documents])
        )
        
        return {
            "query_valid": validation["is_valid"],
            "validation_reason": validation["reason"],
            "potential_hallucination": hallucination,
            "confidence": confidence,
            "sources_count": len(source_documents),
            "recommendation": ResponsibleAI._get_recommendation(
                confidence, 
                len(source_documents),
                hallucination
            )
        }
    
    # ============================================
    # HELPER FUNCTION
    # ============================================
    
    @staticmethod
    def _get_recommendation(confidence: float, sources: int, hallucination: bool) -> str:
        """
        Provide recommendation based on safety checks
        """
        if not confidence > 0.5:
            return "⚠️ Low confidence - verify this answer independently"
        elif hallucination:
            return "⚠️ Potential hallucination detected - read sources carefully"
        elif sources < 2:
            return "ℹ️ Only one source - consider cross-referencing"
        else:
            return "✅ Answer appears reliable"