"""
Pharmaceutical Document Extraction Package

This package provides tools for extracting and parsing pharmaceutical documents
(primarily PDF format) into structured data optimized for LLM processing.
"""

from pharma_extraction.parsers.llm_optimized_parser import LLMOptimizedPharmaParser
from pharma_extraction.parsers.phrase_based_parser import PhraseBasedPharmaParser
from pharma_extraction.parsers.section_aware_parser import SectionAwarePharmaParser

__version__ = "1.0.0"
__all__ = [
    "LLMOptimizedPharmaParser",
    "PhraseBasedPharmaParser",
    "SectionAwarePharmaParser",
]
