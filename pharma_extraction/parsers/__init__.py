"""
Pharmaceutical Document Parsers

This module contains different parsing strategies for pharmaceutical documents:

- LLMOptimizedPharmaParser: Multi-representation structure optimized for LLM queries
- PhraseBasedPharmaParser: Phrase-level extraction with hierarchical context
- SectionAwarePharmaParser: Section-aware analysis with entity extraction
"""

from pharma_extraction.parsers.llm_optimized_parser import LLMOptimizedPharmaParser
from pharma_extraction.parsers.phrase_based_parser import PhraseBasedPharmaParser
from pharma_extraction.parsers.section_aware_parser import SectionAwarePharmaParser

__all__ = [
    "LLMOptimizedPharmaParser",
    "PhraseBasedPharmaParser",
    "SectionAwarePharmaParser",
]
