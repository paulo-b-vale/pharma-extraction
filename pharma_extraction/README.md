# Pharma Extraction Package

Production-ready Python modules for extracting and parsing pharmaceutical documents (primarily PDF format) into structured data optimized for LLM processing.

## Overview

This package provides three different parsing strategies, each optimized for different use cases:

### 1. LLMOptimizedPharmaParser
Creates multiple representations of pharmaceutical documents optimized for different LLM tasks.

**Best for:**
- General LLM queries
- Multiple use cases requiring different representations
- Search and retrieval tasks

**Features:**
- Flat blocks for general queries
- Hierarchical structure for document organization
- Markdown for human reading
- Semantic graph for relationship queries
- Search index for efficient lookup
- Batch processing support

### 2. PhraseBasedPharmaParser
Extracts documents at the phrase level with complete hierarchical context.

**Best for:**
- Granular information retrieval
- Precise entity extraction
- Fine-grained semantic analysis

**Features:**
- Phrase-level segmentation
- Complete hierarchical context for each phrase
- Semantic classification (dosage, indication, contraindication, etc.)
- Table extraction with context linking

### 3. SectionAwarePharmaParser
Section-aware analysis with entity extraction using local LLMs.

**Best for:**
- Entity extraction with section context
- Understanding document structure
- Interactive querying

**Features:**
- Intelligent section detection
- Pharmaceutical abbreviation awareness
- Entity extraction (medication names, dosages, indications, etc.)
- Section-organized results
- Ollama LLM integration

## Installation

### Dependencies

```bash
pip install pymupdf4llm pdfplumber
```

For SectionAwarePharmaParser, you also need:
- [Ollama](https://ollama.ai/) installed and running locally
- A compatible LLM model (e.g., llama3:8b)

```bash
# Install Ollama, then pull a model
ollama pull llama3:8b
```

## Usage

### LLMOptimizedPharmaParser

```python
from pharma_extraction import LLMOptimizedPharmaParser

# Initialize parser
parser = LLMOptimizedPharmaParser()

# Process a single PDF
if parser.load_document("document.pdf"):
    # Save the optimized structure
    parser.save_optimized_json("output.json")

    # Query the document
    context = parser.get_context_for_llm("What is the dosage?", max_blocks=5)
    print(context)

# Process multiple PDFs
results = parser.process_multiple_pdfs(
    folder_path="pdfs/",
    output_dir="output/"
)
```

### PhraseBasedPharmaParser

```python
from pharma_extraction import PhraseBasedPharmaParser

# Initialize parser
parser = PhraseBasedPharmaParser()

# Process a PDF
if parser.load_document("document.pdf"):
    # Save the phrase-based structure
    parser.save_optimized_json("output.json")

    # Access phrase blocks
    phrases = parser.optimized_structure["document_structure"]["phrase_blocks"]

    for phrase in phrases[:5]:
        print(f"Content: {phrase['content']}")
        print(f"Type: {phrase['metadata']['phrase_type']}")
        print(f"Section: {phrase['context']['breadcrumb']}")
```

### SectionAwarePharmaParser

```python
from pharma_extraction import SectionAwarePharmaParser

# Initialize parser with specific model
parser = SectionAwarePharmaParser(model_name="llama3:8b")

# Process a PDF (requires Ollama running)
if parser.process_document("document.pdf"):
    # Save results
    parser.save_results("output.json")

    # Query the document
    answer = parser.query_document("What are the contraindications?")
    print(answer)
```

## Package Structure

```
pharma_extraction/
├── __init__.py                          # Package initialization
├── README.md                            # This file
└── parsers/
    ├── __init__.py                      # Parser module initialization
    ├── llm_optimized_parser.py         # Multi-representation parser
    ├── phrase_based_parser.py          # Phrase-level parser
    └── section_aware_parser.py         # Section-aware entity extractor
```

## Key Differences

| Feature | LLMOptimized | PhraseBased | SectionAware |
|---------|--------------|-------------|--------------|
| Output granularity | Block-level | Phrase-level | Sentence-level |
| Section awareness | Yes | Yes | Yes |
| Multiple representations | Yes | No | No |
| Entity extraction | No | No | Yes (with LLM) |
| Batch processing | Yes | Yes | No |
| LLM required | No | No | Yes (Ollama) |
| Best for | General queries | Precise retrieval | Entity extraction |

## Output Formats

### LLMOptimizedPharmaParser Output

```json
{
  "metadata": {
    "total_blocks": 150,
    "recommended_use": {
      "general_queries": "flat_blocks",
      "structure_questions": "hierarchical",
      "human_reading": "markdown"
    }
  },
  "representations": {
    "flat_blocks": [...],
    "hierarchical": {...},
    "markdown": "...",
    "semantic_graph": {...},
    "search_index": {...}
  }
}
```

### PhraseBasedPharmaParser Output

```json
{
  "metadata": {
    "extraction_type": "phrase_based",
    "total_phrases": 450
  },
  "document_structure": {
    "context_hierarchy": {...},
    "phrase_blocks": [
      {
        "phrase_id": "phrase_1",
        "content": "...",
        "context": {
          "hierarchy": {...},
          "breadcrumb": "Section 1: Title > Subsection"
        },
        "metadata": {
          "phrase_type": "dosage_instruction",
          "contains_dosage": true
        }
      }
    ]
  }
}
```

### SectionAwarePharmaParser Output

```json
{
  "metadata": {...},
  "sentence_analyses": [...],
  "section_entities": {
    "entities_by_section": {
      "DOSAGEM": {
        "medication_names": ["..."],
        "dosages": ["500mg", "3x ao dia"]
      }
    },
    "all_entities": [...]
  }
}
```

## Important Notes

1. **All Google Colab code has been removed** - these modules work with local file paths only
2. **No Google Drive dependencies** - use standard file I/O operations
3. **Type hints** - Added where obvious for better IDE support
4. **Organized imports** - Standard library, third-party, and local imports are separated
5. **Comprehensive docstrings** - All classes and main methods include detailed documentation

## Common Use Cases

### Batch Processing
```python
parser = LLMOptimizedPharmaParser()
results = parser.process_multiple_pdfs("input_folder/", "output_folder/")
print(f"Processed: {results['successful_count']} files")
```

### Finding Specific Information
```python
parser = PhraseBasedPharmaParser()
parser.load_document("document.pdf")

# Find all dosage-related phrases
phrases = parser.optimized_structure["document_structure"]["phrase_blocks"]
dosage_phrases = [
    p for p in phrases
    if p["metadata"]["phrase_type"] == "dosage_instruction"
]
```

### Entity Extraction
```python
parser = SectionAwarePharmaParser()
parser.process_document("document.pdf")

# Get all entities by section
entities = parser.structured_data["section_entities"]["entities_by_section"]
for section, section_entities in entities.items():
    print(f"{section}: {section_entities['medication_names']}")
```

## License

This package is provided as-is for pharmaceutical document processing tasks.

## Support

For issues or questions, refer to the example_usage.py file for working examples.
