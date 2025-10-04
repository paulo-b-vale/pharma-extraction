# Entity Extraction Modules - Usage Guide

## Overview

This package provides three complementary knowledge extraction modules for pharmaceutical documents. Each extractor produces **knowledge triples** - structured representations of facts in the form `[entity, relation, value]`.

## What are Knowledge Triples?

Knowledge triples are a way to represent facts as structured data. Each triple consists of:
- **Entity**: The subject of the fact (e.g., "Paracetamol", "medication")
- **Relation**: The relationship or property (e.g., "has_dosage", "is_indicated_for")
- **Value**: The object or value (e.g., "500mg", "headache")

Examples:
```python
["Paracetamol", "has_dosage", "500mg 3x ao dia"]
["Amoxicilina", "treats", "infecções respiratórias"]
["medication", "is_contraindicated_in", "pregnancy"]
```

## Available Extractors

### 1. EnhancedPharmaceuticalKnowledgeExtractor

**Best for**: Phrase-based JSON files with detailed structure

```python
from pharma_extraction.extractors import EnhancedPharmaceuticalKnowledgeExtractor
from pathlib import Path

# Initialize
extractor = EnhancedPharmaceuticalKnowledgeExtractor(
    model_name="llama3.2:3b",
    ollama_url="http://localhost:11434/api/generate",
    max_retries=3,
    request_delay=0.5
)

# Process a single file
input_file = Path("data/document_phrase_optimized.json")
result = extractor.process_phrase_based_json(input_file)

# Or process a directory
input_dir = Path("data/phrase_optimized")
output_dir = Path("output/graph_data")
extractor.process_directory(input_dir, output_dir)
```

**Features**:
- Processes phrase-by-phrase with full context
- Handles table data extraction
- Pharmaceutical keyword filtering
- Multiple parsing strategies
- Comprehensive logging

**Input Format**: JSON files with `phrase_blocks` and `table_blocks`

**Output**: Enhanced graph data with phrase and table extractions

### 2. PharmaceuticalKnowledgeExtractor

**Best for**: Block-based JSON files (flat structure)

```python
from pharma_extraction.extractors import PharmaceuticalKnowledgeExtractor
from pathlib import Path

# Initialize
extractor = PharmaceuticalKnowledgeExtractor(
    model_name="llama3.2:3b",
    max_retries=3,
    request_delay=0.5
)

# Process a single file
input_file = Path("data/document_llm_optimized.json")
result = extractor.process_json_file(input_file)

# Or process a directory
input_dir = Path("data/llm_optimized")
output_dir = Path("output/basic_graph")
extractor.process_directory(input_dir, output_dir)
```

**Features**:
- Block-by-block processing
- Section-aware prompting
- Content relevance filtering
- JSON parsing with regex fallback

**Input Format**: JSON files with `flat_blocks` in `representations`

**Output**: Graph data with block-level extractions

### 3. AutomatedPharmaParser

**Best for**: Direct PDF processing with full pipeline

```python
from pharma_extraction.extractors import AutomatedPharmaParser

# Initialize
parser = AutomatedPharmaParser(model_name="llama3.2:3b")

# Process a PDF
success = parser.process_document("data/bula.pdf")

if success:
    # Save results
    output_file = parser.save_results("output/analysis.json")

    # Query the document
    answer = parser.query_document("Qual é a dosagem recomendada?")
    print(answer)
```

**Features**:
- Complete end-to-end pipeline
- PDF text extraction (pdfplumber + pymupdf4llm)
- Document structure analysis
- Entity extraction with chunking
- Summary generation
- Interactive querying

**Input Format**: PDF files

**Output**: Complete analysis with entities, structure, and summary

## Choosing the Right Extractor

| Extractor | Use When | Input | Output |
|-----------|----------|-------|--------|
| **Enhanced** | You have phrase-based JSONs with detailed structure | `*_phrase_optimized.json` | Enhanced graph with phrase/table extractions |
| **Basic** | You have block-based JSONs from initial parsing | `*_llm_optimized.json` | Basic graph with block extractions |
| **Automated** | You have raw PDFs and want full pipeline | `*.pdf` | Complete analysis + query capability |

## Output Formats

### Enhanced Extractor Output
```json
{
  "document_metadata": {...},
  "extraction_summary": {
    "total_phrase_blocks": 150,
    "total_table_blocks": 5,
    "total_triples_extracted": 234
  },
  "phrase_extractions": [...],
  "table_extractions": [...],
  "all_extracted_triples": [
    ["entity", "relation", "value"],
    ...
  ]
}
```

### Basic Extractor Output
```json
{
  "document_metadata": {...},
  "extraction_summary": {
    "total_triples_extracted": 187
  },
  "graph_extractions": [
    {
      "block_id": "block_0",
      "breadcrumb": "Section > Subsection",
      "triples": [...]
    }
  ]
}
```

### Automated Parser Output
```json
{
  "metadata": {...},
  "document_structure": {...},
  "extracted_entities": [
    {"entity": "...", "relation": "...", "value": "..."}
  ],
  "comprehensive_summary": {...}
}
```

## Requirements

All extractors require:
- **Ollama**: Running locally with the specified model
- **Python 3.7+**
- **Dependencies**: requests, pathlib, json, re

The Automated Parser additionally requires:
- **pymupdf4llm**: For PDF text extraction
- **pdfplumber**: For PDF table extraction

## Error Handling

All extractors include:
- Automatic retry logic for API failures
- Connection testing on initialization
- Comprehensive error logging
- Graceful degradation when parsing fails

## Performance Tips

1. **Adjust request_delay**: Increase if hitting rate limits
2. **Use max_retries**: Set based on network stability
3. **Monitor logs**: Check `*_extraction.log` files for issues
4. **Batch processing**: Use `process_directory()` for multiple files
5. **Model selection**: Smaller models (3B) are faster but less accurate

## Examples

See the `examples/` directory for complete working examples of each extractor.
