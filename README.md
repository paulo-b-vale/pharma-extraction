# Pharmaceutical Document Extraction & Knowledge Graph System

> **Production-ready Python package** for extracting structured information and knowledge graphs from pharmaceutical documents (PDFs). Optimized for Brazilian drug information leaflets ("bulas").

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Project Overview

This system transforms unstructured pharmaceutical PDFs into structured knowledge graphs stored in MongoDB. It uses local LLMs (Ollama) to extract entities and relationships, making drug information queryable and analyzable.

**Key Innovation**: Multi-representation parsing that optimizes document structure for different LLM tasks (search, Q&A, relationship extraction).

## ✨ Features

### 📄 PDF Processing
- **3 Parsing Strategies**:
  - **Phrase-Based**: Granular extraction with semantic classification (dosage, indication, contraindication, etc.)
  - **LLM-Optimized**: Multi-representation output (flat, hierarchical, markdown, semantic graph, search index)
  - **Section-Aware**: Intelligent section detection with hierarchy tracking

### 🧠 Knowledge Extraction
- **Entity-Relation-Value Triples**: `["Paracetamol", "has_dosage", "500mg"]`
- **Context Preservation**: Every triple linked to source phrase, section, and page number
- **LLM-Powered**: Uses Ollama (llama3.2:3b) for intelligent extraction
- **Template Filtering**: Removes boilerplate text, keeps actual drug information

### 💾 MongoDB Integration
- **Document Store**: Full parsed PDFs with searchable phrases
- **Knowledge Graph**: Triples stored for fast querying
- **High-Level Query Interface**: Simple Python API for complex queries
- **Optimized Indexes**: Fast search across millions of triples

### 🏗️ Production-Ready
- Clean module architecture (no notebooks)
- Environment-based configuration
- Comprehensive error handling
- Type hints throughout
- Batch processing support
- Logging and monitoring

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# MongoDB (choose one):
# Option 1: Docker
docker run -d -p 27017:27017 --name mongo mongo

# Option 2: Local install
# Windows: https://www.mongodb.com/try/download/community
# Linux: sudo apt install mongodb
# macOS: brew install mongodb-community

# Ollama with llama3.2:3b
# Download from https://ollama.ai/
ollama pull llama3.2:3b
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pharma-extraction.git
cd pharma-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage (30 seconds to first result)

```python
from pharma_extraction import PhraseBasedPharmaParser
from pharma_extraction.database import MongoDBClient, DocumentStore

# 1. Parse PDF
parser = PhraseBasedPharmaParser()
parser.load_document("data/pdfs/paracetamol.pdf")

# 2. Store in MongoDB
with MongoDBClient() as client:
    store = DocumentStore(client)
    doc_id = store.store_document("paracetamol.pdf", parser.structure)
    print(f"✓ Stored document: {doc_id}")
```

### Extract Knowledge

```python
from pharma_extraction.extractors import EnhancedPharmaceuticalKnowledgeExtractor
from pharma_extraction.database import KnowledgeStore

# Extract knowledge triples
extractor = EnhancedPharmaceuticalKnowledgeExtractor()
extractor.extract_from_json("parsed_output.json")

# Store in MongoDB
with MongoDBClient() as client:
    knowledge = KnowledgeStore(client)
    count = knowledge.store_knowledge_graph(
        document_id=doc_id,
        extraction_results=extractor.results['extraction_results']
    )
    print(f"✓ Extracted {count} knowledge triples")
```

### Query the Knowledge Graph

```python
from pharma_extraction.database.query_interface import PharmaQueryInterface

with PharmaQueryInterface() as query:
    # Get all info about a drug
    info = query.get_drug_info("Paracetamol")
    print(f"Dosages: {info['dosages']}")
    print(f"Indications: {info['indications']}")

    # Find drugs for a symptom
    drugs = query.search_by_symptom("dor de cabeça")
    print(f"Found {len(drugs)} drugs for headache")

    # Compare two drugs
    comparison = query.compare_drugs("Paracetamol", "Ibuprofeno")
```

## 📁 Project Structure

```
pharma-extraction/
├── pharma_extraction/              # Main package
│   ├── parsers/                    # PDF parsing (3 strategies)
│   │   ├── phrase_based_parser.py
│   │   ├── llm_optimized_parser.py
│   │   └── section_aware_parser.py
│   ├── extractors/                 # Knowledge extraction
│   │   ├── enhanced_knowledge_extractor.py
│   │   ├── basic_knowledge_extractor.py
│   │   └── automated_pipeline.py
│   ├── database/                   # MongoDB integration
│   │   ├── mongodb_client.py
│   │   ├── document_store.py
│   │   ├── knowledge_store.py
│   │   └── query_interface.py
│   ├── utils/                      # Utilities
│   │   ├── llm_client.py
│   │   ├── text_processing.py
│   │   └── file_handlers.py
│   └── config/                     # Configuration
│       └── settings.py
├── examples/                       # Working examples
│   ├── basic_parsing.py
│   ├── knowledge_extraction.py
│   ├── batch_processing.py
│   └── mongodb_integration.py
├── data/
│   ├── pdfs/                       # Input PDFs
│   └── outputs/                    # Generated outputs
├── tests/                          # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## 📊 Example Output

### Knowledge Triple Format
```json
{
  "entity": "Paracetamol",
  "relation": "has_dosage",
  "value": "500mg a cada 6 horas",
  "source_phrase": "A dose recomendada é 500mg a cada 6 horas.",
  "context": {
    "section_title": "POSOLOGIA",
    "breadcrumb": "4. COMO DEVO USAR ESTE MEDICAMENTO? > 4.1 Posologia"
  }
}
```

### Query Results
```python
>>> query.get_drug_info("Paracetamol")
{
  'drug_name': 'Paracetamol',
  'dosages': ['500mg 3-4x ao dia', '1g máximo por dose'],
  'indications': ['dor', 'febre', 'dor de cabeça'],
  'contraindications': ['insuficiência hepática grave'],
  'side_effects': ['náusea', 'reações alérgicas']
}
```

## 🔧 Configuration

### Environment Variables
```bash
# MongoDB
export MONGODB_URI="mongodb://localhost:27017"

# Ollama
export PHARMA_OLLAMA_URL="http://localhost:11434"
export PHARMA_OLLAMA_MODEL="llama3.2:3b"
export PHARMA_OLLAMA_TIMEOUT=120

# Paths
export PHARMA_PDF_DIR="./data/pdfs"
export PHARMA_OUTPUT_DIR="./data/outputs"

# Logging
export PHARMA_LOG_LEVEL="INFO"
```

### Programmatic Configuration
```python
from pharma_extraction.config import Config

config = Config()
config.OLLAMA_MODEL = "llama3:8b"
config.OLLAMA_TEMPERATURE = 0.1
config.create_directories()
```

## 📚 Documentation

- [MongoDB Setup Guide](MONGODB_SETUP.md) - Complete MongoDB integration guide
- [Examples](examples/) - Working code examples
- [API Documentation](pharma_extraction/) - Module-level docs

## 🎓 Use Cases

- **Pharmaceutical Research**: Extract structured data from thousands of drug leaflets
- **Healthcare AI**: Build knowledge bases for medical Q&A systems
- **Regulatory Compliance**: Compare drug information across manufacturers
- **Clinical Decision Support**: Query drug interactions, contraindications, dosages
- **Drug Database Creation**: Build searchable pharmaceutical databases

## 🛣️ Roadmap

- [x] PDF parsing with multiple strategies
- [x] LLM-based knowledge extraction
- [x] MongoDB storage and querying
- [ ] REST API with FastAPI
- [ ] Web dashboard for visualization
- [ ] Vector search for semantic queries
- [ ] Multi-language support (English, Spanish)
- [ ] Docker Compose for easy deployment
- [ ] Automated testing suite
- [ ] CI/CD pipeline

## 🧪 Development

### Run Examples
```bash
# Parse PDFs
python examples/basic_parsing.py

# Extract knowledge
python examples/knowledge_extraction.py

# MongoDB integration
python examples/mongodb_integration.py
```

### Run Tests
```bash
pytest
```

### Code Quality
```bash
black pharma_extraction/
flake8 pharma_extraction/
mypy pharma_extraction/
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- PDF processing powered by [pdfplumber](https://github.com/jsvine/pdfplumber)
- Database: [MongoDB](https://www.mongodb.com/)

## 📈 Project Stats

- **~5,000 lines** of production Python code
- **6 parsing modules** + 4 database modules
- **4 working examples** with documentation
- **MongoDB integration** with optimized indexes
- **Type hints** throughout for better IDE support

---

**⭐ Star this repo if you find it useful!**

*Built to showcase production-ready ML/NLP engineering skills for enterprise job applications.*
#   p h a r m a - e x t r a c t i o n  
 #   p h a r m a - e x t r a c t i o n  
 