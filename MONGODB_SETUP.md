# MongoDB Integration Guide

This guide explains how to set up and use MongoDB with the pharmaceutical extraction system.

## Why MongoDB?

MongoDB is perfect for this use case because:

✅ **Direct JSON Storage** - Your parsed PDFs are already in JSON format
✅ **Flexible Schema** - Handles different document structures easily
✅ **Powerful Queries** - Find drugs, symptoms, dosages with simple queries
✅ **Scalable** - Grows with your data from hundreds to millions of documents
✅ **Industry Standard** - Widely used in enterprises (great for job applications)

## Installation

### Option 1: Install MongoDB Locally

**Windows:**
1. Download from https://www.mongodb.com/try/download/community
2. Run installer (MongoDB Compass GUI included)
3. MongoDB will start automatically on `localhost:27017`

**Linux (Ubuntu/Debian):**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Option 2: Use Docker (Easiest)

```bash
# Run MongoDB in Docker
docker run -d \
  --name pharma-mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:latest

# Verify it's running
docker ps
```

### Option 3: MongoDB Atlas (Cloud - Free Tier)

1. Go to https://www.mongodb.com/cloud/atlas
2. Create free account
3. Create a free cluster
4. Get your connection string
5. Set environment variable:
   ```bash
   export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/pharma_extraction"
   ```

## Verify Installation

```bash
# Test connection
python -c "from pymongo import MongoClient; print('✓ Connected' if MongoClient().server_info() else '✗ Failed')"
```

## Database Structure

The system creates 3 collections:

### 1. `documents` Collection
Stores parsed PDF documents with their complete structure.

```javascript
{
  "_id": ObjectId("..."),
  "filename": "paracetamol.pdf",
  "upload_date": ISODate("2025-10-04T..."),
  "metadata": {
    "extraction_type": "phrase_based",
    "total_phrases": 167
  },
  "document_structure": {
    "phrase_blocks": [...]
  }
}
```

**Indexes:**
- `filename` (unique)
- `upload_date` (descending)
- `metadata.drug_name` (text search)

### 2. `phrases` Collection
Individual phrases extracted from documents for fast querying.

```javascript
{
  "_id": ObjectId("..."),
  "document_id": "...",
  "phrase_id": "phrase_1",
  "content": "Paracetamol 500mg é indicado para dor de cabeça",
  "phrase_type": "indication",
  "section_title": "INDICAÇÕES",
  "page_number": 2
}
```

**Indexes:**
- `document_id`
- `phrase_type`
- `content` (text search)
- `section_title`

### 3. `knowledge_triples` Collection
Extracted knowledge in triple format: [entity, relation, value]

```javascript
{
  "_id": ObjectId("..."),
  "document_id": "...",
  "entity": "Paracetamol",
  "relation": "has_dosage",
  "value": "500mg 3x ao dia",
  "source_phrase": "A dose recomendada é 500mg...",
  "context": {
    "section_title": "POSOLOGIA",
    "breadcrumb": "4. POSOLOGIA"
  },
  "extracted_date": ISODate("...")
}
```

**Indexes:**
- `document_id`
- `entity` (text search)
- `relation`
- `value` (text search)
- Compound index: `(entity, relation, value)`

## Quick Start

### 1. Setup Database

```python
from pharma_extraction.database.query_interface import PharmaQueryInterface

# Connect and setup (creates indexes)
query = PharmaQueryInterface()
query.setup_database()
```

### 2. Store Documents

```python
from pharma_extraction import PhraseBasedPharmaParser
from pharma_extraction.database import MongoDBClient, DocumentStore

# Parse PDF
parser = PhraseBasedPharmaParser()
parser.load_document("bula_paracetamol.pdf")

# Store in MongoDB
with MongoDBClient() as client:
    store = DocumentStore(client)
    doc_id = store.store_document(
        filename="bula_paracetamol.pdf",
        parsed_data=parser.structure
    )
    print(f"Stored with ID: {doc_id}")
```

### 3. Extract and Store Knowledge

```python
from pharma_extraction.extractors import EnhancedPharmaceuticalKnowledgeExtractor
from pharma_extraction.database import KnowledgeStore

# Extract knowledge
extractor = EnhancedPharmaceuticalKnowledgeExtractor()
extractor.extract_from_json("parsed_output.json")

# Store triples
with MongoDBClient() as client:
    knowledge = KnowledgeStore(client)
    count = knowledge.store_knowledge_graph(
        document_id=doc_id,
        extraction_results=extractor.results['extraction_results'],
        statistics=extractor.results['statistics']
    )
    print(f"Stored {count} knowledge triples")
```

### 4. Query the Database

```python
with PharmaQueryInterface() as query:

    # Get all info about a drug
    info = query.get_drug_info("Paracetamol")
    print(info['dosages'])
    print(info['indications'])

    # Search by symptom
    drugs = query.search_by_symptom("dor de cabeça")
    print(f"Found {len(drugs)} drugs for headache")

    # Compare drugs
    comparison = query.compare_drugs("Paracetamol", "Ibuprofeno")
    print(comparison['comparison']['indications']['overlap'])

    # Get statistics
    stats = query.get_database_stats()
    print(f"Total documents: {stats['documents']['total_documents']}")
    print(f"Total triples: {stats['knowledge_graph']['total_triples']}")
```

## Common Queries

### Find all drugs in database
```python
drugs = query.list_all_drugs()
```

### Get dosage information
```python
dosages = query.get_dosage_info("Paracetamol")
```

### Find similar drugs
```python
similar = query.find_similar_drugs("Paracetamol", by="indication")
```

### Search documents
```python
docs = query.search_documents("analgésico")
```

### Get drug statistics
```python
stats = query.get_drug_stats("Paracetamol")
```

## MongoDB Compass (GUI)

MongoDB Compass is a free GUI tool for visualizing your data:

1. **Download**: https://www.mongodb.com/try/download/compass
2. **Connect**: `mongodb://localhost:27017`
3. **Browse**: Navigate through collections
4. **Query**: Use the query builder
5. **Visualize**: See your data structure

## Configuration

### Environment Variables

```bash
# MongoDB connection
export MONGODB_URI="mongodb://localhost:27017"

# Or for MongoDB Atlas
export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/pharma_extraction"
```

### In Code

```python
from pharma_extraction.database import MongoDBClient

# Custom connection
client = MongoDBClient(
    connection_string="mongodb://localhost:27017",
    database_name="my_pharma_db"
)
```

## Performance Tips

### 1. Indexes are Crucial
The setup creates indexes automatically, but verify with:
```python
client.create_indexes()
```

### 2. Batch Inserts
For large datasets, use batch operations:
```python
# Don't do this (slow)
for triple in triples:
    collection.insert_one(triple)

# Do this instead (fast)
collection.insert_many(triples)
```

### 3. Limit Results
Always use `.limit()` for large result sets:
```python
results = query.search_documents("drug", limit=100)
```

### 4. Use Projections
Only fetch fields you need:
```python
docs = collection.find(
    {"filename": "paracetamol.pdf"},
    {"metadata": 1, "filename": 1}  # Only these fields
)
```

## Backup and Restore

### Backup
```bash
# Backup entire database
mongodump --db pharma_extraction --out backup/

# Backup specific collection
mongodump --db pharma_extraction --collection knowledge_triples --out backup/
```

### Restore
```bash
# Restore database
mongorestore --db pharma_extraction backup/pharma_extraction/
```

## Troubleshooting

### Can't connect to MongoDB
```bash
# Check if MongoDB is running
sudo systemctl status mongod  # Linux
brew services list  # macOS
docker ps  # Docker

# Check port
netstat -an | grep 27017
```

### Connection timeout
- Check firewall settings
- Verify MongoDB is listening on 0.0.0.0 (not just 127.0.0.1)
- For Atlas: whitelist your IP address

### Slow queries
- Ensure indexes are created
- Use MongoDB Compass to analyze query performance
- Check collection stats: `db.collection.stats()`

## Next Steps

1. **Run the examples**: `python examples/mongodb_integration.py`
2. **Process your PDFs**: Store all bulas in MongoDB
3. **Build a REST API**: Use FastAPI to expose the query interface
4. **Create a dashboard**: Visualize drug information
5. **Add vector search**: Use MongoDB Atlas Vector Search for semantic queries

## Resources

- MongoDB Documentation: https://docs.mongodb.com/
- PyMongo Tutorial: https://pymongo.readthedocs.io/
- MongoDB University: https://university.mongodb.com/ (Free courses)
- Atlas Free Tier: https://www.mongodb.com/cloud/atlas/register
