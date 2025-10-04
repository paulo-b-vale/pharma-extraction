"""MongoDB integration example.

This example demonstrates how to use MongoDB to store and query
pharmaceutical documents and knowledge graphs.
"""

import logging
from pathlib import Path
from pharma_extraction import PhraseBasedPharmaParser
from pharma_extraction.extractors import EnhancedPharmaceuticalKnowledgeExtractor
from pharma_extraction.database import MongoDBClient, DocumentStore, KnowledgeStore
from pharma_extraction.database.query_interface import PharmaQueryInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def example_setup_database():
    """Example 1: Setup MongoDB database."""
    print("=" * 60)
    print("Example 1: Setting Up MongoDB Database")
    print("=" * 60)

    print("\nConnecting to MongoDB...")
    with MongoDBClient() as client:
        if not client.is_connected():
            print("\n⚠️  Failed to connect to MongoDB!")
            print("Make sure MongoDB is running:")
            print("  - Install: https://www.mongodb.com/try/download/community")
            print("  - Or use Docker: docker run -d -p 27017:27017 mongo")
            return False

        print("✓ Connected to MongoDB")

        # Create indexes
        print("\nCreating indexes for optimal query performance...")
        client.create_indexes()
        print("✓ Indexes created")

        # Show stats
        stats = client.get_database_stats()
        print(f"\nDatabase: {stats.get('database_name')}")
        print(f"Collections: {len(stats.get('collections', {}))}")

        return True


def example_store_documents():
    """Example 2: Parse PDFs and store in MongoDB."""
    print("\n" + "=" * 60)
    print("Example 2: Storing Documents in MongoDB")
    print("=" * 60)

    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  No PDFs found in {pdf_dir}")
        return

    with MongoDBClient() as client:
        if not client.is_connected():
            print("\n⚠️  MongoDB not available")
            return

        doc_store = DocumentStore(client)
        parser = PhraseBasedPharmaParser()

        for pdf_path in pdf_files[:3]:  # Process first 3 PDFs
            print(f"\nProcessing: {pdf_path.name}")

            # Check if already processed
            existing = doc_store.get_document_by_filename(pdf_path.name)
            if existing:
                print(f"  ⏭️  Already in database, skipping")
                continue

            # Parse PDF
            parser.load_document(str(pdf_path))

            # Store in MongoDB
            doc_id = doc_store.store_document(
                filename=pdf_path.name,
                parsed_data=parser.structure,
                pdf_path=str(pdf_path)
            )

            if doc_id:
                print(f"  ✓ Stored with ID: {doc_id}")
            else:
                print(f"  ✗ Failed to store")

        # Show stats
        stats = doc_store.get_statistics()
        print(f"\n📊 Storage Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total phrases: {stats['total_phrases']}")


def example_store_knowledge():
    """Example 3: Extract and store knowledge triples."""
    print("\n" + "=" * 60)
    print("Example 3: Storing Knowledge Triples")
    print("=" * 60)

    with MongoDBClient() as client:
        if not client.is_connected():
            print("\n⚠️  MongoDB not available")
            return

        doc_store = DocumentStore(client)
        knowledge_store = KnowledgeStore(client)
        extractor = EnhancedPharmaceuticalKnowledgeExtractor()

        # Get all documents
        docs = doc_store.search_documents(limit=5)

        if not docs:
            print("\n⚠️  No documents in database")
            print("Run example_store_documents() first")
            return

        print(f"\nProcessing {len(docs)} document(s)...")

        for doc in docs:
            doc_id = doc['_id']
            filename = doc['filename']

            print(f"\n  {filename}")

            # Check if already extracted
            existing_triples = knowledge_store.get_triples_by_document(doc_id)
            if existing_triples:
                print(f"    ⏭️  Already extracted ({len(existing_triples)} triples)")
                continue

            # Extract knowledge from document structure
            # Note: We need the full parsed JSON, which is in doc['document_structure']
            extractor.results = {'extraction_results': {}, 'statistics': {}}

            # Simplified extraction for demo
            phrase_blocks = doc.get('document_structure', {}).get('phrase_blocks', [])

            if phrase_blocks:
                print(f"    Extracting from {len(phrase_blocks)} phrases...")

                # Here you would normally call extractor.extract_from_json()
                # For this example, we'll create some mock triples
                mock_results = {}
                for phrase in phrase_blocks[:5]:  # Just first 5 phrases
                    phrase_id = phrase.get('id')
                    mock_results[phrase_id] = {
                        'status': 'success',
                        'phrase_content': phrase.get('content'),
                        'context': phrase.get('context', {}),
                        'triples': []  # Would contain actual extracted triples
                    }

                # Store triples
                count = knowledge_store.store_knowledge_graph(
                    document_id=doc_id,
                    extraction_results=mock_results
                )
                print(f"    ✓ Stored {count} triples")

        # Show stats
        stats = knowledge_store.get_statistics()
        print(f"\n📊 Knowledge Graph Statistics:")
        print(f"  Total triples: {stats['total_triples']}")
        print(f"  Unique entities: {stats['unique_entities']}")
        print(f"  Unique relations: {stats['unique_relations']}")


def example_query_interface():
    """Example 4: Using the high-level query interface."""
    print("\n" + "=" * 60)
    print("Example 4: Querying with PharmaQueryInterface")
    print("=" * 60)

    with PharmaQueryInterface() as query:

        # Get database statistics
        print("\n📊 Database Overview:")
        stats = query.get_database_stats()

        doc_stats = stats.get('documents', {})
        kg_stats = stats.get('knowledge_graph', {})

        print(f"  Documents: {doc_stats.get('total_documents', 0)}")
        print(f"  Phrases: {doc_stats.get('total_phrases', 0)}")
        print(f"  Knowledge triples: {kg_stats.get('total_triples', 0)}")
        print(f"  Unique drugs: {kg_stats.get('unique_entities', 0)}")

        # List all drugs
        print("\n💊 Drugs in Database:")
        drugs = query.list_all_drugs()
        if drugs:
            for drug in drugs[:10]:  # Show first 10
                print(f"  - {drug}")
            if len(drugs) > 10:
                print(f"  ... and {len(drugs) - 10} more")
        else:
            print("  (No drugs extracted yet)")

        # Search documents
        print("\n🔍 Search Example:")
        search_term = "paracetamol"
        results = query.search_documents(search_term)
        print(f"  Found {len(results)} document(s) matching '{search_term}'")


def example_drug_queries():
    """Example 5: Query specific drug information."""
    print("\n" + "=" * 60)
    print("Example 5: Drug-Specific Queries")
    print("=" * 60)

    with PharmaQueryInterface() as query:

        # Get all drugs
        drugs = query.list_all_drugs()

        if not drugs:
            print("\n⚠️  No drugs in database")
            print("Run example_store_knowledge() first")
            return

        # Query first drug
        drug_name = drugs[0]
        print(f"\n💊 Information for: {drug_name}")

        # Get comprehensive info
        info = query.get_drug_info(drug_name)

        print(f"\n  Dosages:")
        for dosage in info.get('dosages', [])[:3]:
            print(f"    - {dosage}")

        print(f"\n  Indications:")
        for indication in info.get('indications', [])[:3]:
            print(f"    - {indication}")

        print(f"\n  Contraindications:")
        for contra in info.get('contraindications', [])[:3]:
            print(f"    - {contra}")

        # Get statistics
        stats = query.get_drug_stats(drug_name)
        print(f"\n  Statistics:")
        print(f"    Total triples: {stats.get('total_triples', 0)}")
        print(f"    Documents: {stats.get('document_count', 0)}")


def example_advanced_queries():
    """Example 6: Advanced query patterns."""
    print("\n" + "=" * 60)
    print("Example 6: Advanced Queries")
    print("=" * 60)

    with PharmaQueryInterface() as query:

        drugs = query.list_all_drugs()

        if len(drugs) < 2:
            print("\n⚠️  Need at least 2 drugs for comparison")
            return

        # Find similar drugs
        print(f"\n🔄 Finding drugs similar to {drugs[0]}...")
        similar = query.find_similar_drugs(drugs[0], by='indication')
        if similar:
            print(f"  Similar drugs:")
            for drug in similar[:5]:
                print(f"    - {drug}")
        else:
            print(f"  No similar drugs found")

        # Compare two drugs
        if len(drugs) >= 2:
            print(f"\n⚖️  Comparing {drugs[0]} vs {drugs[1]}...")
            comparison = query.compare_drugs(drugs[0], drugs[1])

            overlap = comparison['comparison']['indications']['overlap']
            if overlap:
                print(f"  Common indications:")
                for indication in overlap[:3]:
                    print(f"    - {indication}")
            else:
                print(f"  No overlapping indications")

        # Search by symptom
        print(f"\n🔍 Searching for drugs that treat 'dor'...")
        results = query.search_by_symptom('dor')
        if results:
            print(f"  Found {len(results)} drug(s):")
            for drug in results[:5]:
                print(f"    - {drug}")


if __name__ == "__main__":
    print("\n🚀 MongoDB Integration Examples\n")

    # Run examples in order
    if example_setup_database():
        example_store_documents()
        example_store_knowledge()
        example_query_interface()
        example_drug_queries()
        example_advanced_queries()

    print("\n" + "=" * 60)
    print("All MongoDB examples completed!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("   - Use MongoDB Compass to visualize your data")
    print("   - Build a REST API on top of the query interface")
    print("   - Add more advanced analytics")
    print("   - Create a web dashboard")
