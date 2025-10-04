"""Knowledge extraction example.

This example demonstrates how to extract knowledge triples
from parsed pharmaceutical documents.
"""

import logging
from pathlib import Path
from pharma_extraction import PhraseBasedPharmaParser
from pharma_extraction.extractors import (
    EnhancedPharmaceuticalKnowledgeExtractor,
    PharmaceuticalKnowledgeExtractor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_enhanced_extraction():
    """Example: Enhanced knowledge extraction from phrase-based JSON."""
    print("=" * 60)
    print("Example 1: Enhanced Knowledge Extraction")
    print("=" * 60)

    # First, parse the PDF using phrase-based parser
    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")
    parsed_json = Path("data/outputs/phrase_based_for_extraction.json")

    if pdf_path.exists():
        print(f"\nStep 1: Parsing PDF with PhraseBasedParser...")
        parser = PhraseBasedPharmaParser()
        parser.load_document(str(pdf_path))
        parser.save_optimized_json(str(parsed_json))
        print(f"✓ Parsed and saved to: {parsed_json}")
    elif not parsed_json.exists():
        print(f"\n⚠️  No PDF or parsed JSON found")
        print(f"Please place a PDF in: {pdf_path}")
        return

    # Extract knowledge triples
    print(f"\nStep 2: Extracting knowledge triples...")
    extractor = EnhancedPharmaceuticalKnowledgeExtractor()

    # Process the parsed JSON
    extractor.extract_from_json(str(parsed_json))

    # Save results
    output_path = Path("data/outputs/knowledge_triples_enhanced.json")
    extractor.save_results(str(output_path))

    print(f"\n✓ Knowledge extraction complete!")
    print(f"  Output saved to: {output_path}")

    # Display statistics
    stats = extractor.results.get('statistics', {})
    print(f"\nStatistics:")
    print(f"  Total phrases processed: {stats.get('total_phrases_processed', 0)}")
    print(f"  Total triples extracted: {stats.get('total_triples_extracted', 0)}")
    print(f"  Phrases with triples: {stats.get('phrases_with_triples', 0)}")

    # Show sample triples
    print(f"\nSample triples extracted:")
    for phrase_id, result in list(extractor.results.get('extraction_results', {}).items())[:3]:
        triples = result.get('triples', [])
        if triples:
            print(f"\n  From: {result.get('phrase_content', '')[:60]}...")
            for triple in triples[:2]:  # Show first 2 triples
                if isinstance(triple, list) and len(triple) == 3:
                    print(f"    [{triple[0]}] --{triple[1]}--> [{triple[2]}]")


def example_basic_extraction():
    """Example: Basic knowledge extraction from block-based JSON."""
    print("\n" + "=" * 60)
    print("Example 2: Basic Knowledge Extraction")
    print("=" * 60)

    from pharma_extraction import LLMOptimizedPharmaParser

    # First, parse the PDF using LLM-optimized parser
    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")
    parsed_json = Path("data/outputs/llm_optimized_for_extraction.json")

    if pdf_path.exists():
        print(f"\nStep 1: Parsing PDF with LLMOptimizedParser...")
        parser = LLMOptimizedPharmaParser()
        parser.load_document(str(pdf_path))
        parser.save_optimized_json(str(parsed_json))
        print(f"✓ Parsed and saved to: {parsed_json}")
    elif not parsed_json.exists():
        print(f"\n⚠️  No PDF or parsed JSON found")
        return

    # Extract knowledge triples
    print(f"\nStep 2: Extracting knowledge triples...")
    extractor = PharmaceuticalKnowledgeExtractor()

    # Process the parsed JSON
    extractor.extract_from_json(str(parsed_json))

    # Save results
    output_path = Path("data/outputs/knowledge_triples_basic.json")
    extractor.save_results(str(output_path))

    print(f"\n✓ Knowledge extraction complete!")
    print(f"  Output saved to: {output_path}")


def example_full_pipeline():
    """Example: Complete PDF to knowledge graph pipeline."""
    print("\n" + "=" * 60)
    print("Example 3: Full Automated Pipeline")
    print("=" * 60)

    from pharma_extraction.extractors import AutomatedPharmaParser

    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")

    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found: {pdf_path}")
        return

    print(f"\nProcessing complete pipeline for: {pdf_path.name}")
    print("This will:")
    print("  1. Extract text from PDF")
    print("  2. Analyze document structure")
    print("  3. Extract entities and relationships")
    print("  4. Generate knowledge graph")

    # Initialize automated pipeline
    pipeline = AutomatedPharmaParser()

    # Process PDF to knowledge graph
    output_dir = Path("data/outputs/automated_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline.process_pdf_to_knowledge_graph(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir)
        )

        print(f"\n✓ Pipeline complete!")
        print(f"  Check outputs in: {output_dir}")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        logging.exception("Pipeline error")


def compare_extractors():
    """Example: Compare enhanced vs basic extraction."""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Extractors")
    print("=" * 60)

    print("\n📊 Comparison of Extraction Methods:")
    print("\n1. Enhanced Extractor (Phrase-Based):")
    print("   ✓ More granular extraction")
    print("   ✓ Phrase-level context")
    print("   ✓ Better for detailed information")
    print("   ✓ Higher precision")
    print("   - Slower processing")

    print("\n2. Basic Extractor (Block-Based):")
    print("   ✓ Faster processing")
    print("   ✓ Good for overview extraction")
    print("   ✓ Works with any JSON structure")
    print("   - Less granular")
    print("   - May miss fine details")

    print("\n💡 Recommendation:")
    print("   - Use Enhanced for detailed pharmaceutical analysis")
    print("   - Use Basic for quick extraction or large batches")


if __name__ == "__main__":
    # Ensure directories exist
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    print("\n🚀 Starting Knowledge Extraction Examples\n")

    # Run examples
    example_enhanced_extraction()
    example_basic_extraction()
    example_full_pipeline()
    compare_extractors()

    print("\n" + "=" * 60)
    print("All knowledge extraction examples completed!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("   - Review the extracted knowledge graphs")
    print("   - Adjust extraction parameters in config")
    print("   - Build a database to store the triples")
    print("   - Create a query interface")
