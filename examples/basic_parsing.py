"""Basic PDF parsing example.

This example demonstrates how to use the different parsers
to extract information from pharmaceutical PDFs.
"""

from pathlib import Path
from pharma_extraction import (
    LLMOptimizedPharmaParser,
    PhraseBasedPharmaParser,
    SectionAwarePharmaParser
)


def example_phrase_based_parsing():
    """Example: Phrase-based parsing with semantic classification."""
    print("=" * 60)
    print("Example 1: Phrase-Based Parser")
    print("=" * 60)

    # Initialize parser
    parser = PhraseBasedPharmaParser()

    # Process a PDF file
    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")

    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found: {pdf_path}")
        print("Please place a pharmaceutical PDF in data/pdfs/ directory")
        return

    print(f"\nProcessing: {pdf_path.name}")

    # Load and parse document
    parser.load_document(str(pdf_path))

    # Save output
    output_path = Path("data/outputs/phrase_based_output.json")
    parser.save_optimized_json(str(output_path))

    print(f"\n✓ Saved phrase-based output to: {output_path}")
    print(f"  Total phrases extracted: {len(parser.structure.get('document_structure', {}).get('phrase_blocks', []))}")


def example_llm_optimized_parsing():
    """Example: Multi-representation parsing for different LLM tasks."""
    print("\n" + "=" * 60)
    print("Example 2: LLM-Optimized Parser (Multi-Representation)")
    print("=" * 60)

    # Initialize parser
    parser = LLMOptimizedPharmaParser()

    # Process a PDF file
    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")

    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found: {pdf_path}")
        return

    print(f"\nProcessing: {pdf_path.name}")

    # Load and parse document
    parser.load_document(str(pdf_path))

    # Save output
    output_path = Path("data/outputs/llm_optimized_output.json")
    parser.save_optimized_json(str(output_path))

    print(f"\n✓ Saved multi-representation output to: {output_path}")
    print(f"  Representations created:")
    print(f"    - flat_blocks: Sequential blocks for general queries")
    print(f"    - hierarchical: Nested structure for organization")
    print(f"    - markdown: Human-readable format")
    print(f"    - semantic_graph: Relationships and concepts")
    print(f"    - search_index: Optimized for search tasks")


def example_section_aware_parsing():
    """Example: Section-aware parsing with hierarchy detection."""
    print("\n" + "=" * 60)
    print("Example 3: Section-Aware Parser")
    print("=" * 60)

    # Initialize parser
    parser = SectionAwarePharmaParser()

    # Process a PDF file
    pdf_path = Path("data/pdfs/sample_pharmaceutical_document.pdf")

    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found: {pdf_path}")
        return

    print(f"\nProcessing: {pdf_path.name}")

    # Load and parse document
    parser.load_document(str(pdf_path))

    # Save output
    output_path = Path("data/outputs/section_aware_output.json")
    parser.save_optimized_json(str(output_path))

    print(f"\n✓ Saved section-aware output to: {output_path}")

    # Show section hierarchy
    if parser.structure:
        sections = parser.structure.get('document_structure', {}).get('sections', [])
        print(f"\n  Detected {len(sections)} main sections:")
        for section in sections[:5]:  # Show first 5
            print(f"    {section.get('number')}. {section.get('title', 'Untitled')}")


def batch_process_pdfs():
    """Example: Process multiple PDFs in batch."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)

    pdf_dir = Path("data/pdfs")
    output_dir = Path("data/outputs/batch")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  No PDFs found in {pdf_dir}")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) to process")

    # Use phrase-based parser for batch processing
    parser = PhraseBasedPharmaParser()

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        try:
            # Parse document
            parser.load_document(str(pdf_path))

            # Save output
            output_name = pdf_path.stem + "_parsed.json"
            output_path = output_dir / output_name
            parser.save_optimized_json(str(output_path))

            print(f"  ✓ Saved to: {output_path.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n✓ Batch processing complete!")
    print(f"  Outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Ensure directories exist
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    # Run examples
    example_phrase_based_parsing()
    example_llm_optimized_parsing()
    example_section_aware_parsing()
    batch_process_pdfs()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
