"""Batch processing example.

This example demonstrates how to process multiple PDFs
and extract knowledge at scale.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict
from pharma_extraction import PhraseBasedPharmaParser
from pharma_extraction.extractors import EnhancedPharmaceuticalKnowledgeExtractor
from pharma_extraction.utils import save_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for multiple pharmaceutical documents."""

    def __init__(self, pdf_dir: str, output_dir: str):
        """Initialize batch processor.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory for output files
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parser = PhraseBasedPharmaParser()
        self.extractor = EnhancedPharmaceuticalKnowledgeExtractor()

        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'results': []
        }

    def process_all(self) -> Dict:
        """Process all PDFs in the directory.

        Returns:
            Dictionary with batch processing statistics
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        self.stats['total_pdfs'] = len(pdf_files)

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return self.stats

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        start_time = time.time()

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

            try:
                result = self._process_single_pdf(pdf_path)
                result['status'] = 'success'
                self.stats['successful'] += 1

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                result = {
                    'filename': pdf_path.name,
                    'status': 'failed',
                    'error': str(e)
                }
                self.stats['failed'] += 1

            self.stats['results'].append(result)

        self.stats['total_time'] = time.time() - start_time

        # Save batch report
        self._save_batch_report()

        return self.stats

    def _process_single_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Parse PDF
        self.parser.load_document(str(pdf_path))

        # Save parsed JSON
        parsed_json = self.output_dir / f"{pdf_path.stem}_parsed.json"
        self.parser.save_optimized_json(str(parsed_json))

        # Extract knowledge
        self.extractor.extract_from_json(str(parsed_json))

        # Save knowledge graph
        knowledge_json = self.output_dir / f"{pdf_path.stem}_knowledge.json"
        self.extractor.save_results(str(knowledge_json))

        processing_time = time.time() - start_time

        # Get statistics
        stats = self.extractor.results.get('statistics', {})

        return {
            'filename': pdf_path.name,
            'processing_time': round(processing_time, 2),
            'phrases_processed': stats.get('total_phrases_processed', 0),
            'triples_extracted': stats.get('total_triples_extracted', 0),
            'parsed_output': str(parsed_json),
            'knowledge_output': str(knowledge_json)
        }

    def _save_batch_report(self):
        """Save batch processing report."""
        report = {
            'summary': {
                'total_pdfs': self.stats['total_pdfs'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'total_time': round(self.stats['total_time'], 2),
                'avg_time_per_pdf': round(
                    self.stats['total_time'] / max(self.stats['total_pdfs'], 1), 2
                )
            },
            'results': self.stats['results']
        }

        report_path = self.output_dir / "batch_report.json"
        save_json(report, report_path)
        logger.info(f"Batch report saved to: {report_path}")


def example_simple_batch():
    """Example: Simple batch processing."""
    print("=" * 60)
    print("Example 1: Simple Batch Processing")
    print("=" * 60)

    pdf_dir = "data/pdfs"
    output_dir = "data/outputs/batch_simple"

    processor = BatchProcessor(pdf_dir, output_dir)
    stats = processor.process_all()

    print(f"\n✓ Batch processing complete!")
    print(f"\nSummary:")
    print(f"  Total PDFs: {stats['total_pdfs']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total time: {stats['total_time']:.2f}s")

    if stats['successful'] > 0:
        avg_time = stats['total_time'] / stats['successful']
        print(f"  Avg time per PDF: {avg_time:.2f}s")


def example_parallel_batch():
    """Example: Parallel batch processing (conceptual).

    Note: This is a conceptual example. For true parallelization,
    use multiprocessing or async processing.
    """
    print("\n" + "=" * 60)
    print("Example 2: Parallel Processing Concept")
    print("=" * 60)

    print("\n💡 For production, consider:")
    print("   1. Using multiprocessing.Pool for CPU-bound tasks")
    print("   2. Using asyncio for I/O-bound tasks")
    print("   3. Using Celery for distributed task queue")
    print("   4. Processing PDFs in chunks")

    print("\nExample parallel approach:")
    print("""
    from multiprocessing import Pool

    def process_pdf(pdf_path):
        parser = PhraseBasedPharmaParser()
        parser.load_document(pdf_path)
        # ... rest of processing
        return result

    with Pool(processes=4) as pool:
        results = pool.map(process_pdf, pdf_files)
    """)


def example_selective_batch():
    """Example: Batch processing with filtering."""
    print("\n" + "=" * 60)
    print("Example 3: Selective Batch Processing")
    print("=" * 60)

    pdf_dir = Path("data/pdfs")
    output_dir = Path("data/outputs/batch_selective")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  No PDFs found in {pdf_dir}")
        return

    print(f"\nFound {len(pdf_files)} PDF files")

    # Filter: Only process PDFs larger than 100KB
    MIN_SIZE = 100 * 1024  # 100KB
    filtered_pdfs = [p for p in pdf_files if p.stat().st_size > MIN_SIZE]

    print(f"Filtering: Only processing PDFs > 100KB")
    print(f"Selected: {len(filtered_pdfs)} PDF(s)")

    if filtered_pdfs:
        processor = BatchProcessor(pdf_dir, output_dir)

        for pdf_path in filtered_pdfs:
            try:
                logger.info(f"Processing: {pdf_path.name}")
                result = processor._process_single_pdf(pdf_path)
                print(f"  ✓ {result['triples_extracted']} triples extracted")
            except Exception as e:
                print(f"  ✗ Error: {e}")


def example_incremental_batch():
    """Example: Incremental batch processing (skip already processed)."""
    print("\n" + "=" * 60)
    print("Example 4: Incremental Batch Processing")
    print("=" * 60)

    pdf_dir = Path("data/pdfs")
    output_dir = Path("data/outputs/batch_incremental")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  No PDFs found in {pdf_dir}")
        return

    # Check which PDFs are already processed
    processed_files = set()
    for json_file in output_dir.glob("*_knowledge.json"):
        # Extract original PDF name from output filename
        pdf_name = json_file.name.replace("_knowledge.json", ".pdf")
        processed_files.add(pdf_name)

    # Filter unprocessed PDFs
    unprocessed = [p for p in pdf_files if p.name not in processed_files]

    print(f"\nTotal PDFs: {len(pdf_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"To process: {len(unprocessed)}")

    if unprocessed:
        processor = BatchProcessor(pdf_dir, output_dir)
        print(f"\nProcessing {len(unprocessed)} new PDF(s)...")

        for pdf_path in unprocessed:
            try:
                result = processor._process_single_pdf(pdf_path)
                print(f"  ✓ {pdf_path.name}: {result['triples_extracted']} triples")
            except Exception as e:
                print(f"  ✗ {pdf_path.name}: {e}")
    else:
        print("\n✓ All PDFs already processed!")


if __name__ == "__main__":
    # Ensure directories exist
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)

    print("\n🚀 Starting Batch Processing Examples\n")

    # Run examples
    example_simple_batch()
    example_parallel_batch()
    example_selective_batch()
    example_incremental_batch()

    print("\n" + "=" * 60)
    print("All batch processing examples completed!")
    print("=" * 60)
