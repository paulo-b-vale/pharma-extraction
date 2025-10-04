"""
LLM-Optimized Pharmaceutical Document Parser

This parser creates multiple representations of pharmaceutical documents optimized
for different LLM tasks. It extracts content from PDFs and structures it into:
- Flat blocks for general queries
- Hierarchical structure for understanding document organization
- Markdown for human reading
- Semantic graph for relationship queries
- Search index for efficient lookup

The parser handles both text and tables, maintaining complete context throughout.
"""

import json
import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pymupdf4llm
import pdfplumber


class LLMOptimizedPharmaParser:
    """
    Parser that creates LLM-optimized multi-representation structures from pharmaceutical PDFs.

    This parser processes pharmaceutical documents and creates multiple structured
    representations optimized for different LLM use cases. It preserves complete
    hierarchical context and supports both text and tabular data.
    """

    def __init__(self, model_name: str = "llama3.2:3b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize parser with multiple LLM-optimized representations.

        Args:
            model_name: Name of the LLM model to use (default: llama3.2:3b)
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.raw_content = ""
        self.optimized_structure = {}
        self.document_loaded = False

    def list_pdfs_in_folder(self, folder_path: str) -> List[str]:
        """
        List all PDF files in a folder.

        Args:
            folder_path: Path to folder containing PDF files

        Returns:
            List of paths to PDF files found
        """
        try:
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                return []

            # Find all PDF files
            pdf_pattern = os.path.join(folder_path, "*.pdf")
            pdf_files = glob.glob(pdf_pattern, recursive=False)

            # Also check subdirectories
            pdf_pattern_recursive = os.path.join(folder_path, "**/*.pdf")
            pdf_files.extend(glob.glob(pdf_pattern_recursive, recursive=True))

            # Remove duplicates and sort
            pdf_files = sorted(list(set(pdf_files)))

            print(f"Found {len(pdf_files)} PDF files in {folder_path}")
            for i, pdf_file in enumerate(pdf_files, 1):
                rel_path = os.path.relpath(pdf_file, folder_path)
                print(f"   {i}. {rel_path}")

            return pdf_files

        except Exception as e:
            print(f"Error listing PDFs: {e}")
            return []

    def process_multiple_pdfs(self, folder_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDFs
            output_dir: Optional output directory for results

        Returns:
            Dictionary containing summary of processing results
        """
        # Get list of PDFs
        pdf_files = self.list_pdfs_in_folder(folder_path)

        if not pdf_files:
            return {
                "success": False,
                "error": "No PDF files found in the specified folder",
                "folder_path": folder_path
            }

        # Setup output directory
        if not output_dir:
            output_dir = f"processed_pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(output_dir, exist_ok=True)

        # Process each PDF
        results = {
            "success": True,
            "folder_path": folder_path,
            "output_directory": output_dir,
            "total_files": len(pdf_files),
            "processed_files": [],
            "failed_files": [],
            "processing_summary": {},
            "start_time": datetime.now().isoformat()
        }

        print(f"\nStarting batch processing of {len(pdf_files)} PDFs...")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = os.path.basename(pdf_path)
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_name}")

            try:
                # Reset parser state for new document
                self.raw_content = ""
                self.optimized_structure = {}
                self.document_loaded = False

                # Load and process the document
                if self.load_document(pdf_path):
                    # Generate output filename
                    output_filename = f"{Path(pdf_name).stem}_llm_optimized.json"
                    output_path = os.path.join(output_dir, output_filename)

                    # Save the optimized structure
                    json_file = self.save_optimized_json(output_path)

                    # Record success
                    file_result = {
                        "file_name": pdf_name,
                        "file_path": pdf_path,
                        "output_file": json_file,
                        "status": "success",
                        "total_blocks": self.optimized_structure.get("metadata", {}).get("total_blocks", 0),
                        "extraction_method": self.optimized_structure.get("document_metadata", {}).get("extraction_method", "unknown"),
                        "processing_time": datetime.now().isoformat()
                    }
                    results["processed_files"].append(file_result)

                    print(f"   Success - {file_result['total_blocks']} blocks extracted")

                else:
                    # Record failure
                    file_result = {
                        "file_name": pdf_name,
                        "file_path": pdf_path,
                        "status": "failed",
                        "error": "Document loading failed",
                        "processing_time": datetime.now().isoformat()
                    }
                    results["failed_files"].append(file_result)
                    print(f"   Failed to process")

            except Exception as e:
                # Record exception
                file_result = {
                    "file_name": pdf_name,
                    "file_path": pdf_path,
                    "status": "error",
                    "error": str(e),
                    "processing_time": datetime.now().isoformat()
                }
                results["failed_files"].append(file_result)
                print(f"   Error: {e}")

        # Finalize results
        results["end_time"] = datetime.now().isoformat()
        results["successful_count"] = len(results["processed_files"])
        results["failed_count"] = len(results["failed_files"])

        # Create processing summary
        if results["processed_files"]:
            extraction_methods = {}
            total_blocks = 0

            for file_result in results["processed_files"]:
                method = file_result.get("extraction_method", "unknown")
                extraction_methods[method] = extraction_methods.get(method, 0) + 1
                total_blocks += file_result.get("total_blocks", 0)

            results["processing_summary"] = {
                "total_blocks_extracted": total_blocks,
                "average_blocks_per_file": total_blocks / len(results["processed_files"]),
                "extraction_methods_used": extraction_methods
            }

        # Save batch processing report
        report_path = os.path.join(output_dir, "batch_processing_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: Dict[str, Any]):
        """Print a summary of batch processing results."""
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)

        print(f"Folder: {results['folder_path']}")
        print(f"Output: {results['output_directory']}")
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful_count']}")
        print(f"Failed: {results['failed_count']}")

        if results.get("processing_summary"):
            summary = results["processing_summary"]
            print(f"\nProcessing Statistics:")
            print(f"   Total blocks extracted: {summary.get('total_blocks_extracted', 0):,}")
            print(f"   Average blocks per file: {summary.get('average_blocks_per_file', 0):.1f}")
            print(f"   Extraction methods used:")
            for method, count in summary.get('extraction_methods_used', {}).items():
                print(f"      {method}: {count} files")

        if results["failed_files"]:
            print(f"\nFailed files:")
            for failed in results["failed_files"]:
                print(f"   {failed['file_name']}: {failed.get('error', 'Unknown error')}")

        print(f"\nReport saved: {os.path.join(results['output_directory'], 'batch_processing_report.json')}")
        print("=" * 70)

    def extract_pdf_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract PDF with full structural information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing extracted text, tables, and metadata
        """
        result = {
            "text": "",
            "tables": [],
            "pages": [],
            "method_used": ""
        }

        # Try pdfplumber first for better structure
        try:
            print("Extracting with pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                pages_data = []
                all_text_parts = []
                all_tables = []

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        all_text_parts.append(f"=== PAGE {page_num + 1} ===\n{page_text}")

                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table and any(any(cell for cell in row if cell) for row in table):
                            table_structure = self._process_table(table, page_num + 1, table_num + 1)
                            all_tables.append(table_structure)

                result["text"] = "\n\n".join(all_text_parts)
                result["tables"] = all_tables
                result["method_used"] = "pdfplumber"

                if len(result["text"].strip()) > 100:
                    return result

        except Exception as e:
            print(f"pdfplumber failed: {e}")

        # Fallback to pymupdf4llm
        try:
            print("Fallback to pymupdf4llm...")
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            if markdown_text:
                result["text"] = markdown_text
                result["method_used"] = "pymupdf4llm"
                result["tables"] = self._extract_markdown_tables(markdown_text)
                return result
        except Exception as e:
            print(f"pymupdf4llm failed: {e}")

        raise Exception("All extraction methods failed")

    def _process_table(self, table_data: List[List[str]], page: int, table_num: int) -> Dict[str, Any]:
        """Process table into LLM-friendly format."""
        if not table_data:
            return {}

        # Clean table data
        cleaned_rows = []
        for row in table_data:
            if row and any(cell for cell in row if cell and str(cell).strip()):
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return {}

        # Try to identify header
        potential_header = cleaned_rows[0] if cleaned_rows else []
        has_header = len(potential_header) > 0 and all(cell.strip() for cell in potential_header)

        return {
            "page": page,
            "table_number": table_num,
            "has_header": has_header,
            "header": potential_header if has_header else [],
            "data_rows": cleaned_rows[1:] if has_header else cleaned_rows,
            "raw_data": cleaned_rows,
            "formatted_text": self._table_to_text(cleaned_rows),
            "semantic_summary": self._analyze_table_semantically(cleaned_rows, has_header)
        }

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table to readable text format."""
        if not table_data:
            return ""

        formatted_lines = []
        for row in table_data:
            formatted_lines.append(" | ".join(str(cell) if cell else "" for cell in row))
        return "\n".join(formatted_lines)

    def _analyze_table_semantically(self, table_data: List[List[str]], has_header: bool) -> Dict[str, Any]:
        """Basic semantic analysis of table content."""
        if not table_data:
            return {}

        # Look for common pharmaceutical patterns
        dosage_keywords = ['mg', 'ml', 'dose', 'dosagem', 'quantidade', 'concentração']
        frequency_keywords = ['dia', 'vezes', 'horas', 'diário', 'semanal']
        age_keywords = ['anos', 'idade', 'adulto', 'criança', 'pediátrico']

        all_text = " ".join(" ".join(str(cell) for cell in row) for row in table_data).lower()

        semantic_type = "general"
        if any(keyword in all_text for keyword in dosage_keywords):
            if any(keyword in all_text for keyword in frequency_keywords):
                semantic_type = "dosage_schedule"
            else:
                semantic_type = "dosage_information"
        elif any(keyword in all_text for keyword in age_keywords):
            semantic_type = "age_specific_information"

        return {
            "semantic_type": semantic_type,
            "contains_dosage": any(keyword in all_text for keyword in dosage_keywords),
            "contains_frequency": any(keyword in all_text for keyword in frequency_keywords),
            "contains_age_info": any(keyword in all_text for keyword in age_keywords)
        }

    def _extract_markdown_tables(self, text: str) -> List[Dict]:
        """Extract tables from markdown text."""
        tables = []
        table_pattern = r'(\|[^\n]*\|\n(?:\|[^\n]*\|\n)*)'
        matches = re.finditer(table_pattern, text, re.MULTILINE)

        for i, match in enumerate(matches):
            table_text = match.group(1)
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            if len(lines) > 2:
                header_line = lines[0].strip('|').split('|')
                header = [cell.strip() for cell in header_line]

                data_rows = []
                for line in lines[2:]:  # Skip separator
                    row = [cell.strip() for cell in line.strip('|').split('|')]
                    data_rows.append(row)

                tables.append({
                    "page": "unknown",
                    "table_number": i + 1,
                    "has_header": True,
                    "header": header,
                    "data_rows": data_rows,
                    "raw_data": [header] + data_rows,
                    "formatted_text": table_text,
                    "semantic_summary": self._analyze_table_semantically([header] + data_rows, True)
                })

        return tables

    def create_llm_optimized_structure(self, text: str, tables: List[Dict]) -> Dict[str, Any]:
        """
        Create multiple representations optimized for different LLM tasks.

        Args:
            text: Extracted text from document
            tables: List of extracted tables

        Returns:
            Dictionary with multiple representations of the document
        """
        # Parse basic structure
        sections = self._parse_numbered_sections(text)

        # Create the multi-representation structure
        structure = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "total_blocks": 0,
                "optimization": "multi_representation_for_llm",
                "recommended_use": {
                    "general_queries": "flat_blocks",
                    "structure_questions": "hierarchical",
                    "human_reading": "markdown",
                    "relationship_queries": "semantic_graph",
                    "search_tasks": "search_index"
                }
            },
            "representations": {
                "flat_blocks": [],
                "hierarchical": {"sections": []},
                "markdown": "",
                "semantic_graph": {"nodes": {}, "relationships": [], "concepts": []},
                "search_index": {"sections": {}, "concepts": {}, "entities": {}, "full_text_blocks": []}
            }
        }

        # Build flat blocks (most important for LLM understanding)
        block_id = 1
        for section in sections:
            # Add section header as block
            section_block = {
                "id": f"block_{block_id}",
                "type": "primary_header",
                "content": f"{section['number']}. {section['title']}",
                "context": {
                    "section": section['number'],
                    "section_title": section['title'],
                    "full_path": [f"{section['number']}. {section['title']}"],
                    "breadcrumb": f"{section['number']}. {section['title']}",
                    "level": 1
                },
                "metadata": {
                    "importance": "high",
                    "content_type": "section_delimiter"
                }
            }
            structure["representations"]["flat_blocks"].append(section_block)
            block_id += 1

            # Process subsections and content
            for subsection in section.get('subsections', []):
                # Subsection header
                sub_block = {
                    "id": f"block_{block_id}",
                    "type": "unnumbered_header",
                    "content": subsection['title'],
                    "context": {
                        "section": section['number'],
                        "section_title": section['title'],
                        "full_path": [f"{section['number']}. {section['title']}", subsection['title']],
                        "breadcrumb": f"{section['number']}. {section['title']} > {subsection['title']}",
                        "level": 2,
                        "parent_type": "primary_header"
                    },
                    "metadata": {
                        "importance": "medium",
                        "content_type": "subsection_delimiter",
                        "unnumbered_level": 1
                    }
                }
                structure["representations"]["flat_blocks"].append(sub_block)
                block_id += 1

                # Content paragraphs
                for paragraph in subsection.get('paragraphs', []):
                    para_block = {
                        "id": f"block_{block_id}",
                        "type": "paragraph",
                        "content": paragraph,
                        "context": {
                            "section": section['number'],
                            "section_title": section['title'],
                            "full_path": [f"{section['number']}. {section['title']}", subsection['title']],
                            "breadcrumb": f"{section['number']}. {section['title']} > {subsection['title']}",
                            "level": 3,
                            "parent_type": "unnumbered_header",
                            "immediate_parent": subsection['title']
                        },
                        "metadata": {
                            "word_count": len(paragraph.split()),
                            "char_count": len(paragraph),
                            "semantic_context": f"Content under {subsection['title']} in section {section['number']}"
                        }
                    }
                    structure["representations"]["flat_blocks"].append(para_block)
                    block_id += 1

            # Add direct paragraphs (not under subsections)
            for paragraph in section.get('paragraphs', []):
                para_block = {
                    "id": f"block_{block_id}",
                    "type": "paragraph",
                    "content": paragraph,
                    "context": {
                        "section": section['number'],
                        "section_title": section['title'],
                        "full_path": [f"{section['number']}. {section['title']}"],
                        "breadcrumb": f"{section['number']}. {section['title']}",
                        "level": 2,
                        "parent_type": "primary_header"
                    },
                    "metadata": {
                        "word_count": len(paragraph.split()),
                        "char_count": len(paragraph),
                        "semantic_context": f"Direct content under section {section['number']}"
                    }
                }
                structure["representations"]["flat_blocks"].append(para_block)
                block_id += 1

        # Build hierarchical representation
        structure["representations"]["hierarchical"]["sections"] = [
            {
                "number": s['number'],
                "title": s['title'],
                "unnumbered_subsections": [
                    {
                        "title": sub['title'],
                        "level": 1,
                        "paragraphs": [{"text": p, "word_count": len(p.split()), "char_count": len(p)}
                                     for p in sub.get('paragraphs', [])],
                        "subsections": [],
                        "tables": []
                    } for sub in s.get('subsections', [])
                ],
                "paragraphs": [{"text": p, "word_count": len(p.split()), "char_count": len(p)}
                             for p in s.get('paragraphs', [])],
                "tables": []
            } for s in sections
        ]

        # Build markdown representation
        markdown_lines = ["<!-- LLM-Optimized Pharmaceutical Document -->",
                         "<!-- This document uses enhanced markdown with metadata for optimal LLM understanding -->", ""]

        for section in sections:
            markdown_lines.append(f"# {section['number']}. {section['title']}")
            markdown_lines.append(f'<!-- metadata: {{"type": "primary_header", "section": "{section["number"]}", "importance": "high"}} -->')
            markdown_lines.append("")

            for subsection in section.get('subsections', []):
                markdown_lines.append(f"## {subsection['title']}")
                markdown_lines.append(f'<!-- metadata: {{"type": "unnumbered_header", "level": 2, "importance": "medium"}} -->')
                markdown_lines.append("")

                for paragraph in subsection.get('paragraphs', []):
                    markdown_lines.append(paragraph)
                    markdown_lines.append(f'<!-- metadata: {{"type": "paragraph", "parent": "{subsection["title"]}", "level": 3}} -->')
                    markdown_lines.append("")

            for paragraph in section.get('paragraphs', []):
                markdown_lines.append(paragraph)
                markdown_lines.append(f'<!-- metadata: {{"type": "paragraph", "parent": "{section["title"]}", "level": 2}} -->')
                markdown_lines.append("")

        structure["representations"]["markdown"] = "\n".join(markdown_lines)

        # Build semantic graph (basic)
        nodes = {}
        for section in sections:
            node_id = f"section_{section['number']}"
            nodes[node_id] = {
                "type": "section",
                "title": section['title'],
                "content_type": "primary_header"
            }

        structure["representations"]["semantic_graph"]["nodes"] = nodes

        # Build search index
        search_sections = {}
        for section in sections:
            all_content = " ".join(section.get('paragraphs', []))
            for subsection in section.get('subsections', []):
                all_content += " " + " ".join(subsection.get('paragraphs', []))

            search_sections[f"section_{section['number']}"] = {
                "title": section['title'],
                "number": section['number'],
                "content_summary": all_content[:200] + "..." if len(all_content) > 200 else all_content,
                "subsections": [sub['title'] for sub in section.get('subsections', [])]
            }

        structure["representations"]["search_index"]["sections"] = search_sections

        # Update metadata
        structure["metadata"]["total_blocks"] = len(structure["representations"]["flat_blocks"])

        return structure

    def _parse_numbered_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse numbered sections from text."""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_subsection = None
        paragraph_buffer = []

        # Pattern for numbered headers (1., 2., 3., etc.)
        numbered_pattern = r'^(\d+)\.\s*(.+)'

        for line in lines:
            line = line.strip()
            if "expediente" in line or "Notificação de Alteração" in line or "Inclusão inicial de" in line:
                continue
            if not line:
                if paragraph_buffer:
                    paragraph_buffer.append("")
                continue

            # Check for numbered section
            numbered_match = re.match(numbered_pattern, line)
            if numbered_match:
                # Save previous paragraph
                if paragraph_buffer:
                    self._add_paragraph_to_section(current_section, current_subsection, paragraph_buffer)
                    paragraph_buffer = []

                # Create new section
                section_num = numbered_match.group(1)
                section_title = numbered_match.group(2).strip()

                current_section = {
                    "number": section_num,
                    "title": section_title,
                    "subsections": [],
                    "paragraphs": []
                }
                sections.append(current_section)
                current_subsection = None
                continue

            # Check if it's likely a subsection header
            if self._is_likely_header(line) and current_section and not re.match(r'^\d+\.', line):
                # Save previous paragraph
                if paragraph_buffer:
                    self._add_paragraph_to_section(current_section, current_subsection, paragraph_buffer)
                    paragraph_buffer = []

                # Create subsection
                current_subsection = {
                    "title": line,
                    "paragraphs": []
                }
                current_section["subsections"].append(current_subsection)
                continue

            # Regular content
            paragraph_buffer.append(line)

        # Don't forget the last paragraph
        if paragraph_buffer:
            self._add_paragraph_to_section(current_section, current_subsection, paragraph_buffer)

        return sections

    def _is_likely_header(self, line: str) -> bool:
        """Determine if a line is likely a header."""
        if len(line) > 100 or len(line) < 3:
            return False

        header_indicators = [
            line.isupper(),
            line.istitle(),
            len(line.split()) <= 8,
            not line.endswith('.'),
            any(keyword in line.lower() for keyword in ['indicações', 'contraindicações', 'posologia',
                                                       'dosagem', 'administração', 'precauções'])
        ]

        return sum(header_indicators) >= 2

    def _add_paragraph_to_section(self, section, subsection, paragraph_lines):
        """Add paragraph to appropriate section."""
        if not paragraph_lines or not any(line.strip() for line in paragraph_lines):
            return

        paragraph_text = '\n'.join(paragraph_lines).strip()

        if subsection:
            subsection["paragraphs"].append(paragraph_text)
        elif section:
            section["paragraphs"].append(paragraph_text)

    def load_document(self, pdf_path: str) -> bool:
        """
        Load and parse document into LLM-optimized structure.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if document was loaded successfully, False otherwise
        """
        print(f"Loading document: {os.path.basename(pdf_path)}")

        if not Path(pdf_path).exists():
            print(f"File not found: {pdf_path}")
            return False

        try:
            # Extract PDF content
            extraction_result = self.extract_pdf_with_structure(pdf_path)
            self.raw_content = extraction_result["text"]

            # Create LLM-optimized structure
            print("Creating LLM-optimized structure...")
            self.optimized_structure = self.create_llm_optimized_structure(
                self.raw_content,
                extraction_result.get("tables", [])
            )

            # Add document metadata
            self.optimized_structure["document_metadata"] = {
                "file_path": pdf_path,
                "file_name": Path(pdf_path).name,
                "extraction_method": extraction_result["method_used"],
                "processing_date": datetime.now().isoformat(),
                "total_text_length": len(self.raw_content),
                "total_tables": len(extraction_result.get("tables", []))
            }

            self.document_loaded = True
            print("Document processed successfully!")
            self._show_optimization_summary()
            return True

        except Exception as e:
            print(f"Error loading document: {e}")
            return False

    def _show_optimization_summary(self):
        """Show summary of optimization."""
        print("\n" + "=" * 70)
        print("LLM OPTIMIZATION SUMMARY")
        print("=" * 70)

        metadata = self.optimized_structure.get("metadata", {})
        doc_metadata = self.optimized_structure.get("document_metadata", {})

        print(f"File: {doc_metadata.get('file_name', 'Unknown')}")
        print(f"Method: {doc_metadata.get('extraction_method', 'Unknown')}")
        print(f"Total blocks: {metadata.get('total_blocks', 0)}")

        # Show block distribution
        flat_blocks = self.optimized_structure.get("representations", {}).get("flat_blocks", [])
        block_types = {}
        for block in flat_blocks:
            block_type = block.get("type", "unknown")
            block_types[block_type] = block_types.get(block_type, 0) + 1

        print("\nBlock distribution:")
        for block_type, count in block_types.items():
            print(f"   {block_type}: {count}")

        print("\nRecommended usage:")
        for usage, representation in metadata.get("recommended_use", {}).items():
            print(f"   {usage}: {representation}")

        print("=" * 70)

    def save_optimized_json(self, output_path: Optional[str] = None) -> str:
        """
        Save the optimized structure as JSON.

        Args:
            output_path: Optional path for output file

        Returns:
            Path to saved file
        """
        if not self.document_loaded:
            raise Exception("No document loaded")

        if not output_path:
            file_name = self.optimized_structure["document_metadata"]["file_name"]
            pdf_name = Path(file_name).stem
            output_path = f"{pdf_name}_llm_optimized.json"

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimized_structure, f, indent=2, ensure_ascii=False)

        print(f"\nLLM-optimized JSON saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size:,} bytes")
        return str(output_file)

    def query_flat_blocks(self, question: str) -> List[Dict]:
        """
        Query using flat blocks (best for most LLM tasks).

        Args:
            question: Query string

        Returns:
            List of relevant blocks sorted by relevance
        """
        if not self.document_loaded:
            return []

        flat_blocks = self.optimized_structure.get("representations", {}).get("flat_blocks", [])

        # Simple keyword matching for demo (you could enhance this)
        question_lower = question.lower()
        relevant_blocks = []

        for block in flat_blocks:
            content = block.get("content", "").lower()
            context = str(block.get("context", {})).lower()

            # Score based on keyword matches
            score = 0
            for word in question_lower.split():
                if word in content:
                    score += 2
                if word in context:
                    score += 1

            if score > 0:
                block_copy = block.copy()
                block_copy["relevance_score"] = score
                relevant_blocks.append(block_copy)

        # Sort by relevance
        relevant_blocks.sort(key=lambda x: x["relevance_score"], reverse=True)

        return relevant_blocks[:10]  # Return top 10

    def get_context_for_llm(self, question: str, max_blocks: int = 5) -> str:
        """
        Get optimized context for LLM query.

        Args:
            question: Query string
            max_blocks: Maximum number of blocks to include

        Returns:
            Formatted context string for LLM
        """
        relevant_blocks = self.query_flat_blocks(question)[:max_blocks]

        context_lines = [
            "=== RELEVANT DOCUMENT CONTEXT ===",
            f"Question: {question}",
            ""
        ]

        for i, block in enumerate(relevant_blocks, 1):
            context_lines.extend([
                f"[BLOCK {i}] {block['context']['breadcrumb']}",
                f"Type: {block['type']}",
                f"Content: {block['content']}",
                f"Relevance: {block['relevance_score']}",
                ""
            ])

        return "\n".join(context_lines)
