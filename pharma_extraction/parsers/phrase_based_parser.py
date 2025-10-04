"""
Phrase-Based Pharmaceutical Document Parser

This parser extracts pharmaceutical documents at the phrase level, breaking down
paragraphs into individual phrases while maintaining complete hierarchical context.
Each phrase is enriched with metadata about its semantic type, location in the
document hierarchy, and pharmaceutical relevance.

This approach enables more granular LLM queries and better precision in information
retrieval compared to paragraph-level extraction.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pymupdf4llm
import pdfplumber


class PhraseBasedPharmaParser:
    """
    Parser that extracts pharmaceutical documents at the phrase level with context.

    This parser breaks down document content into individual phrases while preserving
    the complete hierarchical context. It's optimized for precise information retrieval
    and supports both text and table data with semantic classification.
    """

    def __init__(self, model_name: str = "llama3.2:3b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize parser with phrase-level extraction and context mapping.

        Args:
            model_name: Name of the LLM model to use (default: llama3.2:3b)
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.raw_content = ""
        self.optimized_structure = {}
        self.document_loaded = False

        # Phrase ending patterns - more comprehensive detection
        self.phrase_endings = [
            r'\.',  # Period
            r';',   # Semicolon
            r':',   # Colon
            r'\!',  # Exclamation
            r'\?',  # Question mark
            r'\n\n', # Double newline (paragraph break)
            r'(?<=\d)\s*mg(?=\s|$)',  # After dosage amounts
            r'(?<=\d)\s*ml(?=\s|$)',  # After volume amounts
            r'(?<=\d)\s*g(?=\s|$)',   # After weight amounts
            r'(?<=\d)\s*%(?=\s|$)',   # After percentages
            r'(?<=\w)\s*\)(?=\s|$)',  # After closing parentheses
        ]

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
            "method_used": ""
        }

        # Try pdfplumber first for better structure
        try:
            print("Extracting with pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                all_text_parts = []
                all_tables = []

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        all_text_parts.append(f"=== PAGE {page_num + 1} ===\n{page_text}")

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

        cleaned_rows = []
        for row in table_data:
            if row and any(cell for cell in row if cell and str(cell).strip()):
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return {}

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
        return "\n".join([" | ".join(str(cell) if cell else "" for cell in row) for row in table_data])

    def _analyze_table_semantically(self, table_data: List[List[str]], has_header: bool) -> Dict[str, Any]:
        """Basic semantic analysis of table content."""
        if not table_data:
            return {}

        dosage_keywords = ['mg', 'ml', 'dose', 'dosagem', 'quantidade', 'concentração']
        frequency_keywords = ['dia', 'vezes', 'horas', 'diário', 'semanal']
        age_keywords = ['anos', 'idade', 'adulto', 'criança', 'pediátrico']
        all_text = " ".join(" ".join(str(cell) for cell in row) for row in table_data).lower()

        semantic_type = "general"
        if any(keyword in all_text for keyword in dosage_keywords):
            semantic_type = "dosage_information"
            if any(keyword in all_text for keyword in frequency_keywords):
                semantic_type = "dosage_schedule"
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

    def _split_into_phrases(self, text: str) -> List[str]:
        """
        Split text into meaningful phrases using multiple ending patterns.

        Ensures phrases are complete thoughts and not just sentence fragments.

        Args:
            text: Text to split into phrases

        Returns:
            List of phrase strings
        """
        if not text or not text.strip():
            return []

        # Clean up the text first
        text = re.sub(r'\s+', ' ', text.strip())

        # Create a comprehensive pattern for phrase endings
        ending_pattern = '|'.join(self.phrase_endings)

        # Split by the patterns but keep the delimiters
        parts = re.split(f'({ending_pattern})', text)

        phrases = []
        current_phrase = ""

        for i, part in enumerate(parts):
            if not part.strip():
                continue

            # If this part matches an ending pattern
            if re.match(f'^({ending_pattern})$', part.strip()):
                if current_phrase.strip():
                    # Add the ending to the current phrase
                    complete_phrase = (current_phrase + part).strip()
                    # Only add if it's substantive (more than just punctuation)
                    if len(complete_phrase) > 3 and any(c.isalnum() for c in complete_phrase):
                        phrases.append(complete_phrase)
                    current_phrase = ""
            else:
                current_phrase += part

        # Don't forget the last phrase if it doesn't end with a delimiter
        if current_phrase.strip():
            phrase = current_phrase.strip()
            if len(phrase) > 3 and any(c.isalnum() for c in phrase):
                phrases.append(phrase)

        # Post-process to merge very short phrases with the previous one
        merged_phrases = []
        for phrase in phrases:
            if len(merged_phrases) > 0 and len(phrase) < 15 and not phrase.endswith('.'):
                # Merge with previous phrase if current is too short
                merged_phrases[-1] = merged_phrases[-1] + " " + phrase
            else:
                merged_phrases.append(phrase)

        return merged_phrases

    def _link_tables_to_sections(self, sections: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assigns context from text sections to each table based on page number.

        Args:
            sections: List of section dictionaries
            tables: List of table dictionaries

        Returns:
            Tables with added parent_context
        """
        if not tables or not sections:
            return tables

        for table in tables:
            table_page = table['page']
            best_section = None
            best_subsection = None

            # Find the last section that appeared on or before the table's page
            for section in sections:
                if section['page_number'] <= table_page:
                    best_section = section
                else:
                    break

            # Within that section, find the last subsection
            if best_section:
                for subsection in best_section.get('subsections', []):
                    if subsection['page_number'] <= table_page:
                        best_subsection = subsection
                    else:
                        break

            # Add the found context to the table object
            table['parent_context'] = {}
            if best_section:
                table['parent_context']['section_number'] = best_section['number']
                table['parent_context']['section_title'] = best_section['title']
            if best_subsection:
                table['parent_context']['subsection_title'] = best_subsection['title']

        return tables

    def create_phrase_based_structure(self, text: str, tables: List[Dict]) -> Dict[str, Any]:
        """
        Create phrase-based structure with complete hierarchical context.

        Args:
            text: Extracted text from document
            tables: List of extracted tables

        Returns:
            Dictionary with phrase-based structure
        """
        sections = self._parse_numbered_sections(text)
        linked_tables = self._link_tables_to_sections(sections, tables)

        structure = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "extraction_type": "phrase_based",
                "total_phrases": 0,
                "total_sections": len(sections),
                "total_tables": len(linked_tables),
                "optimization": "phrase_level_with_hierarchical_context",
                "recommended_use": {
                    "precise_queries": "Use phrase_blocks for specific information retrieval",
                    "context_questions": "Use context_hierarchy for understanding document structure",
                    "search_tasks": "Use phrase_blocks with context filtering",
                    "content_analysis": "Combine phrase_blocks with their context metadata"
                }
            },
            "document_structure": {
                "context_hierarchy": self._build_context_hierarchy(sections),
                "phrase_blocks": [],
                "table_blocks": []
            }
        }

        phrase_id = 1

        # Process each section and create phrase blocks
        for section in sections:
            section_context = {
                "section_number": section['number'],
                "section_title": section['title'],
                "page_number": section['page_number']
            }

            # Process direct paragraphs under section
            for paragraph in section.get('paragraphs', []):
                phrases = self._split_into_phrases(paragraph)
                for phrase in phrases:
                    phrase_block = {
                        "phrase_id": f"phrase_{phrase_id}",
                        "content": phrase,
                        "context": {
                            "hierarchy": {
                                "section_number": section['number'],
                                "section_title": section['title'],
                                "subsection_title": None,
                                "level": "section_content"
                            },
                            "breadcrumb": f"Section {section['number']}: {section['title']}",
                            "full_path": [f"{section['number']}. {section['title']}"],
                            "page_number": section['page_number']
                        },
                        "metadata": {
                            "content_type": "section_paragraph_phrase",
                            "character_count": len(phrase),
                            "word_count": len(phrase.split()),
                            "contains_dosage": self._contains_dosage_info(phrase),
                            "contains_numbers": bool(re.search(r'\d', phrase)),
                            "phrase_type": self._classify_phrase_type(phrase)
                        }
                    }
                    structure["document_structure"]["phrase_blocks"].append(phrase_block)
                    phrase_id += 1

            # Process subsections
            for subsection in section.get('subsections', []):
                for paragraph in subsection.get('paragraphs', []):
                    phrases = self._split_into_phrases(paragraph)
                    for phrase in phrases:
                        phrase_block = {
                            "phrase_id": f"phrase_{phrase_id}",
                            "content": phrase,
                            "context": {
                                "hierarchy": {
                                    "section_number": section['number'],
                                    "section_title": section['title'],
                                    "subsection_title": subsection['title'],
                                    "level": "subsection_content"
                                },
                                "breadcrumb": f"Section {section['number']}: {section['title']} > {subsection['title']}",
                                "full_path": [f"{section['number']}. {section['title']}", subsection['title']],
                                "page_number": subsection['page_number']
                            },
                            "metadata": {
                                "content_type": "subsection_paragraph_phrase",
                                "character_count": len(phrase),
                                "word_count": len(phrase.split()),
                                "contains_dosage": self._contains_dosage_info(phrase),
                                "contains_numbers": bool(re.search(r'\d', phrase)),
                                "phrase_type": self._classify_phrase_type(phrase)
                            }
                        }
                        structure["document_structure"]["phrase_blocks"].append(phrase_block)
                        phrase_id += 1

        # Process tables with context
        for table in linked_tables:
            parent_context = table.get('parent_context', {})

            # Create context hierarchy for table
            table_context = {
                "hierarchy": {
                    "section_number": parent_context.get("section_number"),
                    "section_title": parent_context.get("section_title"),
                    "subsection_title": parent_context.get("subsection_title"),
                    "level": "table_content"
                }
            }

            # Build breadcrumb
            breadcrumb_parts = []
            if parent_context.get("section_number") and parent_context.get("section_title"):
                breadcrumb_parts.append(f"Section {parent_context['section_number']}: {parent_context['section_title']}")
            if parent_context.get("subsection_title"):
                breadcrumb_parts.append(parent_context["subsection_title"])
            breadcrumb_parts.append(f"Table {table.get('table_number', '?')}")
            table_context["breadcrumb"] = " > ".join(breadcrumb_parts)

            # Build full path
            full_path = []
            if parent_context.get("section_number") and parent_context.get("section_title"):
                full_path.append(f"{parent_context['section_number']}. {parent_context['section_title']}")
            if parent_context.get("subsection_title"):
                full_path.append(parent_context["subsection_title"])
            full_path.append(f"Table {table.get('table_number', '?')}")
            table_context["full_path"] = full_path
            table_context["page_number"] = table.get("page", "unknown")

            table_block = {
                "table_id": f"table_{len(structure['document_structure']['table_blocks']) + 1}",
                "content": {
                    "formatted_text": table.get("formatted_text", ""),
                    "header": table.get("header", []),
                    "data_rows": table.get("data_rows", [])
                },
                "context": table_context,
                "metadata": {
                    "content_type": "structured_table_data",
                    "has_header": table.get("has_header", False),
                    "rows": len(table.get("data_rows", [])),
                    "columns": len(table.get("header", [])) or (len(table.get("data_rows", [[]])[0]) if table.get("data_rows") else 0),
                    "semantic_summary": table.get("semantic_summary", {}),
                    "table_type": self._classify_table_type(table)
                }
            }
            structure["document_structure"]["table_blocks"].append(table_block)

        structure["metadata"]["total_phrases"] = len(structure["document_structure"]["phrase_blocks"])
        return structure

    def _build_context_hierarchy(self, sections: List[Dict]) -> Dict[str, Any]:
        """Build a clear hierarchical map of the document structure."""
        hierarchy = {
            "document_outline": [],
            "section_map": {},
            "navigation": {
                "total_sections": len(sections),
                "total_subsections": sum(len(s.get('subsections', [])) for s in sections)
            }
        }

        for section in sections:
            section_info = {
                "section_number": section['number'],
                "section_title": section['title'],
                "page_number": section['page_number'],
                "has_direct_content": len(section.get('paragraphs', [])) > 0,
                "subsections": []
            }

            for subsection in section.get('subsections', []):
                subsection_info = {
                    "subsection_title": subsection['title'],
                    "page_number": subsection['page_number'],
                    "has_content": len(subsection.get('paragraphs', [])) > 0
                }
                section_info["subsections"].append(subsection_info)

            hierarchy["document_outline"].append(section_info)
            hierarchy["section_map"][section['number']] = section_info

        return hierarchy

    def _contains_dosage_info(self, text: str) -> bool:
        """Check if text contains dosage-related information."""
        dosage_patterns = [
            r'\d+\s*(mg|ml|g|mcg|μg|%)',
            r'dose|dosagem|posologia',
            r'administr|aplicar|tomar',
            r'vezes?\s*(ao|por)\s*dia',
            r'\d+\s*x\s*dia'
        ]
        return any(re.search(pattern, text.lower()) for pattern in dosage_patterns)

    def _classify_phrase_type(self, phrase: str) -> str:
        """Classify the type of phrase for better LLM understanding."""
        phrase_lower = phrase.lower()

        if re.search(r'\d+\s*(mg|ml|g|%)', phrase_lower):
            return "dosage_instruction"
        elif any(word in phrase_lower for word in ['indicações', 'indicado', 'tratamento']):
            return "indication"
        elif any(word in phrase_lower for word in ['contraindicações', 'contraindicado', 'não']):
            return "contraindication"
        elif any(word in phrase_lower for word in ['precauções', 'cuidado', 'atenção']):
            return "precaution"
        elif any(word in phrase_lower for word in ['efeitos', 'reações', 'adversos']):
            return "side_effect"
        elif re.search(r'\d', phrase):
            return "numerical_data"
        elif phrase.endswith(':'):
            return "section_header"
        else:
            return "general_information"

    def _classify_table_type(self, table: Dict) -> str:
        """Classify table type based on content."""
        semantic_summary = table.get('semantic_summary', {})

        if semantic_summary.get('semantic_type') == 'dosage_schedule':
            return "dosage_schedule"
        elif semantic_summary.get('contains_dosage'):
            return "dosage_information"
        elif semantic_summary.get('contains_age_info'):
            return "age_specific_data"
        else:
            return "general_data"

    def _parse_numbered_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse numbered sections from text, tracking page numbers."""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_subsection = None
        paragraph_buffer = []
        current_page = 1

        numbered_pattern = r'^(\d+)\.\s*(.+)'
        page_pattern = r'^=== PAGE (\d+) ===$'

        for line in lines:
            page_match = re.match(page_pattern, line.strip())
            if page_match:
                current_page = int(page_match.group(1))
                continue

            line = line.strip()
            if "expediente" in line or "Notificação de Alteração" in line or "Inclusão inicial de" in line:
                continue

            if not line:
                if paragraph_buffer:
                    paragraph_buffer.append("")
                continue

            numbered_match = re.match(numbered_pattern, line)
            if numbered_match:
                if paragraph_buffer:
                    self._add_paragraph_to_section(current_section, current_subsection, paragraph_buffer)
                    paragraph_buffer = []

                section_num = numbered_match.group(1)
                section_title = numbered_match.group(2).strip()
                current_section = {
                    "number": section_num, "title": section_title, "page_number": current_page,
                    "subsections": [], "paragraphs": []
                }
                sections.append(current_section)
                current_subsection = None
                continue

            if self._is_likely_header(line) and current_section and not re.match(r'^\d+\.', line):
                if paragraph_buffer:
                    self._add_paragraph_to_section(current_section, current_subsection, paragraph_buffer)
                    paragraph_buffer = []

                current_subsection = {
                    "title": line, "page_number": current_page, "paragraphs": []
                }
                current_section["subsections"].append(current_subsection)
                continue

            paragraph_buffer.append(line)

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
        Load and parse document into phrase-based structure.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if document was loaded successfully, False otherwise
        """
        print(f"Loading document: {Path(pdf_path).name}")
        if not Path(pdf_path).exists():
            print(f"File not found: {pdf_path}")
            return False

        try:
            extraction_result = self.extract_pdf_with_structure(pdf_path)
            self.raw_content = extraction_result["text"]

            print("Creating phrase-based structure...")
            self.optimized_structure = self.create_phrase_based_structure(
                self.raw_content,
                extraction_result.get("tables", [])
            )

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
        """Show summary of phrase-based optimization."""
        print("\n" + "=" * 70)
        print("PHRASE-BASED OPTIMIZATION SUMMARY")
        print("=" * 70)

        metadata = self.optimized_structure.get("metadata", {})
        doc_metadata = self.optimized_structure.get("document_metadata", {})

        print(f"File: {doc_metadata.get('file_name', 'Unknown')}")
        print(f"Method: {doc_metadata.get('extraction_method', 'Unknown')}")
        print(f"Total phrases: {metadata.get('total_phrases', 0)}")
        print(f"Total sections: {metadata.get('total_sections', 0)}")
        print(f"Total tables: {metadata.get('total_tables', 0)}")

        phrase_blocks = self.optimized_structure.get("document_structure", {}).get("phrase_blocks", [])
        phrase_types = {}
        for block in phrase_blocks:
            phrase_type = block.get("metadata", {}).get("phrase_type", "unknown")
            phrase_types[phrase_type] = phrase_types.get(phrase_type, 0) + 1

        print("\nPhrase type distribution:")
        for phrase_type, count in phrase_types.items():
            print(f"  {phrase_type}: {count}")

        print("\nUsage recommendations:")
        for usage, description in metadata.get("recommended_use", {}).items():
            print(f"  {usage}: {description}")
        print("=" * 70)

    def save_optimized_json(self, output_path: Optional[str] = None) -> str:
        """
        Save the phrase-based structure as JSON.

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
            output_path = f"{pdf_name}_phrase_optimized.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimized_structure, f, indent=2, ensure_ascii=False)

        print(f"\nPhrase-optimized JSON saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size:,} bytes")
        return str(output_file)
