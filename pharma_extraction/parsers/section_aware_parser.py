"""
Section-Aware Pharmaceutical Document Parser

This parser recognizes and tracks pharmaceutical document sections and headers,
providing context-aware entity extraction. It uses intelligent sentence splitting
that respects pharmaceutical abbreviations and section boundaries.

The parser integrates with Ollama LLMs to extract structured entities (medication
names, dosages, indications, contraindications, etc.) while maintaining full
awareness of which section each piece of information comes from.
"""

import json
import os
import re
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pymupdf4llm
import pdfplumber


class SectionAwarePharmaParser:
    """
    Parser with section-aware analysis and entity extraction.

    This parser identifies document structure (sections and subsections) and
    extracts pharmaceutical entities while maintaining complete section context.
    It uses local LLM integration for intelligent entity recognition.
    """

    def __init__(self, model_name: str = "llama3:8b"):
        """
        Initialize section-aware parser with header tracking.

        Args:
            model_name: Name of the Ollama model to use for entity extraction
        """
        self.model_name = model_name
        self.raw_content = ""
        self.structured_data = {}
        self.document_loaded = False

        # Common pharmaceutical abbreviations that should NOT end sentences
        self.pharma_abbreviations = {
            'mg', 'ml', 'mcg', 'kg', 'g', 'l', 'dl', 'mmol', 'mol',  # Units
            'q.s.p', 'c.q.s', 'q.s', 'c.s.p',  # Pharmaceutical Latin
            'ltda', 'ltd', 'inc', 'corp', 'sa', 'co',  # Company abbreviations
            'dr', 'dra', 'prof', 'sr', 'sra',  # Titles
            'etc', 'ex', 'vs', 'e.g', 'i.e',  # Common abbreviations
            'cnpj', 'cpf', 'rg', 'crf', 'crm',  # Brazilian document types
            'anvisa', 'ms', 'rdc', 'vp', 'vps',  # Brazilian regulatory
            'd.d', 'p.ex', 'n°', 'nº'  # Other common abbreviations
        }

        # Brazilian pharmaceutical document section patterns
        self.section_patterns = {
            # Primary numbered sections
            r'^\s*I+\)\s*(.+)$': 'primary_section',  # I), II), III)
            r'^\s*\d+\.\s*(.+)$': 'numbered_section',  # 1., 2., 3.

            # Common pharmaceutical sections
            r'^\s*(IDENTIFICAÇÃO|IDENTIFICACAO)\s*(DO\s*MEDICAMENTO)?\s*$': 'identification',
            r'^\s*(INFORMAÇÕES|INFORMACOES)\s*(AO\s*PACIENTE)?\s*$': 'patient_info',
            r'^\s*(COMPOSIÇÃO|COMPOSICAO)\s*$': 'composition',
            r'^\s*(APRESENTAÇÕES|APRESENTACOES)\s*$': 'presentations',
            r'^\s*(INDICAÇÕES|INDICACOES)\s*$': 'indications',
            r'^\s*(CONTRAINDICAÇÕES|CONTRAINDICACOES)\s*$': 'contraindications',
            r'^\s*(PRECAUÇÕES|PRECAUCOES)\s*$': 'precautions',
            r'^\s*(REAÇÕES\s*ADVERSAS|REACOES\s*ADVERSAS|EFEITOS\s*ADVERSOS)\s*$': 'adverse_effects',
            r'^\s*(INTERAÇÕES|INTERACOES)\s*(MEDICAMENTOSAS)?\s*$': 'drug_interactions',
            r'^\s*(POSOLOGIA|DOSAGEM)\s*$': 'dosage',
            r'^\s*(SUPERDOSAGEM|SUPERDOSE)\s*$': 'overdose',
            r'^\s*ARMAZENAMENTO\s*$': 'storage',
            r'^\s*DIZERES\s*LEGAIS\s*$': 'legal_info',

            # Question-style headers
            r'^\s*\d+\.\s*(PARA\s*QUE|O\s*QUE|COMO|QUANDO|ONDE|QUAIS)\s*.*\?\s*$': 'question_header'
        }

        self.setup_ollama()

    def setup_ollama(self):
        """Setup Ollama model automatically."""
        print(f"Setting up Ollama model: {self.model_name}")
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            print("Ollama CLI found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("Ollama CLI not found. Please install Ollama first.")

        try:
            print(f"Pulling model {self.model_name}...")
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"Model {self.model_name} ready")
        except Exception as e:
            print(f"Error with model setup: {e}")

    def call_ollama_raw(self, prompt: str, extra_flags: str = "") -> str:
        """
        Call ollama with exact prompt.

        Args:
            prompt: Prompt to send to the model
            extra_flags: Additional command-line flags

        Returns:
            Model response as string
        """
        cmd = ["ollama", "run", self.model_name]
        if extra_flags:
            cmd += shlex.split(extra_flags)

        try:
            proc = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True, timeout=60
            )
            return proc.stdout.strip() or proc.stderr.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama call timed out")
        except Exception as e:
            raise RuntimeError(f"Error calling ollama: {e}")

    def extract_pdf_content(self, pdf_path: str) -> str:
        """
        Extract text content from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)
                if all_text:
                    return "\n\n".join(all_text)
        except Exception as e:
            print(f"pdfplumber failed: {e}")

        try:
            return pymupdf4llm.to_markdown(pdf_path)
        except Exception as e:
            print(f"pymupdf4llm failed: {e}")
            raise Exception("All extraction methods failed")

    def detect_section_header(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect if text is a section header and return (section_type, section_title).

        Args:
            text: Text to analyze

        Returns:
            Tuple of (section_type, section_title) or (None, None)
        """
        text_clean = text.strip()

        # Skip very short lines
        if len(text_clean) < 3:
            return None, None

        # Check against section patterns
        for pattern, section_type in self.section_patterns.items():
            match = re.match(pattern, text_clean, re.IGNORECASE)
            if match:
                if section_type == 'primary_section' or section_type == 'numbered_section':
                    section_title = match.group(1).strip()
                else:
                    section_title = text_clean
                return section_type, section_title

        # Check for all-caps headers (common in pharmaceutical docs)
        if (text_clean.isupper() and
            len(text_clean) > 5 and
            len(text_clean) < 100 and
            not re.search(r'\d{2,}', text_clean)):  # Not just numbers
            return 'caps_header', text_clean

        return None, None

    def is_likely_abbreviation(self, text: str) -> bool:
        """
        Check if text ending with period is likely an abbreviation.

        Args:
            text: Text to check

        Returns:
            True if likely an abbreviation
        """
        if not text or len(text) < 2:
            return False

        word = text.rstrip('.').lower()

        if word in self.pharma_abbreviations:
            return True

        patterns = [
            r'^[a-z]{1,4}$',  # Short lowercase words
            r'^[A-Z]{2,6}$',  # All caps short words
            r'^[A-Z][a-z]{1,3}$',  # Capitalized short words
            r'^\d+[a-z]+$',  # Numbers with letters
            r'^[a-z]\.[a-z]',  # Pattern like q.s.p
            r'[0-9]$'  # Ends with number
        ]

        for pattern in patterns:
            if re.match(pattern, word):
                return True

        return False

    def smart_sentence_split_with_sections(self, text: str) -> List[Dict]:
        """
        Split text into sentences with section awareness.

        Args:
            text: Text to split

        Returns:
            List of dicts with sentence and section info
        """
        print("Splitting text with section tracking...")

        # First split by lines to identify headers
        lines = text.split('\n')

        current_section_type = 'unknown'
        current_section_title = 'Document Start'
        sentence_data = []
        current_sentence = ""

        for line_num, line in enumerate(lines):
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            # Check if this line is a section header
            section_type, section_title = self.detect_section_header(line)

            if section_type and section_title:
                # This is a header - finish current sentence if any
                if current_sentence.strip():
                    sentences = self._split_sentence_safely(current_sentence)
                    for sent in sentences:
                        if sent.strip() and len(sent.strip()) > 10:
                            sentence_data.append({
                                'sentence': sent.strip(),
                                'section_type': current_section_type,
                                'section_title': current_section_title,
                                'line_number': line_num,
                                'is_header': False
                            })
                    current_sentence = ""

                # Update current section
                current_section_type = section_type
                current_section_title = section_title

                # Add header as special sentence
                sentence_data.append({
                    'sentence': line,
                    'section_type': section_type,
                    'section_title': section_title,
                    'line_number': line_num,
                    'is_header': True
                })

                print(f"Section detected: {section_type} - {section_title}")

            else:
                # Regular content line - add to current sentence
                if current_sentence:
                    current_sentence += " " + line
                else:
                    current_sentence = line

        # Process any remaining sentence
        if current_sentence.strip():
            sentences = self._split_sentence_safely(current_sentence)
            for sent in sentences:
                if sent.strip() and len(sent.strip()) > 10:
                    sentence_data.append({
                        'sentence': sent.strip(),
                        'section_type': current_section_type,
                        'section_title': current_section_title,
                        'line_number': len(lines),
                        'is_header': False
                    })

        # Filter out headers from regular processing
        content_sentences = [s for s in sentence_data if not s['is_header']]

        print(f"Found {len(sentence_data)} total items ({len(content_sentences)} content sentences)")
        print(f"Sections identified: {len(set(s['section_title'] for s in sentence_data))}")

        return content_sentences

    def _split_sentence_safely(self, text: str) -> List[str]:
        """
        Split text into sentences with abbreviation awareness.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        sentences = []
        current_sentence = ""

        # Split by potential sentence endings
        parts = re.split(r'([.!?]+)', text)

        i = 0
        while i < len(parts):
            if i % 2 == 0:  # Text part
                current_sentence += parts[i]
            else:  # Punctuation part
                punctuation = parts[i]
                current_sentence += punctuation

                if '.' in punctuation:
                    words = current_sentence.split()
                    if words:
                        last_word = words[-1]
                        if not self.is_likely_abbreviation(last_word):
                            if current_sentence.strip():
                                sentences.append(current_sentence.strip())
                            current_sentence = ""
                    else:
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = ""
                else:
                    # ! or ? - definitely sentence endings
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
            i += 1

        # Add any remaining sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return [s for s in sentences if s.strip() and len(s.strip()) > 10]

    def create_section_aware_prompt(self, sentence_data: Dict) -> str:
        """
        Create a prompt that includes section context.

        Args:
            sentence_data: Dictionary with sentence and metadata

        Returns:
            Formatted prompt for LLM
        """
        sentence = sentence_data['sentence']
        section_type = sentence_data['section_type']
        section_title = sentence_data['section_title']

        prompt = f"""Analyze this sentence from a Brazilian pharmaceutical document. The sentence comes from the "{section_title}" section.

RESPOND ONLY WITH VALID JSON. No explanations, no markdown.

Context: This sentence is from the {section_type} section titled "{section_title}".

Extract relevant pharmaceutical information considering the section context.

Format:
{{
  "entities": [
    {{"type": "medication_name", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "dosage", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "indication", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "contraindication", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "side_effect", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "manufacturer", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "storage", "value": "...", "confidence": "high|medium|low"}},
    {{"type": "administration", "value": "...", "confidence": "high|medium|low"}}
  ],
  "section_relevance": "high|medium|low",
  "key_info_found": true/false
}}

Sentence: "{sentence}"

JSON:"""

        return prompt

    def parse_json_response(self, response: str) -> Any:
        """
        Enhanced JSON parsing with better error handling.

        Args:
            response: JSON string from LLM

        Returns:
            Parsed JSON object or None
        """
        if not response or not response.strip():
            return None

        cleaned = response.strip()

        # Remove markdown code blocks
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.rfind("```")
            if start < end:
                cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.rfind("```")
            if start < end:
                cleaned = cleaned[start:end].strip()

        # Find JSON boundaries
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            cleaned = cleaned[json_start:json_end]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            try:
                # Fix common issues
                fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                fixed = re.sub(r'(?<!\\)"(?=[^,}\]]*[,}\]])', r'\\"', fixed)
                return json.loads(fixed)
            except:
                print(f"JSON parse error: {cleaned[:100]}...")
                return None

    def analyze_sentence_with_context(self, sentence_data: Dict, index: int) -> Dict:
        """
        Analyze a sentence with section context.

        Args:
            sentence_data: Dictionary with sentence and metadata
            index: Sentence index

        Returns:
            Analysis result dictionary
        """
        try:
            prompt = self.create_section_aware_prompt(sentence_data)
            response = self.call_ollama_raw(prompt)

            result = self.parse_json_response(response)

            if result and isinstance(result, dict):
                # Add metadata with null checks
                result['sentence_index'] = index
                result['original_sentence'] = sentence_data.get('sentence', '')
                result['section_type'] = sentence_data.get('section_type', 'unknown')
                result['section_title'] = sentence_data.get('section_title', 'Unknown Section')
                result['line_number'] = sentence_data.get('line_number', 0)
                return result
            else:
                return self._create_empty_result(sentence_data, index, 'parsing_error')

        except Exception as e:
            print(f"Error analyzing sentence {index}: {e}")
            return self._create_empty_result(sentence_data, index, f'analysis_error: {e}')

    def _create_empty_result(self, sentence_data: Dict, index: int, error_type: str = None) -> Dict:
        """Create empty result structure with safe defaults."""
        return {
            'entities': [],
            'section_relevance': 'low',
            'key_info_found': False,
            'sentence_index': index,
            'original_sentence': sentence_data.get('sentence', ''),
            'section_type': sentence_data.get('section_type', 'unknown'),
            'section_title': sentence_data.get('section_title', 'Unknown Section'),
            'line_number': sentence_data.get('line_number', 0),
            'error': error_type
        }

    def process_sentences_with_context(self, sentence_data_list: List[Dict]) -> List[Dict]:
        """
        Process all sentences with section context.

        Args:
            sentence_data_list: List of sentence dictionaries

        Returns:
            List of analysis results
        """
        print(f"Analyzing {len(sentence_data_list)} sentences with section context...")

        analyses = []
        successful_analyses = 0

        for i, sentence_data in enumerate(sentence_data_list):
            sentence = sentence_data.get('sentence', '')[:60]
            section = sentence_data.get('section_title', 'Unknown')

            print(f"Processing {i+1}/{len(sentence_data_list)} in [{section}]: {sentence}...")

            analysis = self.analyze_sentence_with_context(sentence_data, i)
            analyses.append(analysis)

            if analysis.get('key_info_found') and not analysis.get('error'):
                successful_analyses += 1
                entity_count = len(analysis.get('entities', []))
                if entity_count > 0:
                    print(f"  Found {entity_count} entities")

            import time
            time.sleep(0.1)

        print(f"Completed analysis: {successful_analyses}/{len(sentence_data_list)} sentences with entities")
        return analyses

    def aggregate_entities_by_section(self, analyses: List[Dict]) -> Dict:
        """
        Aggregate entities by section with null-safe processing.

        Args:
            analyses: List of analysis results

        Returns:
            Dictionary with aggregated entities by section
        """
        print("Aggregating entities by section...")

        section_entities = {}
        all_entities = []
        section_stats = {}

        for analysis in analyses:
            if not analysis or analysis.get('error'):
                continue

            section_title = analysis.get('section_title', 'Unknown Section')

            # Initialize section if not exists
            if section_title not in section_entities:
                section_entities[section_title] = {
                    'medication_names': set(), 'dosages': set(), 'indications': set(),
                    'contraindications': set(), 'side_effects': set(), 'manufacturers': set(),
                    'storage_conditions': set(), 'administration_info': set()
                }
                section_stats[section_title] = {'sentences': 0, 'entities': 0}

            section_stats[section_title]['sentences'] += 1

            entities = analysis.get('entities', [])
            if not entities:
                continue

            section_stats[section_title]['entities'] += len(entities)

            for entity in entities:
                # Check if the entity is a valid dictionary
                if not isinstance(entity, dict):
                    continue

                # Get the type and value, which could be None
                entity_type = entity.get('type')
                entity_value = entity.get('value')

                # Ensure both type and value are not None or empty before stripping
                if not entity_type or not entity_value:
                    continue

                entity_type = entity_type.strip()
                entity_value = str(entity_value).strip()  # Convert to string to be safe

                type_mapping = {
                    'medication_name': 'medication_names', 'dosage': 'dosages',
                    'indication': 'indications', 'contraindication': 'contraindications',
                    'side_effect': 'side_effects', 'manufacturer': 'manufacturers',
                    'storage': 'storage_conditions', 'administration': 'administration_info'
                }

                if entity_type in type_mapping:
                    collection_key = type_mapping[entity_type]
                    section_entities[section_title][collection_key].add(entity_value)

                    all_entities.append({
                        'type': entity_type, 'value': entity_value,
                        'section': section_title,
                        'confidence': entity.get('confidence', 'medium'),
                        'sentence_index': analysis.get('sentence_index', -1)
                    })

        # Convert sets to lists for JSON serialization
        for section in section_entities:
            for entity_type in section_entities[section]:
                section_entities[section][entity_type] = list(section_entities[section][entity_type])

        result = {
            'entities_by_section': section_entities, 'all_entities': all_entities,
            'section_statistics': section_stats, 'total_entities': len(all_entities),
            'sections_processed': len(section_entities)
        }

        print(f"Aggregated {len(all_entities)} entities across {len(section_entities)} sections")
        return result

    def process_document(self, pdf_path: str) -> bool:
        """
        Main document processing with section awareness.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if processing succeeded
        """
        print(f"Processing document with section-aware analysis: {pdf_path}")

        if not Path(pdf_path).exists():
            print(f"File not found: {pdf_path}")
            return False

        try:
            # Extract text
            print("Extracting text from PDF...")
            self.raw_content = self.extract_pdf_content(pdf_path)

            if not self.raw_content:
                print("No text content extracted")
                return False

            print(f"Extracted {len(self.raw_content)} characters")

            # Split with section awareness
            sentence_data_list = self.smart_sentence_split_with_sections(self.raw_content)

            # Process sentences with context
            analyses = self.process_sentences_with_context(sentence_data_list)

            # Aggregate by sections
            aggregated_data = self.aggregate_entities_by_section(analyses)

            # Compile results
            self.structured_data = {
                "metadata": {
                    "file_path": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "processing_date": datetime.now().isoformat(),
                    "total_text_length": len(self.raw_content),
                    "model_used": self.model_name,
                    "extraction_method": "section_aware_sentence_analysis"
                },
                "sentence_analyses": analyses,
                "section_entities": aggregated_data,
                "processing_statistics": {
                    "total_sentences": len(sentence_data_list),
                    "sentences_with_entities": len([a for a in analyses if a.get('key_info_found')]),
                    "total_entities_found": aggregated_data.get('total_entities', 0),
                    "sections_identified": aggregated_data.get('sections_processed', 0)
                }
            }

            self.document_loaded = True
            print("Section-aware document processing completed!")
            self._show_processing_summary()
            return True

        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _show_processing_summary(self):
        """Show processing summary with section information."""
        print("\n" + "=" * 70)
        print("SECTION-AWARE PROCESSING SUMMARY")
        print("=" * 70)

        metadata = self.structured_data.get("metadata", {})
        stats = self.structured_data.get("processing_statistics", {})
        section_data = self.structured_data.get("section_entities", {})

        print(f"File: {metadata.get('file_name', 'Unknown')}")
        print(f"Text length: {metadata.get('total_text_length', 0):,} characters")
        print(f"Total sentences: {stats.get('total_sentences', 0)}")
        print(f"Sentences with entities: {stats.get('sentences_with_entities', 0)}")
        print(f"Total entities found: {stats.get('total_entities_found', 0)}")
        print(f"Sections identified: {stats.get('sections_identified', 0)}")

        # Show section statistics
        section_stats = section_data.get('section_statistics', {})
        if section_stats:
            print(f"\nEntity Distribution by Section:")
            for section, stat in section_stats.items():
                print(f"   {section}: {stat['entities']} entities from {stat['sentences']} sentences")

        print("=" * 70)

    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save processing results.

        Args:
            output_path: Optional path for output file

        Returns:
            Path to saved file
        """
        if not self.document_loaded:
            raise Exception("No document processed")

        if not output_path:
            file_name = self.structured_data["metadata"]["file_name"]
            pdf_name = Path(file_name).stem
            output_path = f"{pdf_name}_section_aware_analysis.json"

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.structured_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size:,} bytes")
        return str(output_file)

    def query_document(self, question: str) -> str:
        """
        Query with section context.

        Args:
            question: Question to ask about the document

        Returns:
            Answer from LLM
        """
        if not self.document_loaded:
            return "No document processed."

        section_entities = self.structured_data.get("section_entities", {}).get("entities_by_section", {})

        context_parts = ["MEDICATION INFORMATION BY SECTION:\n"]

        for section, entities in section_entities.items():
            if any(entities.values()):  # Only show sections with entities
                context_parts.append(f"[{section}]")
                for entity_type, values in entities.items():
                    if values:
                        context_parts.append(f"  {entity_type}: {', '.join(values[:3])}")
                context_parts.append("")

        context = "\n".join(context_parts)

        query_prompt = f"""Answer about this medication based on the section-organized information.

Question: {question}

Available Information:
{context[:4000]}

Provide a clear answer in Portuguese, mentioning the relevant sections when appropriate."""

        try:
            response = self.call_ollama_raw(query_prompt)
            return response.strip() if response else "No response received"
        except Exception as e:
            return f"Error: {e}"
