#!/usr/bin/env python3
"""
Automated Pharmaceutical Document Parser with Ollama Integration

This module provides an automated pipeline for processing pharmaceutical documents
and extracting structured information using a local LLM (Ollama).

The AutomatedPharmaParser orchestrates the complete extraction workflow:
1. PDF text extraction
2. Document structure analysis
3. Entity extraction as knowledge triples
4. Comprehensive summary generation
5. Interactive querying

Knowledge Triples:
    A knowledge triple is a structured representation of a fact in the form:
    {"entity": "...", "relation": "...", "value": "..."}

    Examples:
    - {"entity": "Paracetamol", "relation": "has_dosage", "value": "500mg"}
    - {"entity": "medication", "relation": "is_indicated_for", "value": "headache"}
    - {"entity": "substance", "relation": "can_cause", "value": "nausea"}

Automated Pipeline:
    The AutomatedPharmaParser provides a complete extraction pipeline:
    - Automatic model setup and validation
    - Multi-strategy PDF text extraction
    - Chunked processing for large documents
    - LLM-based entity extraction and structure analysis
    - Document querying capabilities
    - Comprehensive result reporting
"""

import json
import re
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pymupdf4llm
import pdfplumber


class AutomatedPharmaParser:
    """
    Automated pharmaceutical document parser with Ollama integration.

    This parser orchestrates the complete extraction workflow from PDF to
    structured knowledge graph using a local LLM.

    It provides:
    - Automatic Ollama model setup
    - PDF text extraction with fallback strategies
    - LLM-based entity extraction with chunking
    - Document structure analysis
    - Summary generation
    - Interactive document querying

    Args:
        model_name: Name of the Ollama model to use (default: "llama3.2:3b")
    """

    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize automated parser with Ollama integration."""
        self.model_name = model_name
        self.raw_content = ""
        self.structured_data = {}
        self.document_loaded = False
        self.setup_ollama()

    def setup_ollama(self) -> None:
        """Setup Ollama model automatically."""
        print(f"Setting up Ollama model: {self.model_name}")

        try:
            # Check if ollama is available
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            print("Ollama CLI found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("Ollama CLI not found. Please install Ollama first.")

        try:
            # Pull model if not available
            print(f"Pulling model {self.model_name}...")
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"Model {self.model_name} ready")
            else:
                print(f"Pull result: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Model pull timed out, but model might already be available")
        except Exception as e:
            print(f"Error pulling model: {e}")

    def call_ollama_raw(self, prompt: str, extra_flags: str = "") -> str:
        """
        Call ollama with exact prompt - no modifications.

        Args:
            prompt: The prompt to send to Ollama
            extra_flags: Additional command-line flags for ollama

        Returns:
            The model's response text

        Raises:
            RuntimeError: If the Ollama call fails or times out
        """
        cmd = ["ollama", "run", self.model_name]
        if extra_flags:
            cmd += shlex.split(extra_flags)

        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=120
            )
            output = proc.stdout.strip()
            if not output:
                output = proc.stderr.strip()
            return output
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama call timed out")
        except Exception as e:
            raise RuntimeError(f"Error calling ollama: {e}")

    def extract_pdf_content(self, pdf_path: str) -> str:
        """
        Extract text content from PDF.

        Tries pdfplumber first, then falls back to pymupdf4llm.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            Exception: If all extraction methods fail
        """
        try:
            # Try pdfplumber first
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
            # Fallback to pymupdf4llm
            return pymupdf4llm.to_markdown(pdf_path)
        except Exception as e:
            print(f"pymupdf4llm failed: {e}")
            raise Exception("All extraction methods failed")

    def create_entity_extraction_prompt(self, text: str, chunk_size: int = 3000) -> List[str]:
        """
        Create prompts for entity extraction from pharmaceutical text.

        Splits large text into chunks and creates a prompt for each chunk.

        Args:
            text: The text to extract entities from
            chunk_size: Maximum size of each text chunk

        Returns:
            List of prompts, one for each chunk
        """
        # Base prompt for pharmaceutical entity extraction
        base_prompt = """System: You are a parser. For each Text below, extract entities, relation, value triples as a JSON array.
Only output valid JSON. DO NOT include any extra text, commentary, or code fences. Output must be parseable by json.loads().

Format:
[
  {"entity": "...", "relation": "...", "value": "..."},
  ...
]

Focus on pharmaceutical information:
- Medication names and active ingredients
- Dosages, concentrations, and administration routes
- Indications, contraindications, and side effects
- Age groups, patient populations
- Storage conditions and expiration
- Manufacturer information

Text: """

        # Split text into chunks if too long
        text_chunks = []
        if len(text) <= chunk_size:
            text_chunks.append(text)
        else:
            words = text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 > chunk_size:
                    if current_chunk:
                        text_chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                current_chunk.append(word)
                current_length += len(word) + 1

            if current_chunk:
                text_chunks.append(" ".join(current_chunk))

        # Create prompts for each chunk
        prompts = []
        for i, chunk in enumerate(text_chunks):
            prompt = f"{base_prompt}{chunk}"
            prompts.append(prompt)

        return prompts

    def create_structure_analysis_prompt(self, text: str) -> str:
        """
        Create prompt for document structure analysis.

        Args:
            text: The document text to analyze

        Returns:
            The formatted prompt for structure analysis
        """
        structure_prompt = f"""System: You are a pharmaceutical document analyzer. Analyze the document structure and create a JSON summary.
Only output valid JSON. DO NOT include any extra text, commentary, or code fences.

Format:
{{
  "document_type": "...",
  "main_sections": [
    {{
      "section_number": "...",
      "section_title": "...",
      "content_type": "...",
      "key_points": ["...", "..."]
    }}
  ],
  "medication_info": {{
    "name": "...",
    "active_ingredient": "...",
    "forms": ["...", "..."],
    "concentrations": ["...", "..."]
  }},
  "critical_information": {{
    "contraindications": ["...", "..."],
    "serious_warnings": ["...", "..."],
    "storage_conditions": "..."
  }}
}}

Text: {text[:4000]}"""

        return structure_prompt

    def parse_json_response(self, response: str) -> Any:
        """
        Parse JSON response, handling common formatting issues.

        Args:
            response: Raw response text from LLM

        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Clean up response
        cleaned = response.strip()

        # Remove code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Remove any leading/trailing text that's not JSON
        start_idx = cleaned.find('[') if cleaned.find('[') != -1 else cleaned.find('{')
        end_idx = cleaned.rfind(']') if cleaned.rfind(']') != -1 else cleaned.rfind('}')

        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx+1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Problematic text: {cleaned[:200]}...")
            return None

    def extract_entities_from_document(self, text: str) -> List[Dict]:
        """
        Extract entities from entire document.

        Processes the document in chunks and aggregates results.

        Args:
            text: The document text

        Returns:
            List of unique entity dictionaries
        """
        print("Extracting entities using LLM...")

        prompts = self.create_entity_extraction_prompt(text)
        all_entities = []

        for i, prompt in enumerate(prompts):
            print(f"Processing chunk {i+1}/{len(prompts)}...")

            try:
                response = self.call_ollama_raw(prompt)
                entities = self.parse_json_response(response)

                if entities and isinstance(entities, list):
                    all_entities.extend(entities)
                    print(f"  Extracted {len(entities)} entities from chunk {i+1}")
                else:
                    print(f"  No valid entities from chunk {i+1}")

            except Exception as e:
                print(f"  Error processing chunk {i+1}: {e}")
                continue

        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in all_entities:
            key = (entity.get('entity', ''), entity.get('relation', ''), entity.get('value', ''))
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        print(f"Total unique entities extracted: {len(unique_entities)}")
        return unique_entities

    def analyze_document_structure(self, text: str) -> Dict:
        """
        Analyze document structure using LLM.

        Args:
            text: The document text

        Returns:
            Dictionary with document structure analysis
        """
        print("Analyzing document structure using LLM...")

        prompt = self.create_structure_analysis_prompt(text)

        try:
            response = self.call_ollama_raw(prompt)
            structure = self.parse_json_response(response)

            if structure and isinstance(structure, dict):
                print("Document structure analyzed successfully")
                return structure
            else:
                print("Could not parse structure analysis")
                return {}

        except Exception as e:
            print(f"Error analyzing structure: {e}")
            return {}

    def create_summary_prompt(self, entities: List[Dict], structure: Dict) -> str:
        """
        Create prompt for generating document summary.

        Args:
            entities: List of extracted entities
            structure: Document structure analysis

        Returns:
            The formatted prompt for summary generation
        """
        entities_text = json.dumps(entities[:50], indent=2)  # Limit to first 50 entities
        structure_text = json.dumps(structure, indent=2)

        summary_prompt = f"""System: You are a pharmaceutical document summarizer. Based on the extracted entities and document structure, create a comprehensive summary.
Only output valid JSON. DO NOT include any extra text, commentary, or code fences.

Format:
{{
  "executive_summary": "...",
  "medication_details": {{
    "name": "...",
    "active_ingredients": ["...", "..."],
    "therapeutic_class": "...",
    "indications": ["...", "..."],
    "dosage_forms": ["...", "..."],
    "key_dosages": ["...", "..."]
  }},
  "safety_information": {{
    "contraindications": ["...", "..."],
    "warnings": ["...", "..."],
    "common_side_effects": ["...", "..."],
    "serious_reactions": ["...", "..."]
  }},
  "administration_info": {{
    "routes": ["...", "..."],
    "dosing_schedule": "...",
    "special_populations": {{
      "pediatric": "...",
      "geriatric": "...",
      "renal_impairment": "...",
      "hepatic_impairment": "..."
    }}
  }},
  "storage_and_handling": "...",
  "manufacturer": "..."
}}

Extracted Entities:
{entities_text}

Document Structure:
{structure_text}"""

        return summary_prompt

    def generate_comprehensive_summary(self, entities: List[Dict], structure: Dict) -> Dict:
        """
        Generate comprehensive summary using LLM.

        Args:
            entities: List of extracted entities
            structure: Document structure analysis

        Returns:
            Dictionary with comprehensive summary
        """
        print("Generating comprehensive summary...")

        prompt = self.create_summary_prompt(entities, structure)

        try:
            response = self.call_ollama_raw(prompt, extra_flags="--temperature 0.1")
            summary = self.parse_json_response(response)

            if summary and isinstance(summary, dict):
                print("Summary generated successfully")
                return summary
            else:
                print("Could not parse summary")
                return {}

        except Exception as e:
            print(f"Error generating summary: {e}")
            return {}

    def process_document(self, pdf_path: str) -> bool:
        """
        Fully automated document processing.

        Args:
            pdf_path: Path to the PDF file to process

        Returns:
            True if processing succeeded, False otherwise
        """
        print(f"Processing document: {pdf_path}")

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

            # Analyze structure
            structure = self.analyze_document_structure(self.raw_content)

            # Extract entities
            entities = self.extract_entities_from_document(self.raw_content)

            # Generate summary
            summary = self.generate_comprehensive_summary(entities, structure)

            # Compile final structure
            self.structured_data = {
                "metadata": {
                    "file_path": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "processing_date": datetime.now().isoformat(),
                    "total_text_length": len(self.raw_content),
                    "total_entities": len(entities),
                    "model_used": self.model_name
                },
                "document_structure": structure,
                "extracted_entities": entities,
                "comprehensive_summary": summary,
                "processing_statistics": {
                    "entities_by_type": self._count_entities_by_type(entities),
                    "structure_sections": len(structure.get('main_sections', [])),
                    "processing_method": "automated_llm_analysis"
                }
            }

            self.document_loaded = True
            print("Document processing completed!")
            self._show_processing_summary()
            return True

        except Exception as e:
            print(f"Error processing document: {e}")
            return False

    def _count_entities_by_type(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by relation type."""
        counts = {}
        for entity in entities:
            relation = entity.get('relation', 'unknown')
            counts[relation] = counts.get(relation, 0) + 1
        return counts

    def _show_processing_summary(self) -> None:
        """Show processing summary."""
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)

        metadata = self.structured_data.get("metadata", {})
        stats = self.structured_data.get("processing_statistics", {})

        print(f"File: {metadata.get('file_name', 'Unknown')}")
        print(f"Model: {metadata.get('model_used', 'Unknown')}")
        print(f"Text length: {metadata.get('total_text_length', 0):,} characters")
        print(f"Total entities: {metadata.get('total_entities', 0)}")
        print(f"Structure sections: {stats.get('structure_sections', 0)}")

        print("\nEntity distribution:")
        for entity_type, count in stats.get('entities_by_type', {}).items():
            print(f"   {entity_type}: {count}")

        # Show sample entities
        entities = self.structured_data.get("extracted_entities", [])
        if entities:
            print("\nSample entities:")
            for i, entity in enumerate(entities[:5]):
                print(f"   {i+1}. {entity.get('entity', 'N/A')} -> {entity.get('relation', 'N/A')} -> {entity.get('value', 'N/A')}")

        print("=" * 70)

    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save processing results.

        Args:
            output_path: Optional path to save results. If not provided,
                        uses the PDF name with '_automated_analysis.json' suffix

        Returns:
            Path to the saved file

        Raises:
            Exception: If no document has been processed
        """
        if not self.document_loaded:
            raise Exception("No document processed")

        if not output_path:
            file_name = self.structured_data["metadata"]["file_name"]
            pdf_name = Path(file_name).stem
            output_path = f"{pdf_name}_automated_analysis.json"

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.structured_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size:,} bytes")
        return str(output_file)

    def query_document(self, question: str) -> str:
        """
        Query the processed document.

        Args:
            question: Question to ask about the document

        Returns:
            Answer based on the processed document data
        """
        if not self.document_loaded:
            return "No document processed. Please process a document first."

        # Create context from processed data
        context_parts = []

        # Add summary
        summary = self.structured_data.get("comprehensive_summary", {})
        if summary:
            context_parts.append("DOCUMENT SUMMARY:")
            context_parts.append(json.dumps(summary, indent=2))

        # Add relevant entities (simple keyword matching)
        entities = self.structured_data.get("extracted_entities", [])
        question_words = question.lower().split()
        relevant_entities = []

        for entity in entities:
            entity_text = f"{entity.get('entity', '')} {entity.get('relation', '')} {entity.get('value', '')}".lower()
            if any(word in entity_text for word in question_words):
                relevant_entities.append(entity)

        if relevant_entities:
            context_parts.append("\nRELEVANT ENTITIES:")
            context_parts.append(json.dumps(relevant_entities[:10], indent=2))

        context = "\n".join(context_parts)

        # Create query prompt
        query_prompt = f"""System: You are a pharmaceutical document assistant. Answer the question based on the provided document context.
Be precise and cite specific information when possible.

Question: {question}

Document Context:
{context[:6000]}

Answer:"""

        try:
            response = self.call_ollama_raw(query_prompt, extra_flags="--temperature 0.1")
            return response.strip()
        except Exception as e:
            return f"Error processing query: {e}"
