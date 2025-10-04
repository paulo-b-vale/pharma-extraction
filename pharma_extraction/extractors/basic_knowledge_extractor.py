#!/usr/bin/env python3
"""
Basic Pharmaceutical Knowledge Graph Extractor

This module provides basic knowledge triple extraction from pharmaceutical documents.
It processes flat blocks from parsed documents and extracts knowledge triples using
an LLM (Ollama).

Knowledge Triples:
    A knowledge triple is a structured representation of a fact in the form:
    [entity, relation, value]

    Examples:
    - ["Amoxicilina", "has_dosage", "500mg 3x ao dia"]
    - ["Amoxicilina", "treats", "infecções respiratórias"]
    - ["medication", "is_contraindicated_in", "pregnancy"]

    This format allows pharmaceutical information to be structured as a knowledge graph
    for downstream processing, querying, and analysis.

Basic Extractor:
    The PharmaceuticalKnowledgeExtractor provides basic extraction capabilities:
    - Block-by-block processing of flat document representations
    - Section-aware prompting based on document breadcrumbs
    - Pharmaceutical content filtering
    - JSON-based triple parsing with fallback strategies
"""

import json
import re
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class PharmaceuticalKnowledgeExtractor:
    """
    Basic knowledge extractor optimized for block-based pharmaceutical documents.

    This extractor processes pharmaceutical documents that have been parsed into
    flat blocks, extracting knowledge triples using an LLM (Ollama).

    It provides:
    - Block-by-block extraction
    - Section-aware prompting
    - Pharmaceutical keyword filtering
    - JSON parsing with regex fallback
    - Statistics tracking

    Args:
        model_name: Name of the Ollama model to use (default: "llama3.2:3b")
        ollama_url: URL of the Ollama API endpoint
        max_retries: Maximum number of API call retries
        request_delay: Delay between API requests in seconds
    """

    def __init__(self,
                 model_name: str = "llama3.2:3b",
                 ollama_url: str = "http://localhost:11434/api/generate",
                 max_retries: int = 3,
                 request_delay: float = 0.5):
        """Initialize the knowledge extractor with Llama 3.2 3B optimizations."""
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.request_delay = request_delay

        self.stats = {
            'files_processed': 0,
            'blocks_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_triples': 0
        }

        self._setup_logging()
        self._test_ollama_connection()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Remove any existing handlers to avoid duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pharma_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama API."""
        try:
            test_payload = {
                "model": self.model_name,
                "prompt": "Test connection",
                "stream": False,
                "format": "json"
            }
            response = requests.post(self.ollama_url, json=test_payload, timeout=15)
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to Ollama with {self.model_name}")
            else:
                self.logger.warning(f"Ollama connection test failed: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError("Cannot connect to Ollama API. Ensure it's running and accessible.")

    def _call_ollama_api(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Call Ollama API with retry logic and error handling.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate

        Returns:
            The LLM response text or None if all retries fail
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": max_tokens,
                "stop": ["\n\n", "---"],
            }
        }

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                response = requests.post(self.ollama_url, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    self.logger.warning(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
            except Exception as e:
                self.logger.warning(f"Request error on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) + self.request_delay
                time.sleep(wait_time)

        self.logger.error("All API call attempts failed")
        return None

    def _create_focused_extraction_prompt(self, content: str, context: Dict) -> str:
        """
        Create a focused prompt based on section keywords.

        Args:
            content: Text content to extract from
            context: Document context with breadcrumb information

        Returns:
            The formatted prompt for the LLM
        """
        content = content[:800].strip() + "..." if len(content) > 800 else content.strip()
        section = context.get('breadcrumb', 'Unknown Section')

        focus_map = {
            "composition": ['composição', 'composition'],
            "dosage": ['dosagem', 'posologia', 'como usar'],
            "indication": ['indicação', 'indication', 'para que'],
            "contraindication": ['contraindicação', 'contraindication', 'não devo usar'],
            "side_effect": ['efeito', 'reação', 'adverse', 'males'],
            "interaction": ['interação', 'interaction']
        }

        focus_type = "general"
        for f_type, keywords in focus_map.items():
            if any(kw in section.lower() for kw in keywords):
                focus_type = f_type
                break

        return f"""Extract pharmaceutical facts as JSON triples from this text.

Section: {section}
Focus: {focus_type}

Text: "{content}"

Extract ONLY factual triples in format [entity, relation, value]. Return a valid JSON array. No explanations.

Example:
[["Amoxicilina", "has_dosage", "500mg 3x ao dia"], ["Amoxicilina", "treats", "infecções respiratórias"]]

JSON:"""

    def _create_comprehensive_prompt(self, content: str, context: Dict) -> str:
        """
        Create a comprehensive prompt for important sections.

        Args:
            content: Text content to extract from
            context: Document context with breadcrumb information

        Returns:
            The formatted prompt for the LLM
        """
        content = content[:600].strip() + "..." if len(content) > 600 else content.strip()
        section = context.get('breadcrumb', 'Unknown Section')

        return f"""Extract all pharmaceutical information from this text as knowledge triples.

Section: {section}
Content: "{content}"

Extract triples for medication names, dosages, conditions (indications/contraindications), side effects, and interactions.
Format: [["entity", "relation", "value"], ...]
Return only a valid JSON array:"""

    def _parse_triples_response(self, response: str) -> List[List[str]]:
        """
        Parse the API response to robustly extract a list of triples.

        Uses JSON parsing with regex fallback for resilience.

        Args:
            response: Raw LLM response text

        Returns:
            List of validated triples
        """
        if not response:
            return []

        cleaned = re.sub(r'```json\s*|```\s*', '', response.strip())

        try:
            start_idx = cleaned.find('[')
            end_idx = cleaned.rfind(']')
            if start_idx == -1 or end_idx == -1:
                return []

            json_str = cleaned[start_idx:end_idx + 1]
            parsed = json.loads(json_str)

            valid_triples = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, list) and len(item) == 3:
                        entity, relation, value = [str(x).strip() for x in item]
                        if all([entity, relation, value]):
                            valid_triples.append([entity, relation, value])
            return valid_triples

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}. Falling back to regex.")
            pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
            matches = re.findall(pattern, cleaned)
            return [[m[0], m[1], m[2]] for m in matches]

    def _should_process_block(self, block: Dict) -> bool:
        """
        Determine if a block has relevant content worth processing.

        Args:
            block: Block dictionary with content

        Returns:
            True if block should be processed
        """
        content = block.get('content', '').strip()
        if len(content) < 25:
            return False

        pharma_keywords = [
            'mg', 'ml', 'dose', 'comprimido', 'cápsula', 'medicamento', 'indicação',
            'contraindicação', 'efeito', 'reação', 'alergia', 'administração',
            'posologia', 'composição', 'princípio ativo'
        ]

        return any(kw in content.lower() for kw in pharma_keywords) or len(content) > 150

    def _extract_block_knowledge(self, block: Dict, block_index: int) -> Dict:
        """
        Extract knowledge from a single block.

        Args:
            block: Block dictionary with content, context, and metadata
            block_index: Index of the block in the document

        Returns:
            Dictionary with extraction results and status
        """
        block_id = block.get('id', f'block_{block_index}')
        if not self._should_process_block(block):
            return {
                'block_id': block_id,
                'triples': [],
                'status': 'skipped_irrelevant'
            }

        content = block.get('content', '').strip()
        context = block.get('context', {})
        self.logger.info(f"Processing block {block_id}: {content[:60]}...")

        try:
            if len(content) > 500:
                prompt = self._create_comprehensive_prompt(content, context)
            else:
                prompt = self._create_focused_extraction_prompt(content, context)

            response = self._call_ollama_api(prompt)

            if response:
                triples = self._parse_triples_response(response)
                self.stats['successful_extractions'] += 1
                self.stats['total_triples'] += len(triples)
                self.logger.info(f"Extracted {len(triples)} triples from {block_id}")
                return {
                    'block_id': block_id,
                    'block_type': block.get('type'),
                    'breadcrumb': context.get('breadcrumb'),
                    'triples': triples,
                    'status': 'success'
                }
            else:
                self.stats['failed_extractions'] += 1
                return {
                    'block_id': block_id,
                    'triples': [],
                    'status': 'api_failed'
                }
        except Exception as e:
            self.logger.error(f"Error in block {block_id}: {e}")
            self.stats['failed_extractions'] += 1
            return {
                'block_id': block_id,
                'triples': [],
                'status': f'error: {str(e)}'
            }
        finally:
            time.sleep(self.request_delay)

    def process_json_file(self, input_file: Path) -> Optional[Dict]:
        """
        Process a single JSON file.

        Args:
            input_file: Path to the LLM-optimized JSON file

        Returns:
            Dictionary with extraction results or None if processing fails
        """
        self.logger.info(f"Processing file: {input_file.name}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {input_file}: {e}")
            return None

        blocks = data.get('representations', {}).get('flat_blocks', [])
        if not blocks:
            self.logger.warning(f"No flat_blocks found in {input_file}")
            return None

        extractions = [self._extract_block_knowledge(block, i) for i, block in enumerate(blocks)]

        total_triples = sum(len(e['triples']) for e in extractions)
        result = {
            'document_metadata': data.get('document_metadata', {}),
            'extraction_summary': {
                'extraction_timestamp': datetime.now().isoformat(),
                'total_triples_extracted': total_triples
            },
            'graph_extractions': extractions
        }

        self.stats['files_processed'] += 1
        self.logger.info(f"Completed {input_file.name}: {total_triples} triples extracted.")
        return result

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Process all relevant JSON files in a directory.

        Args:
            input_dir: Directory containing LLM-optimized JSON files
            output_dir: Directory to save extraction results
        """
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(input_dir.glob('*_llm_optimized.json'))
        if not json_files:
            self.logger.warning(f"No *_llm_optimized.json files found in {input_dir}")
            return

        self.logger.info(f"Found {len(json_files)} files to process.")

        for i, json_file in enumerate(json_files):
            self.logger.info(f"\nProgress: Processing file {i + 1}/{len(json_files)}")
            result = self.process_json_file(json_file)

            if result:
                output_name = json_file.stem.replace('_llm_optimized', '_graph_data') + '.json'
                output_file = output_dir / output_name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved results to: {output_file}")

        self._generate_final_report(output_dir)

    def _generate_final_report(self, output_dir: Path) -> None:
        """Generate and save a final summary report."""
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'total_files_processed': self.stats['files_processed'],
            },
            'statistics': self.stats,
        }
        report_file = output_dir / 'final_extraction_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Final report saved: {report_file}")
