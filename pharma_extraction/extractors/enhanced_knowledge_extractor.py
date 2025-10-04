#!/usr/bin/env python3
"""
Enhanced Pharmaceutical Knowledge Graph Extractor

This module extracts pharmaceutical knowledge as triples (entity, relation, value) from
structured pharmaceutical documents. It's optimized for phrase-based JSON files and uses
LLM-based extraction to identify medical entities and their relationships.

Knowledge Triples:
    A knowledge triple is a structured representation of a fact in the form:
    [entity, relation, value]

    Examples:
    - ["Paracetamol", "has_dosage", "500mg 3x ao dia"]
    - ["medication", "is_indicated_for", "dor de cabeça"]
    - ["substance", "can_cause", "náusea"]

    This format allows pharmaceutical information to be structured as a knowledge graph
    for downstream processing, querying, and analysis.

Enhanced Extractor:
    The EnhancedPharmaceuticalKnowledgeExtractor provides advanced extraction capabilities:
    - Phrase-by-phrase processing with context awareness
    - Table data extraction with structured handling
    - Pharmaceutical keyword detection and relevance filtering
    - Multiple parsing strategies for robust triple extraction
    - Comprehensive logging and statistics tracking
"""

import json
import re
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class EnhancedPharmaceuticalKnowledgeExtractor:
    """
    Enhanced knowledge extractor optimized for phrase-based pharmaceutical documents.

    This extractor processes pharmaceutical documents that have been pre-parsed into
    phrase and table blocks, extracting knowledge triples using an LLM (Ollama).

    It provides:
    - Context-aware extraction based on document structure
    - Pharmaceutical-specific keyword filtering
    - Enhanced prompting for better extraction quality
    - Multiple parsing strategies for robustness
    - Detailed statistics and logging

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
        """Initialize the enhanced knowledge extractor."""
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.request_delay = request_delay

        self.stats = {
            'files_processed': 0,
            'phrase_blocks_processed': 0,
            'table_blocks_processed': 0,
            'phrases_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_triples': 0,
            'skipped_irrelevant': 0
        }

        # Enhanced patterns for better pharmaceutical content detection
        self.pharma_keywords = [
            'mg', 'ml', 'g/', 'mcg', 'μg', '%', 'dose', 'dosagem', 'posologia',
            'comprimido', 'cápsula', 'medicamento', 'fármaco', 'droga',
            'indicação', 'indicado', 'tratamento', 'terapia',
            'contraindicação', 'contraindicado', 'não usar', 'evitar',
            'efeito', 'reação', 'adverso', 'colateral', 'indesejável',
            'alergia', 'hipersensibilidade', 'intolerância',
            'administração', 'aplicar', 'tomar', 'ingerir',
            'composição', 'princípio ativo', 'substância', 'excipiente',
            'interação', 'interagir', 'incompatível', 'interferir',
            'gravidez', 'gestação', 'lactação', 'amamentação',
            'criança', 'pediátrico', 'adulto', 'idoso', 'geriátrico'
        ]

        self._setup_logging()
        self._test_ollama_connection()

    def _setup_logging(self) -> None:
        """Setup enhanced logging configuration."""
        # Remove existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # General logger for progress and errors
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_pharma_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Dedicated logger for prompts and responses
        self.prompt_logger = logging.getLogger('prompt_logger')
        self.prompt_logger.setLevel(logging.INFO)
        prompt_handler = logging.FileHandler('enhanced_prompts_and_responses.log', mode='w')
        prompt_formatter = logging.Formatter('%(message)s')
        prompt_handler.setFormatter(prompt_formatter)

        # Avoid adding handlers if they already exist
        if not self.prompt_logger.handlers:
            self.prompt_logger.addHandler(prompt_handler)

    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama API with enhanced error reporting."""
        try:
            test_payload = {
                "model": self.model_name,
                "prompt": "Teste de conexão. Responda apenas 'OK'.",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 10}
            }
            response = requests.post(self.ollama_url, json=test_payload, timeout=15)
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to Ollama with {self.model_name}")
                result = response.json()
                self.logger.debug(f"Test response: {result.get('response', 'No response')}")
            else:
                self.logger.warning(f"Ollama connection test failed: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to Ollama API. Ensure it's running and accessible.")
            raise ConnectionError("Cannot connect to Ollama API. Ensure it's running and accessible.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Ollama connection failed: {e}")

    def _call_ollama_api(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
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
                "temperature": 0.0,  # Deterministic for structured output
                "top_p": 0.9,
                "top_k": 20,
                "num_predict": max_tokens,
                "stop": ["\n\n", "---", "Exemplos:", "Examples:", "Nota:", "Note:"],
                "repeat_penalty": 1.1,
                "num_ctx": 2048,  # Context window
            }
        }

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                response = requests.post(self.ollama_url, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    if response_text:
                        return response_text
                    else:
                        self.logger.warning("Empty response from API")
                else:
                    self.logger.warning(f"API error {response.status_code}: {response.text[:200]}...")

            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error on attempt {attempt + 1}")
            except Exception as e:
                self.logger.warning(f"Request error on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) + self.request_delay
                self.logger.debug(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        self.logger.error("All API call attempts failed")
        return None

    def _create_enhanced_extraction_prompt(self, phrase: str, context: Dict, phrase_type: str = None) -> str:
        """
        Create an enhanced, more specific prompt for knowledge triple extraction.

        Args:
            phrase: The phrase text to extract from
            context: Document context (breadcrumb, metadata)
            phrase_type: Type of phrase (dosage_instruction, indication, etc.)

        Returns:
            The formatted prompt for the LLM
        """
        section_info = context.get('breadcrumb', 'Seção Desconhecida')
        phrase_category = phrase_type or context.get('metadata', {}).get('phrase_type', 'geral')

        # Create more specific instructions based on phrase type
        specific_instructions = {
            'dosage_instruction': 'Foque em doses, quantidades, frequências de administração.',
            'indication': 'Extraia para que condições ou doenças o medicamento é indicado.',
            'contraindication': 'Identifique quando o medicamento NÃO deve ser usado.',
            'side_effect': 'Extraia efeitos adversos, reações indesejáveis.',
            'precaution': 'Identifique cuidados, precauções, advertências.',
            'numerical_data': 'Extraia dados numéricos relevantes (doses, concentrações).',
            'general_information': 'Extraia qualquer informação farmacêutica relevante.'
        }

        instruction = specific_instructions.get(phrase_category, specific_instructions['general_information'])

        return f"""Você é um especialista em extrair informações farmacêuticas. Analise esta frase e extraia APENAS fatos reais como triplas JSON.

CONTEXTO: {section_info}
TIPO: {phrase_category}
INSTRUÇÃO: {instruction}

FRASE: "{phrase}"

REGRAS IMPORTANTES:
1. Extraia SOMENTE informações que estão EXPLÍCITAS na frase
2. NÃO invente ou suponha informações
3. Use nomes de medicamentos exatos quando mencionados
4. Para doses, inclua unidades (mg, ml, etc.)
5. Se não há informação farmacêutica específica, retorne []

FORMATO: Array JSON de triplas [entidade, relação, valor]

EXEMPLOS DE FORMATO (NÃO COPIE O CONTEÚDO):
- [["Paracetamol", "tem_dose", "500mg"]]
- [["medicamento", "é_indicado_para", "dor de cabeça"]]
- [["substância", "pode_causar", "náusea"]]

JSON:"""

    def _parse_triples_response_enhanced(self, response: str) -> List[List[str]]:
        """
        Enhanced parsing with better error handling and validation.

        Uses multiple parsing strategies to extract triples from LLM responses.

        Args:
            response: Raw LLM response text

        Returns:
            List of validated triples
        """
        if not response:
            return []

        # Clean the response
        cleaned = re.sub(r'```json\s*|```\s*', '', response.strip())
        cleaned = re.sub(r'^[^[]*', '', cleaned)  # Remove text before first [
        cleaned = re.sub(r'[^]]*$', ']', cleaned)  # Ensure ends with ]

        # Try multiple parsing strategies
        strategies = [
            self._parse_json_array,
            self._parse_regex_triples,
            self._parse_fallback_patterns
        ]

        for strategy in strategies:
            try:
                triples = strategy(cleaned)
                if triples:
                    return self._validate_and_filter_triples(triples)
            except Exception as e:
                self.logger.debug(f"Parsing strategy failed: {e}")
                continue

        self.logger.warning(f"Could not parse response: {cleaned[:100]}...")
        return []

    def _parse_json_array(self, text: str) -> List[List[str]]:
        """Parse JSON array directly."""
        # Find the JSON array pattern
        array_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
        return []

    def _parse_regex_triples(self, text: str) -> List[List[str]]:
        """Parse using regex patterns for triple extraction."""
        patterns = [
            r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]',  # Standard format
            r'\[\"([^\"]+)\",\s*\"([^\"]+)\",\s*\"([^\"]+)\"\]',  # Escaped quotes
            r'<([^>]+)>\s*,\s*<([^>]+)>\s*,\s*<([^>]+)>'  # Angle bracket format
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return [[str(x).strip() for x in match] for match in matches]
        return []

    def _parse_fallback_patterns(self, text: str) -> List[List[str]]:
        """Fallback parsing for malformed but recognizable patterns."""
        # Look for entity-relation-value patterns
        lines = text.split('\n')
        triples = []

        for line in lines:
            # Pattern: "entity" relation "value"
            pattern = r'"([^"]+)"\s+(\w+)\s+"([^"]+)"'
            match = re.search(pattern, line)
            if match:
                triples.append([match.group(1), match.group(2), match.group(3)])

        return triples

    def _validate_and_filter_triples(self, triples: List[List[str]]) -> List[List[str]]:
        """
        Validate and filter extracted triples for quality.

        Removes template examples, placeholders, and invalid entries.

        Args:
            triples: Raw list of triples

        Returns:
            Filtered list of valid triples
        """
        valid_triples = []

        # Filter out common template examples and invalid entries
        template_entities = ['medication', 'medicamento', 'paracetamol', 'substância', 'fármaco']
        template_values = ['500mg', 'comprimidos', 'dor de cabeça', 'náusea', 'exemplo']

        for triple in triples:
            if not isinstance(triple, list) or len(triple) != 3:
                continue

            entity, relation, value = [str(x).strip() for x in triple]

            # Skip if any component is empty or too short
            if not all([entity, relation, value]) or any(len(x) < 2 for x in [entity, relation, value]):
                continue

            # Skip template examples
            if (entity.lower() in template_entities and
                any(tv in value.lower() for tv in template_values)):
                continue

            # Skip placeholder patterns
            if any(x.startswith('<') and x.endswith('>') for x in [entity, relation, value]):
                continue

            # Skip overly generic relations
            generic_relations = ['é', 'tem', 'faz', 'usa']
            if relation.lower() in generic_relations and len(value) < 5:
                continue

            valid_triples.append([entity, relation, value])

        return valid_triples

    def _should_process_phrase(self, phrase: str, metadata: Dict = None) -> bool:
        """
        Determine if a phrase should be processed based on content relevance.

        Args:
            phrase: The phrase text
            metadata: Optional metadata containing phrase type

        Returns:
            True if phrase should be processed
        """
        if len(phrase.strip()) < 15:
            return False

        phrase_lower = phrase.lower()

        # Check for pharmaceutical keywords
        has_pharma_content = any(kw in phrase_lower for kw in self.pharma_keywords)

        # Check phrase type from metadata
        if metadata:
            phrase_type = metadata.get('phrase_type', '')
            if phrase_type in ['dosage_instruction', 'indication', 'contraindication', 'side_effect']:
                return True

        # Additional checks for numerical data that might be relevant
        has_numbers = bool(re.search(r'\d', phrase))
        has_units = bool(re.search(r'\d+\s*(mg|ml|g|%|mcg|μg)', phrase_lower))

        return has_pharma_content or has_units or (has_numbers and len(phrase) > 30)

    def _extract_phrase_knowledge(self, phrase_data: Dict) -> Dict:
        """
        Extract knowledge from a single phrase block.

        Args:
            phrase_data: Dictionary containing phrase content, context, and metadata

        Returns:
            Dictionary with extraction results and status
        """
        phrase_id = phrase_data.get('phrase_id', 'unknown')
        phrase_content = phrase_data.get('content', '').strip()
        context = phrase_data.get('context', {})
        metadata = phrase_data.get('metadata', {})

        if not self._should_process_phrase(phrase_content, metadata):
            self.stats['skipped_irrelevant'] += 1
            return {
                'phrase_id': phrase_id,
                'triples': [],
                'status': 'skipped_irrelevant'
            }

        self.logger.debug(f"Processing phrase {phrase_id}: {phrase_content[:50]}...")
        self.stats['phrases_processed'] += 1

        try:
            phrase_type = metadata.get('phrase_type')
            prompt = self._create_enhanced_extraction_prompt(phrase_content, context, phrase_type)

            # Log the interaction
            self.prompt_logger.info(f"--- START PHRASE: {phrase_id} ---")
            self.prompt_logger.info(f"PHRASE TEXT: {phrase_content}")
            self.prompt_logger.info(f"PHRASE TYPE: {phrase_type}")
            self.prompt_logger.info(f"CONTEXT: {context.get('breadcrumb', 'N/A')}")
            self.prompt_logger.info(f"PROMPT SENT:\n{prompt}")

            response = self._call_ollama_api(prompt, max_tokens=300)

            self.prompt_logger.info(f"RAW RESPONSE RECEIVED:\n{response}")
            self.prompt_logger.info(f"--- END PHRASE: {phrase_id} ---\n")

            if response:
                triples = self._parse_triples_response_enhanced(response)
                if triples:
                    self.stats['successful_extractions'] += 1
                    self.stats['total_triples'] += len(triples)
                    self.logger.debug(f"Extracted {len(triples)} triples from phrase {phrase_id}")
                else:
                    self.stats['failed_extractions'] += 1

                return {
                    'phrase_id': phrase_id,
                    'phrase_text': phrase_content,
                    'phrase_type': phrase_type,
                    'context': context.get('breadcrumb'),
                    'triples': triples,
                    'status': 'success' if triples else 'no_triples_found'
                }
            else:
                self.stats['failed_extractions'] += 1
                return {
                    'phrase_id': phrase_id,
                    'phrase_text': phrase_content,
                    'triples': [],
                    'status': 'api_failed'
                }

        except Exception as e:
            self.logger.error(f"Error processing phrase {phrase_id}: {e}")
            self.stats['failed_extractions'] += 1
            return {
                'phrase_id': phrase_id,
                'phrase_text': phrase_content,
                'triples': [],
                'status': f'error: {str(e)}'
            }
        finally:
            time.sleep(self.request_delay)

    def _extract_table_knowledge(self, table_data: Dict) -> Dict:
        """
        Extract knowledge from table blocks with structured data handling.

        Args:
            table_data: Dictionary containing table content, context, and metadata

        Returns:
            Dictionary with extraction results and status
        """
        table_id = table_data.get('table_id', 'unknown')
        content = table_data.get('content', {})
        context = table_data.get('context', {})
        metadata = table_data.get('metadata', {})

        self.logger.info(f"Processing table {table_id}")
        self.stats['table_blocks_processed'] += 1

        # Convert table to text for processing
        formatted_text = content.get('formatted_text', '')
        header = content.get('header', [])
        data_rows = content.get('data_rows', [])

        if not formatted_text and not data_rows:
            return {
                'table_id': table_id,
                'triples': [],
                'status': 'empty_table'
            }

        # Process table as structured text
        table_text = formatted_text or self._format_table_as_text(header, data_rows)

        # Use table-specific processing
        try:
            prompt = self._create_table_extraction_prompt(table_text, context, metadata)

            self.prompt_logger.info(f"--- START TABLE: {table_id} ---")
            self.prompt_logger.info(f"TABLE CONTENT:\n{table_text}")
            self.prompt_logger.info(f"PROMPT SENT:\n{prompt}")

            response = self._call_ollama_api(prompt, max_tokens=400)

            self.prompt_logger.info(f"RAW RESPONSE RECEIVED:\n{response}")
            self.prompt_logger.info(f"--- END TABLE: {table_id} ---\n")

            if response:
                triples = self._parse_triples_response_enhanced(response)
                if triples:
                    self.stats['successful_extractions'] += 1
                    self.stats['total_triples'] += len(triples)

                return {
                    'table_id': table_id,
                    'table_type': metadata.get('table_type'),
                    'context': context.get('breadcrumb'),
                    'triples': triples,
                    'status': 'success' if triples else 'no_triples_found'
                }

        except Exception as e:
            self.logger.error(f"Error processing table {table_id}: {e}")
            self.stats['failed_extractions'] += 1

        return {
            'table_id': table_id,
            'triples': [],
            'status': 'error'
        }

    def _create_table_extraction_prompt(self, table_text: str, context: Dict, metadata: Dict) -> str:
        """Create specialized prompt for table data extraction."""
        table_type = metadata.get('table_type', 'general_data')
        section_info = context.get('breadcrumb', 'Tabela')

        type_instructions = {
            'dosage_schedule': 'Extraia informações de dosagem, horários, frequências.',
            'dosage_information': 'Foque em doses, concentrações, quantidades.',
            'age_specific_data': 'Extraia dados específicos por idade ou grupo.',
            'general_data': 'Extraia qualquer informação farmacêutica estruturada.'
        }

        instruction = type_instructions.get(table_type, type_instructions['general_data'])

        return f"""Analise esta tabela farmacêutica e extraia informações estruturadas como triplas JSON.

CONTEXTO: {section_info}
TIPO DE TABELA: {table_type}
INSTRUÇÃO: {instruction}

TABELA:
{table_text}

REGRAS:
1. Extraia APENAS dados que estão na tabela
2. Para doses, mantenha unidades (mg, ml, etc.)
3. Preserve nomes de medicamentos exatos
4. Se há múltiplas linhas, extraia informação de cada linha relevante
5. Use "linha_N" ou "item_N" para distinguir entradas quando necessário

FORMATO: Array JSON de triplas [entidade, relação, valor]

JSON:"""

    def _format_table_as_text(self, header: List[str], data_rows: List[List[str]]) -> str:
        """Format table data as readable text."""
        if not data_rows:
            return ""

        lines = []
        if header:
            lines.append(" | ".join(header))
            lines.append("-" * (len(" | ".join(header))))

        for row in data_rows:
            lines.append(" | ".join(str(cell) if cell else "" for cell in row))

        return "\n".join(lines)

    def process_phrase_based_json(self, input_file: Path) -> Optional[Dict]:
        """
        Process a phrase-based JSON file from the enhanced parser.

        Args:
            input_file: Path to the phrase-based JSON file

        Returns:
            Dictionary with extraction results or None if processing fails
        """
        self.logger.info(f"Processing phrase-based file: {input_file.name}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {input_file}: {e}")
            return None

        # Get the document structure
        doc_structure = data.get('document_structure', {})
        phrase_blocks = doc_structure.get('phrase_blocks', [])
        table_blocks = doc_structure.get('table_blocks', [])

        if not phrase_blocks and not table_blocks:
            self.logger.warning(f"No phrase_blocks or table_blocks found in {input_file}")
            return None

        self.logger.info(f"Found {len(phrase_blocks)} phrase blocks and {len(table_blocks)} table blocks")

        # Process phrase blocks
        phrase_extractions = []
        for phrase_data in phrase_blocks:
            result = self._extract_phrase_knowledge(phrase_data)
            phrase_extractions.append(result)
            self.stats['phrase_blocks_processed'] += 1

        # Process table blocks
        table_extractions = []
        for table_data in table_blocks:
            result = self._extract_table_knowledge(table_data)
            table_extractions.append(result)

        # Collect all triples
        all_triples = []
        for extraction in phrase_extractions + table_extractions:
            all_triples.extend(extraction.get('triples', []))

        result = {
            'document_metadata': data.get('document_metadata', {}),
            'extraction_summary': {
                'extraction_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'processing_method': 'enhanced_phrase_based',
                'total_phrase_blocks': len(phrase_blocks),
                'total_table_blocks': len(table_blocks),
                'total_phrases_processed': self.stats['phrases_processed'],
                'total_triples_extracted': len(all_triples),
                'successful_extractions': self.stats['successful_extractions'],
                'failed_extractions': self.stats['failed_extractions'],
                'skipped_irrelevant': self.stats['skipped_irrelevant']
            },
            'phrase_extractions': phrase_extractions,
            'table_extractions': table_extractions,
            'all_extracted_triples': all_triples,
            'metadata': data.get('metadata', {})
        }

        self.stats['files_processed'] += 1
        self.logger.info(f"Completed {input_file.name}: {len(all_triples)} total triples extracted")
        return result

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Process all phrase-optimized JSON files in a directory.

        Args:
            input_dir: Directory containing phrase-optimized JSON files
            output_dir: Directory to save extraction results
        """
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Look for phrase-optimized JSON files
        json_files = list(input_dir.glob('*_phrase_optimized.json'))
        if not json_files:
            self.logger.warning(f"No *_phrase_optimized.json files found in {input_dir}")
            return

        self.logger.info(f"Found {len(json_files)} phrase-optimized files to process")

        for i, json_file in enumerate(json_files):
            self.logger.info(f"\nProgress: Processing file {i + 1}/{len(json_files)}")

            # Reset per-file counters
            prev_phrases = self.stats['phrases_processed']
            prev_successful = self.stats['successful_extractions']

            result = self.process_phrase_based_json(json_file)

            if result:
                output_name = json_file.stem.replace('_phrase_optimized', '_enhanced_graph_data') + '.json'
                output_file = output_dir / output_name

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Log file-specific stats
                phrases_this_file = self.stats['phrases_processed'] - prev_phrases
                successful_this_file = self.stats['successful_extractions'] - prev_successful

                self.logger.info(f"Saved results to: {output_file}")
                self.logger.info(f"File stats: {phrases_this_file} phrases processed, {successful_this_file} successful extractions")

        self._generate_enhanced_report(output_dir)

    def _generate_enhanced_report(self, output_dir: Path) -> None:
        """Generate comprehensive final report."""
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'processing_method': 'enhanced_phrase_based',
                'total_files_processed': self.stats['files_processed'],
            },
            'detailed_statistics': self.stats,
            'performance_metrics': {
                'success_rate': (
                    self.stats['successful_extractions'] /
                    max(self.stats['phrases_processed'], 1) * 100
                ),
                'avg_triples_per_successful_extraction': (
                    self.stats['total_triples'] /
                    max(self.stats['successful_extractions'], 1)
                ),
                'processing_efficiency': {
                    'phrases_processed': self.stats['phrases_processed'],
                    'relevant_phrases': self.stats['phrases_processed'] - self.stats['skipped_irrelevant'],
                    'relevance_rate': (
                        (self.stats['phrases_processed'] - self.stats['skipped_irrelevant']) /
                        max(self.stats['phrases_processed'], 1) * 100
                    )
                }
            }
        }

        report_file = output_dir / 'enhanced_final_extraction_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Enhanced final report saved: {report_file}")
        self._print_summary_stats()

    def _print_summary_stats(self) -> None:
        """Print summary statistics to console."""
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Phrase blocks processed: {self.stats['phrase_blocks_processed']}")
        print(f"Table blocks processed: {self.stats['table_blocks_processed']}")
        print(f"Total phrases analyzed: {self.stats['phrases_processed']}")
        print(f"Successful extractions: {self.stats['successful_extractions']}")
        print(f"Failed extractions: {self.stats['failed_extractions']}")
        print(f"Skipped irrelevant: {self.stats['skipped_irrelevant']}")
        print(f"Total triples extracted: {self.stats['total_triples']}")

        if self.stats['phrases_processed'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['phrases_processed']) * 100
            print(f"Success rate: {success_rate:.1f}%")

        print("=" * 70)
