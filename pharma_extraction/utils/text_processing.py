"""Text processing utilities for pharmaceutical documents."""

import re
from typing import List, Dict, Tuple
from pharma_extraction.config import Config


def split_into_sentences(text: str, preserve_abbreviations: bool = True) -> List[str]:
    """Split text into sentences while preserving pharmaceutical abbreviations.

    Args:
        text: Input text to split
        preserve_abbreviations: Preserve common medical abbreviations (mg, ml, etc.)

    Returns:
        List of sentences

    Example:
        >>> text = "Dose: 500mg. Tomar 3x ao dia."
        >>> sentences = split_into_sentences(text)
        >>> print(sentences)
        ['Dose: 500mg.', 'Tomar 3x ao dia.']
    """
    # Common pharmaceutical abbreviations that should not trigger sentence split
    abbreviations = [
        'mg', 'ml', 'mcg', 'UI', 'mL', 'dL', 'kg', 'g',
        'Dr', 'Dra', 'Sr', 'Sra', 'vs', 'etc', 'ex',
        'nº', 'pág', 'vol', 'ed', 'cap'
    ]

    if preserve_abbreviations:
        # Temporarily replace abbreviations with placeholders
        temp_text = text
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            pattern = rf'\b{re.escape(abbr)}\.'
            placeholder = f'__ABBR_{i}__'
            temp_text = re.sub(pattern, f'{abbr}{placeholder}', temp_text)
            placeholders[placeholder] = '.'

        # Split on sentence boundaries
        sentences = re.split(r'[.!?;]\s+', temp_text)

        # Restore abbreviations
        restored = []
        for sentence in sentences:
            for placeholder, replacement in placeholders.items():
                sentence = sentence.replace(placeholder, replacement)
            restored.append(sentence.strip())

        return [s for s in restored if s]
    else:
        # Simple split on punctuation
        return [s.strip() for s in re.split(r'[.!?;]\s+', text) if s.strip()]


def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """Clean and normalize text.

    Args:
        text: Input text to clean
        remove_extra_whitespace: Remove extra spaces and newlines

    Returns:
        Cleaned text

    Example:
        >>> text = "  Paracetamol\\n\\n500mg  "
        >>> clean_text(text)
        'Paracetamol 500mg'
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    if remove_extra_whitespace:
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_sections(text: str, numbered: bool = True) -> List[Dict[str, str]]:
    """Extract numbered sections from pharmaceutical document.

    Args:
        text: Document text
        numbered: Extract numbered sections (e.g., "1. TITLE")

    Returns:
        List of dictionaries with 'number', 'title', and 'content'

    Example:
        >>> text = "1. INDICAÇÕES\\nPara dor de cabeça\\n2. DOSE\\n500mg"
        >>> sections = extract_sections(text)
        >>> print(sections[0])
        {'number': '1', 'title': 'INDICAÇÕES', 'content': 'Para dor de cabeça'}
    """
    sections = []

    if numbered:
        # Pattern for numbered sections: "1. TITLE" or "1 TITLE"
        pattern = r'(\d+(?:\.\d+)*)[.\s]+([A-ZÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜ\s]+?)(?=\n|\r|$)'
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(matches):
            section_number = match.group(1)
            section_title = match.group(2).strip()

            # Extract content until next section
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()

            sections.append({
                'number': section_number,
                'title': section_title,
                'content': content
            })

    return sections


def is_pharmaceutical_content(text: str, threshold: int = 2) -> bool:
    """Check if text contains pharmaceutical content.

    Args:
        text: Text to check
        threshold: Minimum number of pharmaceutical keywords required

    Returns:
        True if text appears to be pharmaceutical content

    Example:
        >>> text = "Paracetamol é indicado para dor de cabeça. Dose: 500mg"
        >>> is_pharmaceutical_content(text)
        True
    """
    config = Config()
    text_lower = text.lower()

    keyword_count = sum(
        1 for keyword in config.PHARMACEUTICAL_KEYWORDS
        if keyword.lower() in text_lower
    )

    return keyword_count >= threshold


def extract_dosage_info(text: str) -> List[Dict[str, str]]:
    """Extract dosage information from text.

    Args:
        text: Text containing dosage information

    Returns:
        List of dictionaries with dosage details

    Example:
        >>> text = "Adultos: 500mg 3x ao dia"
        >>> dosages = extract_dosage_info(text)
        >>> print(dosages[0])
        {'population': 'Adultos', 'amount': '500mg', 'frequency': '3x ao dia'}
    """
    dosages = []

    # Pattern for dosage: "500mg", "500 mg", "1-2 comprimidos"
    dosage_pattern = r'(\d+(?:[.,]\d+)?)\s*(mg|ml|mcg|g|comprimidos?|cápsulas?|gotas?)'

    # Pattern for frequency: "3x ao dia", "duas vezes ao dia", "a cada 8 horas"
    frequency_pattern = r'(\d+x|uma vez|duas vezes|três vezes|a cada \d+ horas?)\s*(?:ao dia|por dia)?'

    # Pattern for population: "Adultos:", "Crianças de 6-12 anos:"
    population_pattern = r'(Adultos?|Crianças?|Idosos?|Gestantes?|Lactantes?)(?:\s*(?:de|com)?\s*[\d\-]+\s*anos?)?:'

    # Simple extraction (can be enhanced)
    dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    frequency_matches = list(re.finditer(frequency_pattern, text, re.IGNORECASE))
    population_matches = list(re.finditer(population_pattern, text, re.IGNORECASE))

    for i, dmatch in enumerate(dosage_matches):
        dosage_info = {
            'amount': dmatch.group(0),
            'frequency': frequency_matches[i].group(0) if i < len(frequency_matches) else '',
            'population': population_matches[i].group(0).rstrip(':') if i < len(population_matches) else ''
        }
        dosages.append(dosage_info)

    return dosages


def normalize_entity_name(entity: str) -> str:
    """Normalize entity names for consistent matching.

    Args:
        entity: Entity name to normalize

    Returns:
        Normalized entity name

    Example:
        >>> normalize_entity_name("  PARACETAMOL  ")
        'paracetamol'
    """
    # Convert to lowercase
    normalized = entity.lower()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Remove special characters but keep hyphens and spaces
    normalized = re.sub(r'[^\w\s\-]', '', normalized)

    return normalized


def extract_numbers(text: str) -> List[Tuple[str, str]]:
    """Extract numbers with their units from text.

    Args:
        text: Text containing numbers

    Returns:
        List of tuples (number, unit)

    Example:
        >>> text = "500mg por dia, máximo 3000mg"
        >>> numbers = extract_numbers(text)
        >>> print(numbers)
        [('500', 'mg'), ('3000', 'mg')]
    """
    pattern = r'(\d+(?:[.,]\d+)?)\s*(mg|ml|mcg|g|kg|L|UI|%|comprimidos?|cápsulas?)?'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    results = []
    for match in matches:
        number = match.group(1)
        unit = match.group(2) if match.group(2) else ''
        results.append((number, unit))

    return results


def detect_section_type(title: str) -> str:
    """Detect the type/category of a document section.

    Args:
        title: Section title

    Returns:
        Section type category

    Example:
        >>> detect_section_type("PARA QUE ESTE MEDICAMENTO É INDICADO?")
        'indication'
    """
    title_lower = title.lower()

    type_mapping = {
        'indication': ['indicado', 'indicação', 'indicações', 'para que'],
        'contraindication': ['contraindicado', 'contraindicação', 'contraindicações', 'não use'],
        'dosage': ['posologia', 'dose', 'dosagem', 'como usar', 'como devo usar'],
        'side_effects': ['efeito colateral', 'efeitos colaterais', 'reação adversa', 'reações adversas'],
        'precaution': ['precauç', 'advertência', 'advertências', 'cuidado'],
        'interaction': ['interação', 'interações medicamentosas'],
        'composition': ['composição', 'fórmula', 'princípio ativo'],
        'storage': ['armazenamento', 'conservação', 'como conservar'],
        'pharmacology': ['farmacocinética', 'farmacodinâmica', 'características farmacológicas']
    }

    for section_type, keywords in type_mapping.items():
        if any(keyword in title_lower for keyword in keywords):
            return section_type

    return 'general'
