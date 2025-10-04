"""Configuration settings for pharmaceutical extraction system.

This module centralizes all configuration values used across parsers and extractors,
allowing easy customization through environment variables.
"""

import os
from typing import Optional
from pathlib import Path


class Config:
    """Configuration class for pharmaceutical extraction.

    All settings can be overridden using environment variables with the prefix 'PHARMA_'.

    Example:
        export PHARMA_OLLAMA_URL=http://my-ollama-server:11434

    Attributes:
        OLLAMA_URL: Base URL for Ollama API server
        OLLAMA_MODEL: Default LLM model to use
        OLLAMA_TIMEOUT: Request timeout in seconds
        OLLAMA_TEMPERATURE: LLM temperature (0.0 = deterministic)
        OLLAMA_TOP_P: Nucleus sampling parameter
        OLLAMA_TOP_K: Top-k sampling parameter
        OLLAMA_MAX_TOKENS: Maximum tokens in LLM response
        OLLAMA_CONTEXT_WINDOW: Context window size
        MAX_RETRIES: Maximum retry attempts for failed requests
        REQUEST_DELAY: Delay between requests in seconds
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        OUTPUT_DIR: Default directory for output files
        PDF_DIR: Default directory for input PDFs
    """

    # Ollama/LLM Configuration
    OLLAMA_URL: str = os.getenv('PHARMA_OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_GENERATE_URL: str = f"{OLLAMA_URL}/api/generate"
    OLLAMA_MODEL: str = os.getenv('PHARMA_OLLAMA_MODEL', 'llama3.2:3b')
    OLLAMA_TIMEOUT: int = int(os.getenv('PHARMA_OLLAMA_TIMEOUT', '120'))
    OLLAMA_TEMPERATURE: float = float(os.getenv('PHARMA_OLLAMA_TEMPERATURE', '0.0'))
    OLLAMA_TOP_P: float = float(os.getenv('PHARMA_OLLAMA_TOP_P', '0.9'))
    OLLAMA_TOP_K: int = int(os.getenv('PHARMA_OLLAMA_TOP_K', '20'))
    OLLAMA_MAX_TOKENS: int = int(os.getenv('PHARMA_OLLAMA_MAX_TOKENS', '400'))
    OLLAMA_CONTEXT_WINDOW: int = int(os.getenv('PHARMA_OLLAMA_CONTEXT_WINDOW', '2048'))

    # Request Configuration
    MAX_RETRIES: int = int(os.getenv('PHARMA_MAX_RETRIES', '3'))
    REQUEST_DELAY: float = float(os.getenv('PHARMA_REQUEST_DELAY', '0.5'))

    # Logging Configuration
    LOG_LEVEL: str = os.getenv('PHARMA_LOG_LEVEL', 'INFO')

    # Directory Configuration
    OUTPUT_DIR: Path = Path(os.getenv('PHARMA_OUTPUT_DIR', './data/outputs'))
    PDF_DIR: Path = Path(os.getenv('PHARMA_PDF_DIR', './data/pdfs'))

    # Parsing Configuration
    MIN_SECTION_LENGTH: int = int(os.getenv('PHARMA_MIN_SECTION_LENGTH', '20'))
    TABLE_DETECTION_ENABLED: bool = os.getenv('PHARMA_TABLE_DETECTION', 'true').lower() == 'true'

    # Entity Extraction Configuration
    PHARMACEUTICAL_KEYWORDS = [
        'medicamento', 'fármaco', 'droga', 'princípio ativo', 'excipiente',
        'dose', 'dosagem', 'posologia', 'administração', 'via de administração',
        'indicação', 'contraindicação', 'efeito colateral', 'reação adversa',
        'precaução', 'advertência', 'interação medicamentosa', 'superdosagem',
        'armazenamento', 'conservação', 'validade', 'fabricante', 'laboratório',
        'composição', 'forma farmacêutica', 'apresentação', 'concentração',
        'população', 'idade', 'adulto', 'criança', 'idoso', 'gestante', 'lactante',
        'mg', 'ml', 'comprimido', 'cápsula', 'solução', 'suspensão', 'xarope',
        'paciente', 'tratamento', 'terapia', 'mecanismo de ação', 'farmacocinética',
        'farmacodinâmica', 'absorção', 'distribuição', 'metabolismo', 'excreção'
    ]

    # Template filtering patterns (common non-informative phrases to exclude)
    TEMPLATE_PATTERNS = [
        'Este medicamento',
        'Não use',
        'Consulte seu médico',
        'Mantenha fora do alcance',
        'Em caso de',
        'Se você',
        'Procure um médico'
    ]

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PDF_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_ollama_connection(cls) -> bool:
        """Check if Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            import requests
            response = requests.get(cls.OLLAMA_URL, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def get_model_info(cls) -> Optional[dict]:
        """Get information about the configured Ollama model.

        Returns:
            Dictionary with model information or None if unavailable
        """
        try:
            import requests
            response = requests.post(
                f"{cls.OLLAMA_URL}/api/show",
                json={"name": cls.OLLAMA_MODEL},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dictionary.

        Returns:
            Dictionary containing all configuration values
        """
        return {
            'ollama': {
                'url': cls.OLLAMA_URL,
                'generate_url': cls.OLLAMA_GENERATE_URL,
                'model': cls.OLLAMA_MODEL,
                'timeout': cls.OLLAMA_TIMEOUT,
                'temperature': cls.OLLAMA_TEMPERATURE,
                'top_p': cls.OLLAMA_TOP_P,
                'top_k': cls.OLLAMA_TOP_K,
                'max_tokens': cls.OLLAMA_MAX_TOKENS,
                'context_window': cls.OLLAMA_CONTEXT_WINDOW,
            },
            'requests': {
                'max_retries': cls.MAX_RETRIES,
                'request_delay': cls.REQUEST_DELAY,
            },
            'logging': {
                'log_level': cls.LOG_LEVEL,
            },
            'directories': {
                'output_dir': str(cls.OUTPUT_DIR),
                'pdf_dir': str(cls.PDF_DIR),
            },
            'parsing': {
                'min_section_length': cls.MIN_SECTION_LENGTH,
                'table_detection_enabled': cls.TABLE_DETECTION_ENABLED,
            }
        }


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
