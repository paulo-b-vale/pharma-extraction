"""LLM client utilities for Ollama integration."""

import json
import time
import logging
import requests
from typing import Optional, Dict, Any
from pharma_extraction.config import Config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM API.

    This class handles all communication with the Ollama server, including
    retry logic, error handling, and response parsing.

    Attributes:
        config: Configuration instance with Ollama settings
        session: Requests session for connection pooling
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize Ollama client.

        Args:
            config: Configuration instance (uses default if None)
        """
        self.config = config or Config()
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format_json: bool = False
    ) -> Optional[str]:
        """Generate text using Ollama LLM.

        Args:
            prompt: Input prompt for the LLM
            model: Model name (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)
            format_json: Force JSON output format

        Returns:
            Generated text or None if generation failed

        Example:
            >>> client = OllamaClient()
            >>> response = client.generate("What is paracetamol?")
            >>> print(response)
        """
        payload = {
            'model': model or self.config.OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature if temperature is not None else self.config.OLLAMA_TEMPERATURE,
                'top_p': self.config.OLLAMA_TOP_P,
                'top_k': self.config.OLLAMA_TOP_K,
                'num_predict': max_tokens or self.config.OLLAMA_MAX_TOKENS,
            }
        }

        if format_json:
            payload['format'] = 'json'

        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.config.OLLAMA_GENERATE_URL,
                    json=payload,
                    timeout=self.config.OLLAMA_TIMEOUT
                )
                response.raise_for_status()

                result = response.json()
                if 'response' in result:
                    return result['response']

                logger.warning(f"Unexpected response format: {result}")
                return None

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}/{self.config.MAX_RETRIES}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.REQUEST_DELAY * (attempt + 1))
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}/{self.config.MAX_RETRIES}: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.REQUEST_DELAY * (attempt + 1))
                continue

            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}")
                return None

        logger.error(f"Failed to generate response after {self.config.MAX_RETRIES} attempts")
        return None

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate JSON response using Ollama LLM.

        Args:
            prompt: Input prompt for the LLM
            model: Model name (uses config default if None)
            temperature: Sampling temperature (uses config default if None)

        Returns:
            Parsed JSON dictionary or None if generation/parsing failed

        Example:
            >>> client = OllamaClient()
            >>> result = client.generate_json("Extract entities as JSON: Paracetamol 500mg")
            >>> print(result)
        """
        response = self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            format_json=True
        )

        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return None

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self.session.get(
                self.config.OLLAMA_URL,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False

    def list_models(self) -> Optional[list]:
        """List available models on Ollama server.

        Returns:
            List of model names or None if request failed
        """
        try:
            response = self.session.get(
                f"{self.config.OLLAMA_URL}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return None

    def pull_model(self, model_name: str) -> bool:
        """Pull (download) a model from Ollama registry.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.post(
                f"{self.config.OLLAMA_URL}/api/pull",
                json={'name': model_name},
                timeout=300  # Longer timeout for model download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    def close(self):
        """Close the client session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
