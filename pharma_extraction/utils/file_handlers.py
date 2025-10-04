"""File handling utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object for the directory

    Example:
        >>> output_dir = ensure_directory(Path("./outputs"))
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(
    data: Any,
    file_path: Path,
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        indent: JSON indentation level
        ensure_ascii: Ensure ASCII-only output

    Returns:
        True if successful, False otherwise

    Example:
        >>> data = {"name": "Paracetamol", "dose": "500mg"}
        >>> save_json(data, Path("output.json"))
    """
    try:
        file_path = Path(file_path)
        ensure_directory(file_path.parent)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(f"Successfully saved JSON to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load data from JSON file.

    Args:
        file_path: Input file path

    Returns:
        Loaded data or None if loading failed

    Example:
        >>> data = load_json(Path("input.json"))
        >>> print(data['name'])
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Successfully loaded JSON from {file_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None

    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes.

    Args:
        file_path: File path

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string

    Example:
        >>> format_file_size(1024)
        '1.00 KB'
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
