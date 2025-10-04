# Contributing to Pharma Extraction

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

Feature requests are welcome! Please open an issue describing:
- The feature and its use case
- Why it would be valuable
- Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/pharma-extraction.git
   cd pharma-extraction
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Add type hints
   - Follow existing code style

4. **Test your changes**
   ```bash
   pytest
   ```

5. **Format your code**
   ```bash
   black pharma_extraction/
   flake8 pharma_extraction/
   ```

6. **Commit your changes**
   ```bash
   git commit -m "Add feature: your feature description"
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8
- Use `black` for formatting
- Add docstrings to all functions/classes
- Use type hints
- Keep functions focused and small

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black pharma_extraction/
```

## Questions?

Feel free to open an issue for any questions!
