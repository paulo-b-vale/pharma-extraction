"""Setup configuration for pharma_extraction package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="pharma_extraction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Pharmaceutical document extraction and knowledge graph generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pharma_extraction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pdfplumber>=0.10.0",
        "pymupdf4llm>=0.0.1",
        "pypdfium2>=4.0.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pharma-extract=pharma_extraction.cli:main",
        ],
    },
)
