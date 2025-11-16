"""
Setup configuration for SIE-X (Semantic Intelligence Engine X).
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.minimal.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sie-x",
    version="1.0.0",
    author="SIE-X Team",
    description="Semantic Intelligence Engine X - Keyword Extraction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robwestz/SEI-X",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "siex=sie_x.cli:main",
        ],
    },
)
