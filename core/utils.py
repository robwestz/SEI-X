"""
Utility functions for SIE-X core.

Provides helper functions for:
- spaCy model loading with auto-download
- Model management
- Common operations
"""

import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str) -> Any:
    """
    Load spaCy model with automatic download on first use.

    If the model is not found, automatically downloads it using
    `python -m spacy download <model_name>` and retries.

    This provides better UX for new users who haven't manually
    downloaded models.

    Args:
        model_name: Name of spaCy model (e.g., 'en_core_web_sm')

    Returns:
        Loaded spaCy Language object

    Raises:
        Exception: If model download or loading fails

    Example:
        >>> nlp = load_spacy_model('en_core_web_sm')
        >>> doc = nlp("Hello world")
    """
    import spacy

    try:
        # Try to load model
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model: {model_name}")
        return nlp

    except OSError as e:
        # Model not found - try to download it
        logger.warning(f"spaCy model '{model_name}' not found, attempting auto-download...")

        try:
            # Download the model
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Successfully downloaded spaCy model: {model_name}")

            # Retry loading
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp

        except subprocess.CalledProcessError as download_error:
            logger.error(f"Failed to download spaCy model '{model_name}': {download_error}")
            raise Exception(
                f"Could not download spaCy model '{model_name}'. "
                f"Please install manually: python -m spacy download {model_name}"
            ) from download_error

        except OSError as load_error:
            logger.error(f"Downloaded model '{model_name}' but failed to load: {load_error}")
            raise Exception(
                f"Model '{model_name}' was downloaded but could not be loaded. "
                f"Try restarting your Python environment."
            ) from load_error

    except Exception as e:
        logger.error(f"Failed to load spaCy model '{model_name}': {e}")
        raise


def check_spacy_model_exists(model_name: str) -> bool:
    """
    Check if a spaCy model is installed without loading it.

    Args:
        model_name: Name of spaCy model

    Returns:
        True if model is installed, False otherwise
    """
    import spacy

    try:
        spacy.load(model_name, disable=["parser", "ner", "textcat"])
        return True
    except OSError:
        return False


def download_spacy_model(model_name: str, force: bool = False) -> bool:
    """
    Download a spaCy model.

    Args:
        model_name: Name of spaCy model to download
        force: If True, re-download even if already installed

    Returns:
        True if download successful, False otherwise
    """
    if not force and check_spacy_model_exists(model_name):
        logger.info(f"spaCy model '{model_name}' already installed")
        return True

    try:
        logger.info(f"Downloading spaCy model: {model_name}")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Successfully downloaded: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download '{model_name}': {e}")
        return False
