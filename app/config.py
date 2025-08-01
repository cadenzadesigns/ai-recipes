"""
Configuration settings for the AI Recipe Extractor Streamlit app.

This module contains only the settings actually used by the web application.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

# =============================================================================
# DOWNLOAD SETTINGS
# =============================================================================

# Download formats
DOWNLOAD_FORMATS = {
    "txt": {"label": "📄 Download as Text", "mime": "text/plain", "extension": ".txt"},
    "json": {
        "label": "📊 Download as JSON",
        "mime": "application/json",
        "extension": ".json",
    },
    "zip": {
        "label": "📦 Download as ZIP",
        "mime": "application/zip",
        "extension": ".zip",
    },
}

# ZIP archive settings
ZIP_COMPRESSION_LEVEL = 6  # 0-9, 6 is good balance of speed/size
MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500MB max ZIP file size

# =============================================================================
# IMAGE EXTRACTION SETTINGS
# =============================================================================

# Manual cropping settings
ENABLE_MANUAL_CROPPING = True  # Enable manual cropping feature
MAX_CROPS_PER_IMAGE = 5  # Maximum number of crop regions per image

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_safe_filename(text: str, max_length: int = 50) -> str:
    """Generate a safe filename from text."""
    import re

    # Remove or replace invalid characters
    safe_text = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe_text = re.sub(r"\s+", "_", safe_text)
    safe_text = safe_text.strip("._")

    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip("_")

    return safe_text.lower()
