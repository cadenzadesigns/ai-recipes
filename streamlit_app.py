"""Main entry point for Streamlit deployment."""
import sys
from pathlib import Path

# Add both src and app directories to Python path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from app.main import main

if __name__ == "__main__":
    # Set up Streamlit page config
    st.set_page_config(
        page_title="AI Recipe Extractor",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Run the main app
    main()
