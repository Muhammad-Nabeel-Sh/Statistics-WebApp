import sys
import os

# Add parent directory to path so we can import from core/ and features/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Data Screening & Diagnostics",
    page_icon="🔬",
    layout="wide",
)

from features.diagnostics import render_diagnostics


def main():
    render_diagnostics()


if __name__ == "__main__":
    main()
