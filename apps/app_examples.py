import sys
import os

# Add parent directory to path so we can import from core/ and features/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Solved Examples",
    page_icon="📚",
    layout="wide",
)

from features.solved_examples import render_solved_examples


def main():
    render_solved_examples()


if __name__ == "__main__":
    main()
