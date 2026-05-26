import sys
import os

# Add parent directory to path so we can import from core/ and features/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Power Analysis & Sample Size",
    page_icon="⚡",
    layout="wide",
)

from features.power_ui import _render_power_analysis


def main():
    _render_power_analysis()


if __name__ == "__main__":
    main()
