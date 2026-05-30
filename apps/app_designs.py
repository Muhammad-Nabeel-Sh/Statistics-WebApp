import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Study Design Explorer",
    page_icon="📐",
    layout="wide",
)

from features.study_designs import render_study_designs
from core.utils import render_footer


def main():
    render_study_designs()
    render_footer()


if __name__ == "__main__":
    main()
