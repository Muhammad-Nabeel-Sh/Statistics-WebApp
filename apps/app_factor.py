import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Factor Analysis",
    page_icon="🧬",
    layout="wide",
)

from features.factor_analysis import render_factor_analysis
from core.utils import render_footer


def main():
    render_factor_analysis()
    render_footer()


if __name__ == "__main__":
    main()
