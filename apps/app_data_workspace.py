import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Data Workspace",
    page_icon="📊",
    layout="wide",
)

from features.data_workspace import render_data_workspace
from core.utils import render_footer


def main():
    render_data_workspace()
    render_footer()


if __name__ == "__main__":
    main()
