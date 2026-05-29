import streamlit as st

st.set_page_config(page_title="Data Screening & Diagnostics", page_icon="🔬", layout="wide")

from features.diagnostics import render_diagnostics
from core.utils import render_footer


def main():
    render_diagnostics()
    render_footer()


if __name__ == "__main__":
    main()
