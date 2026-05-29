import streamlit as st

st.set_page_config(page_title="Statistical Test Finder", page_icon="🔍", layout="wide")

from features.finder_ui import render_test_finder
from core.utils import render_footer


def main():
    render_test_finder()
    render_footer()


if __name__ == "__main__":
    main()
