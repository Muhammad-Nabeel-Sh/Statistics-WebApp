import streamlit as st

st.set_page_config(page_title="Tabulation", page_icon="📋", layout="wide")

from features.tabulation import render_tabulation
from core.utils import render_footer


def main():
    render_tabulation()
    render_footer()


if __name__ == "__main__":
    main()
