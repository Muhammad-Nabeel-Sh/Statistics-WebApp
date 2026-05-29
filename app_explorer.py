import streamlit as st

st.set_page_config(page_title="Graph Explorer", page_icon="📈", layout="wide")

from features.graph_explorer import render_graph_explorer
from core.utils import render_footer


def main():
    render_graph_explorer()
    render_footer()


if __name__ == "__main__":
    main()
