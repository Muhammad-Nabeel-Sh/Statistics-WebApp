import streamlit as st

st.set_page_config(page_title="Solved Examples", page_icon="📚", layout="wide")

from features.solved_examples import render_solved_examples
from core.utils import render_footer


def main():
    render_solved_examples()
    render_footer()


if __name__ == "__main__":
    main()
