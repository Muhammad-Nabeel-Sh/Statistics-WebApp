import streamlit as st

st.set_page_config(page_title="Probability Distributions", page_icon="🎲", layout="wide")

from features.distributions import render_distributions
from core.utils import render_footer


def main():
    render_distributions()
    render_footer()


if __name__ == "__main__":
    main()
