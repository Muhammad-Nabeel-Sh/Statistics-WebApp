import streamlit as st

st.set_page_config(page_title="Power Analysis & Sample Size", page_icon="⚡", layout="wide")

from features.power_ui import _render_power_analysis
from core.utils import render_footer


def main():
    _render_power_analysis()
    render_footer()


if __name__ == "__main__":
    main()
