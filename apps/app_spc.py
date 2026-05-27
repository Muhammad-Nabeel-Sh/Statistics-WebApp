import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="SPC Control Charts",
    page_icon="📊",
    layout="wide",
)

from features.control_charts import render_control_charts
from core.utils import render_footer


def main():
    render_control_charts()
    render_footer()


if __name__ == "__main__":
    main()
