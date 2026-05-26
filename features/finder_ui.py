import streamlit as st
from core.matching import find_matching_tests
from core.data import rules, FIELDS
from features.widgets import render_latex, render_test_widget
from features.flowchart import build_tree, build_sunburst_chart
from features.glossary import render_glossary


def render_test_finder():
    """Render the Test Finder UI."""

    with st.sidebar:
        render_glossary()

    st.title("Statistical Test Finder")

    st.write(
        "Select your study characteristics to identify the appropriate statistical test."
    )

    if "results" not in st.session_state:
        st.session_state.results = None
    if "open_tests" not in st.session_state:
        st.session_state.open_tests = set()

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Research Objective")

        obj_opts = [
            "Comparison",
            "Association/Correlation",
            "Prediction",
            "Diagnostic Accuracy",
            "Survival Analysis",
        ]
        default_obj_idx = 0
        Objective = st.selectbox("What is your goal?", obj_opts, index=default_obj_idx)

        st.subheader("2. Variables")
        st.markdown("##### :green[Dependent Variable]")
        Dependent_Variable = st.selectbox(
            """Outcome / Target Variable / Y variable / Response Variable / Predicted Variable / Disease / Event / Output / Measured Variable / Result / Effect / Endpoint""",
            [
                "Binary/Dichotomous",
                "Categorical",
                "Ordinal",
                "Discrete",
                "Continuous",
                "Multiple Continuous",
                "Time-to-event",
            ],
        )
        st.markdown("##### :red[Independent Variable]")
        Independent_Variable = st.selectbox(
            """Predictor / Explanatory Variable / X variable / Grouping variable / Exposure / Intervention / Treatment / Risk Factor / Input / Covariate / Control Variable""",
            [
                "Binary/Dichotomous",
                "Categorical",
                "Ordinal",
                "Discrete",
                "Continuous",
                "Multiple Continuous",
                "None",
            ],
        )

        st.subheader("3. Experimental Design")

        Groups = st.selectbox(
            "Number of Groups",
            [
                "1",
                "2",
                "More than 2",
                "any",
            ],
        )

        Relation = st.selectbox(
            "Relationship Type",
            [
                "Independent",
                "Dependent",
                "any",
            ],
        )

        Distribution = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Non-normal",
                "any",
            ],
        )

        user_input = {
            "Objective": Objective,
            "Dependent_Variable": Dependent_Variable,
            "Independent_Variable": Independent_Variable,
            "Groups": Groups,
            "Relation": Relation,
            "Distribution": Distribution,
        }

        if st.button("Find My Test", use_container_width=True):
            st.session_state.results = find_matching_tests(user_input)
            st.session_state.open_tests = set()
            st.session_state.power_params = None

    with col_right:
        if st.session_state.results is not None:
            if st.session_state.results:
                st.success("Recommended Statistical Test(s):")

                for test in st.session_state.results:
                    rule = next((r for r in rules if r["name"] == test), None)
                    if rule:
                        is_open = test in st.session_state.open_tests
                        btn_label = f"▶ {test}" if not is_open else f"▼ {test}"

                        if st.button(
                            btn_label, key=f"btn_{test}", use_container_width=True
                        ):
                            if is_open:
                                st.session_state.open_tests.remove(test)
                            else:
                                st.session_state.open_tests.add(test)
                            st.rerun()

                        if test in st.session_state.open_tests:
                            if "Explanation" in rule:
                                st.markdown("## Explanation:")
                                st.markdown(rule["Explanation"])
                            if "Example" in rule:
                                st.markdown("## Example:")
                                st.markdown(rule["Example"])
                            if "Formula" in rule:
                                st.markdown("## Formula:")
                                render_latex(rule["Formula"])
                            if "Decision Rules" in rule:
                                st.markdown("## Decision Rules:")
                                st.info(rule["Decision Rules"])
                            if "Post-Hoc" in rule:
                                st.markdown("## Available Post-Hoc Tests:")
                                st.info(
                                    "\n".join(
                                        f"- {m.strip()}"
                                        for m in rule["Post-Hoc"].split(",")
                                    )
                                )
                            render_test_widget(test)

            else:
                st.error(
                    "No matching statistical test found. Try adjusting your selections."
                )
        else:
            st.info("Results will appear here once you click 'Find My Test'.")

    st.divider()
    st.header("Interactive Statistical Flowchart")
    
    if "show_flowchart" not in st.session_state:
        st.session_state.show_flowchart = False
    
    if not st.session_state.show_flowchart:
        st.info("Click below to load the interactive flowchart (Accordion View + Sunburst Chart)")
        if st.button("📊 Load Flowchart", use_container_width=True):
            st.session_state.show_flowchart = True
            st.rerun()
    else:
        st.write(
            "Expand the branches below to navigate statistical test selection visually."
        )

        tab_acc, tab_sun = st.tabs(["Accordion View", "Sunburst Chart"])

        with tab_acc:
            st.write(
                "Expand the branches below to navigate statistical test selection visually."
            )
            build_tree(rules, FIELDS, user_input)

        with tab_sun:
            st.write(
                "A holistic view of the statistical universe. Click on a slice to zoom in."
            )
            build_sunburst_chart(rules, FIELDS)

    st.markdown("---")

    footer_html = """
<div style="padding: 20px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 20px; text-align: center;">
    <h3 style="margin-top: 0; color: #4CAF50;">Developed By</h3>
    <p style="font-size: 1.2em; margin-bottom: 5px;"><strong>Dr. Muhammad Nabeel Shaesha</strong></p>
    <p style="margin: 0; opacity: 0.8;">Teaching Assistant at the Prosthodontics Department, PUA</p>
    <p style="margin: 0; opacity: 0.8;">Currently enrolled in Masters of Prosthodontics and Implantology Program, PUA</p>
    <div style="margin-top: 20px;">
        <p style="font-size: 0.9em; opacity: 0.7; margin-bottom: 10px;">Built with the help of:</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
            <div style="border: 2px solid #CA6180; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #CA6180;">
                Gemma 4
            </div>
            <div style="border: 2px solid #4B9DA9; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #4B9DA9;">
                OpenCode
            </div>
            <div style="border: 2px solid #8E24AA; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #8E24AA;">
                GeminiCLI
            </div>
            <div style="border: 2px solid #10a37f; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #10a37f;">
                ChatGPT
            </div>
        </div>
        <p style="font-size: 0.9em; opacity: 0.7; margin-bottom: 10px;"><br /> Acknowledgment to my professors who taught me biosatistics and research methodology</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Inas Karawia
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Maha Adel
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Hamida Abu Bakr
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Hadeya Abdel Hamid
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Nancy Bedwany
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; opacity: 0.6; font-size: 0.8em;">
    <p><strong>⚠️ Disclaimer</strong></p>
    <p>This tool is intended for <strong>educational and informational purposes only</strong>. 
    While it follows standard statistical guidelines, it does not account for all possible 
    complexities in study design (e.g., nesting, interaction effects, or specific data anomalies). 
    Recommendations should be verified by a qualified biostatistician or through standard 
    statistical literature before being used for clinical or formal research purposes.</p>
    <p>© 2026 Statistical Test Finder. Built with Streamlit.</p>
</div>
"""
    st.markdown(footer_html, unsafe_allow_html=True)
