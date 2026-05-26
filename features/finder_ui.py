import streamlit as st
from core.matching import find_matching_tests
from core.data import rules, FIELDS
from features.widgets import render_latex, render_test_widget
from features.glossary import render_glossary


def _get_tests_by_objective():
    from collections import defaultdict

    by_obj = defaultdict(list)
    for rule in rules:
        obj = rule.get("Objective", "Unknown")
        if isinstance(obj, list):
            for o in obj:
                by_obj[o].append(rule["name"])
        else:
            by_obj[obj].append(rule["name"])

    for obj in by_obj:
        by_obj[obj] = sorted(set(by_obj[obj]))

    return dict(sorted(by_obj.items()))


def _open_test_directly(test_name):
    rule = next((r for r in rules if r["name"] == test_name), None)
    if rule:
        st.session_state.results = [test_name]
        st.session_state.open_tests = {test_name}
        st.rerun()


def render_all_tests_section():
    st.divider()
    st.header("All Statistical Tests")
    st.info("Click on any test name to view it directly in the finder.")

    by_obj = _get_tests_by_objective()
    objectives = list(by_obj.keys())

    tabs = st.tabs(objectives)

    for tab_idx, obj in enumerate(objectives):
        with tabs[tab_idx]:
            test_names = by_obj[obj]
            total = len(test_names)

            st.markdown(f"**{total} tests** for *{obj}*:")

            test_cols = st.columns(3)
            for i, test_name in enumerate(test_names):
                col_idx = i % 3
                with test_cols[col_idx]:
                    btn_key = (
                        f"alltest_{obj.replace(' ', '_')}_{test_name.replace(' ', '_')}"
                    )
                    if st.button(
                        f"📌 {test_name}", key=btn_key, use_container_width=True
                    ):
                        _open_test_directly(test_name)


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

    render_all_tests_section()
