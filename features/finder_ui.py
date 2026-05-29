import streamlit as st
from core.matching import find_matching_tests
from core.data import rules, FIELDS
from features.widgets import render_latex, render_test_widget
from features.glossary import render_glossary


def _categorize_tests():
    """Categorize tests by design type and parametric/non-parametric."""
    from collections import defaultdict

    def _is_parametric(rule):
        """Check if test is primarily parametric."""
        dist = rule.get("Distribution", "any")
        if isinstance(dist, str):
            return dist == "Normal"
        return "Normal" in dist and "Non-normal" not in dist

    def _is_nonparametric(rule):
        """Check if test is primarily non-parametric."""
        dist = rule.get("Distribution", "any")
        if isinstance(dist, str):
            return dist == "Non-normal"
        return "Non-normal" in dist and "Normal" not in dist

    def _is_any_dist(rule):
        """Check if test works with any distribution."""
        dist = rule.get("Distribution", "any")
        if isinstance(dist, str):
            return dist == "any"
        return "any" in dist or ("Normal" in dist and "Non-normal" in dist)

    def _get_groups(groups):
        if isinstance(groups, list):
            if "1" in groups:
                return "1"
            elif "2" in groups:
                return "2"
            elif "More than 2" in groups:
                return "More than 2"
            return "any"
        return groups

    def _get_relation(relation):
        if isinstance(relation, list):
            if "Dependent" in relation:
                return "Dependent"
            elif "Independent" in relation:
                return "Independent"
            return "any"
        return relation

    categories = defaultdict(list)
    category_order = []

    for rule in rules:
        groups = _get_groups(rule.get("Groups", "any"))
        relation = _get_relation(rule.get("Relation", "any"))
        objective = rule.get("Objective", "Unknown")

        if objective == "Association/Correlation":
            cat_key = ("Correlation & Association",)
        elif objective == "Prediction":
            cat_key = ("Regression & Prediction",)
        elif objective == "Survival Analysis":
            cat_key = ("Survival Analysis",)
        elif objective == "Diagnostic Accuracy":
            cat_key = ("Diagnostic Accuracy",)
        elif groups == "1":
            if _is_parametric(rule):
                cat_key = ("One-sample", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("One-sample", "Non-parametric")
            else:
                cat_key = ("One-sample", "Any Distribution")
        elif groups == "2" and relation == "Independent":
            if _is_parametric(rule):
                cat_key = ("Two-sample (Independent)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Two-sample (Independent)", "Non-parametric")
            else:
                cat_key = ("Two-sample (Independent)", "Any Distribution")
        elif groups == "2" and relation == "Dependent":
            if _is_parametric(rule):
                cat_key = ("Two-sample (Dependent/Paired)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Two-sample (Dependent/Paired)", "Non-parametric")
            else:
                cat_key = ("Two-sample (Dependent/Paired)", "Any Distribution")
        elif groups == "More than 2" and relation == "Independent":
            if _is_parametric(rule):
                cat_key = ("Multi-sample (Independent)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Multi-sample (Independent)", "Non-parametric")
            else:
                cat_key = ("Multi-sample (Independent)", "Any Distribution")
        elif groups == "More than 2" and relation == "Dependent":
            if _is_parametric(rule):
                cat_key = ("Multi-sample (Dependent/Paired)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Multi-sample (Dependent/Paired)", "Non-parametric")
            else:
                cat_key = ("Multi-sample (Dependent/Paired)", "Any Distribution")
        else:
            if _is_any_dist(rule):
                cat_key = ("Other Tests", "Flexible/Any Design")
            elif _is_parametric(rule):
                cat_key = ("Other Tests", "Parametric")
            else:
                cat_key = ("Other Tests", "Non-parametric")

        if cat_key not in categories:
            category_order.append(cat_key)
        categories[cat_key].append(rule["name"])

    for cat_key in categories:
        categories[cat_key] = sorted(set(categories[cat_key]))

    def _category_sort_key(key):
        priority = {
            ("One-sample", "Parametric"): 100,
            ("One-sample", "Non-parametric"): 110,
            ("One-sample", "Any Distribution"): 120,
            ("Two-sample (Independent)", "Parametric"): 200,
            ("Two-sample (Independent)", "Non-parametric"): 210,
            ("Two-sample (Independent)", "Any Distribution"): 220,
            ("Two-sample (Dependent/Paired)", "Parametric"): 300,
            ("Two-sample (Dependent/Paired)", "Non-parametric"): 310,
            ("Two-sample (Dependent/Paired)", "Any Distribution"): 320,
            ("Multi-sample (Independent)", "Parametric"): 400,
            ("Multi-sample (Independent)", "Non-parametric"): 410,
            ("Multi-sample (Independent)", "Any Distribution"): 420,
            ("Multi-sample (Dependent/Paired)", "Parametric"): 500,
            ("Multi-sample (Dependent/Paired)", "Non-parametric"): 510,
            ("Multi-sample (Dependent/Paired)", "Any Distribution"): 520,
            ("Correlation & Association",): 600,
            ("Regression & Prediction",): 700,
            ("Survival Analysis",): 800,
            ("Diagnostic Accuracy",): 900,
            ("Other Tests", "Parametric"): 1000,
            ("Other Tests", "Non-parametric"): 1010,
            ("Other Tests", "Flexible/Any Design"): 1020,
        }
        return priority.get(key, 9999)

    category_order.sort(key=_category_sort_key)

    return categories, category_order


def render_all_tests_section():
    st.divider()
    st.header("All Statistical Tests")
    st.info("Click on any test name to view it directly in the finder.")

    categories, category_order = _categorize_tests()

    prev_main_cat = None

    for cat_key in category_order:
        main_cat = cat_key[0]
        sub_cat = cat_key[1] if len(cat_key) > 1 else None

        if main_cat != prev_main_cat:
            st.divider()
            st.subheader(main_cat)
            prev_main_cat = main_cat

        if sub_cat:
            st.markdown(f"**{sub_cat}:**")
        else:
            st.markdown("")

        test_names = categories[cat_key]
        total = len(test_names)

        test_cols = st.columns(3)
        for i, test_name in enumerate(test_names):
            col_idx = i % 3
            with test_cols[col_idx]:
                btn_key = f"alltest_cat_{main_cat.replace(' ', '_')}_{test_name.replace(' ', '_')}"
                if st.button(f"📌 {test_name}", key=btn_key, use_container_width=True):
                    _open_test_directly(test_name)


def _open_test_directly(test_name):
    """Open a test directly in the finder's right panel."""
    st.session_state.results = [test_name]
    st.session_state.open_tests = {test_name}
    st.rerun()


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
