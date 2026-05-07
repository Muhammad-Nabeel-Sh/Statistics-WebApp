import streamlit as st

rules = [
    # Comparison Tests
    {
        "name": "One-sample t-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Normal",
        "Explanation": "One-Sample t-test This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean. It assumes that the data is continuous and follows a normal distribution. It is typically used when comparing a clinical measurement (like blood pressure) against a standard clinical threshold. ",
        "Example": "Example: A researcher wants to test if the average systolic blood pressure of a group of patients is significantly different from the standard threshold of 120 mmHg. The researcher collects blood pressure readings from 30 patients and performs a one-sample t-test to compare the sample mean against the known population mean of 120 mmHg.",
        "Formula": r"$$ t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}$$ where X̄ is the sample mean, μ is the population mean, s is the sample standard deviation, n is the sample size and $\frac{s}{\sqrt{n}}$ is the standard error of the mean.",
    },
    {
        "name": "One-sample z-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Normal",
    },
    {
        "name": "One-sample Proportion Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "One-sample Wilcoxon Signed-Rank Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Non-normal",
    },
    {
        "name": "Student's t-test (Independent)",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Normal",
    },
    {
        "name": "Welch's t-test (Independent, Unequal Variances)",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Normal",
    },
    {
        "name": "Paired t-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "Normal",
    },
    {
        "name": "One-way ANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
    },
    {
        "name": "Wilcoxon Signed-Rank Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "Non-normal",
    },
    {
        "name": "Mann-Whitney U Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
    },
    {
        "name": "Kruskal-Wallis Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
    },
    {
        "name": "Repeated Measures ANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": "Normal",
    },
    {
        "name": "MANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Multiple Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
    },
    {
        "name": "Friedman Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": "Non-normal",
    },
    {
        "name": "Permutation MANOVA or Non-Parametric MANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Multiple Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Goodness-of-Fit Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": "Independent",
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    {
        "name": "McNemar's Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    {
        "name": "Cochran's Q Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    {
        "name": "Fisher's Exact Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    # Association/Correlation Tests
    {
        "name": "Pearson Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Normal",
    },
    {
        "name": "Spearman Rank Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Ordinal", "Continuous"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Test of Independence",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "Categorical",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    {
        "name": "Point-Biserial Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
    },
    # Prediction Tests
    {
        "name": "Simple Linear Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "Multiple Linear Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Multiple Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "Multinomial Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "Ordinal Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Ordinal",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
    {
        "name": "Poisson Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Discrete",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
    },
]


# =========================
# MATCHING ENGINE
# =========================

CRITERIA_FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]


def matches_rule(user_input, rule):

    for key in CRITERIA_FIELDS:

        rule_val = rule.get(key)
        if rule_val is None:
            continue

        user_val = user_input.get(key)

        # Rule accepts anything
        if rule_val == "any":
            continue

        # Handle LISTS in rules
        if isinstance(rule_val, list):
            if user_val not in rule_val:
                return False

        # Handle normal strings
        else:
            if user_val != rule_val:
                return False

    return True


def find_matching_tests(user_input):

    matches = []

    for rule in rules:
        if matches_rule(user_input, rule):
            matches.append(rule["name"])

    return matches


# =========================
# STREAMLIT UI
# =========================
def main():

    st.set_page_config(page_title="Statistical Test Finder")

    st.title("🔬 Statistical Test Finder")

    st.write(
        "Select your study characteristics to identify the appropriate statistical test."
    )

    # =========================
    # RESEARCH Objective
    # =========================
    st.subheader("1. Research Objective")

    Objective = st.selectbox(
        "What is your goal?",
        [
            "Comparison",
            "Association/Correlation",
            "Prediction",
        ],
    )

    # =========================
    # VARIABLES
    # =========================
    st.subheader("2. Variables")

    Dependent_Variable = st.selectbox(
        "Dependent Variable Type",
        [
            "Binary/Dichotomous",
            "Categorical",
            "Ordinal",
            "Discrete",
            "Continuous",
            "Multiple Continuous",
        ],
    )

    Independent_Variable = st.selectbox(
        "Independent Variable Type",
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

    # =========================
    # DESIGN
    # =========================
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

    # =========================
    # USER INPUT OBJECT
    # =========================
    user_input = {
        "Objective": Objective,
        "Dependent_Variable": Dependent_Variable,
        "Independent_Variable": Independent_Variable,
        "Groups": Groups,
        "Relation": Relation,
        "Distribution": Distribution,
    }

    # =========================
    # FIND TEST
    # =========================
    if st.button("Find My Test"):

        results = find_matching_tests(user_input)

        if results:

            st.success("Recommended Statistical Test(s):")

            for test in results:
                rule = next((r for r in rules if r["name"] == test), None)
                if rule:
                    with st.expander(f"✅ {test}"):
                        if "Explanation" in rule:
                            st.markdown("## Explanation:")
                            st.markdown(rule["Explanation"])
                        if "Example" in rule:
                            st.markdown("## Example:")
                            st.markdown(rule["Example"])
                        if "Formula" in rule:
                            st.markdown("## Formula:")
                            render_latex(rule["Formula"])
                        render_test_widget(test)

        else:

            st.error(
                "No matching statistical test found. Try adjusting your selections."
            )
    # =========================
    # FLOWCHART MODE
    # =========================

    st.divider()

    st.header("🌳 Interactive Statistical Flowchart")

    st.write(
        "Expand the branches below to navigate statistical test selection visually."
    )

    build_tree(rules, FIELDS)


# =========================
# INTERACTIVE WIDGETS
# =========================


def render_latex(formula_text):
    """Render LaTeX formulas from text with $$ delimiters."""
    import re

    last_end = 0
    for match in re.finditer(r'\$\$(.*?)\$\$', formula_text, re.DOTALL):
        # Text before this match
        text_before = formula_text[last_end:match.start()]
        if text_before.strip():
            st.markdown(text_before)

        # The LaTeX block (without $$ delimiters)
        latex_code = match.group(1).strip()
        st.latex(latex_code)

        last_end = match.end()

    # Text after the last match
    text_after = formula_text[last_end:]
    if text_after.strip():
        st.markdown(text_after)


def render_test_widget(test_name):
    """Render interactive widget for specific statistical test."""
    if test_name == "One-sample t-test":
        st.markdown("**Interactive Calculator:**")
        col1, col2 = st.columns(2)
        with col1:
            sample_mean = st.number_input(
                "Sample Mean (x̄)", value=125.0, key="t_sample_mean"
            )
            pop_mean = st.number_input(
                "Population Mean (μ₀)", value=120.0, key="t_pop_mean"
            )
        with col2:
            sample_std = st.number_input(
                "Sample Std Dev (s)", value=10.0, min_value=0.01, key="t_sample_std"
            )
            sample_size = st.number_input(
                "Sample Size (n)", value=30, min_value=2, key="t_sample_size"
            )
        if sample_size > 0 and sample_std > 0:
            std_error = sample_std / (sample_size**0.5)
            t_stat = (sample_mean - pop_mean) / std_error
            st.markdown(f"**Standard Error:** {std_error:.4f}")
            st.markdown(f"**t-statistic:** {t_stat:.4f}")
            st.markdown(f"**Degrees of Freedom:** {sample_size - 1}")

    elif test_name == "One-sample z-test":
        st.markdown("**Interactive Calculator:**")
        col1, col2 = st.columns(2)
        with col1:
            sample_mean = st.number_input(
                "Sample Mean (x̄)", value=125.0, key="z_sample_mean"
            )
            pop_mean = st.number_input(
                "Population Mean (μ₀)", value=120.0, key="z_pop_mean"
            )
        with col2:
            pop_std = st.number_input(
                "Population Std Dev (σ)", value=10.0, min_value=0.01, key="z_pop_std"
            )
            sample_size = st.number_input(
                "Sample Size (n)", value=30, min_value=2, key="z_sample_size"
            )
        if sample_size > 0 and pop_std > 0:
            std_error = pop_std / (sample_size**0.5)
            z_stat = (sample_mean - pop_mean) / std_error
            st.markdown(f"**Standard Error:** {std_error:.4f}")
            st.markdown(f"**z-statistic:** {z_stat:.4f}")

    else:
        st.info("Interactive widget coming soon for this test.")


# =========================
# FLOWCHART VIEW
# =========================

FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]


def build_tree(rule_subset, fields, level=0):

    # No more fields → show tests
    if not fields:

        for rule in rule_subset:
            st.success(f"✅ {rule['name']}")

        return

    current_field = fields[0]
    # Skip meaningless all-any branches
    all_any = all(rule[current_field] == "any" for rule in rule_subset)

    if all_any:
        build_tree(rule_subset, fields[1:], level + 1)
        return

    # Group rules by current field
    grouped = {}

    for rule in rule_subset:

        value = rule[current_field]

        if isinstance(value, list):

            for item in value:
                grouped.setdefault(item, []).append(rule)

        else:
            grouped.setdefault(value, []).append(rule)

    # Render Groups
    for value, subrules in grouped.items():

        # Pretty label
        if isinstance(value, tuple):
            label = " OR ".join(value)
        else:
            label = str(value)

        with st.expander(f"{current_field}: {label}"):

            build_tree(subrules, fields[1:], level + 1)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
