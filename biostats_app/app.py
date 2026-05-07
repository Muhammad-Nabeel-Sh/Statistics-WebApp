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
        "name": "Student's t-test (Independent)",
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
def matches_rule(user_input, rule):

    for key, rule_val in rule.items():

        # Skip test name
        if key == "name":
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
                st.write(f"✅ {test}")

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
