import streamlit as st

RULES = [
    # Comparison Tests
    {
        "name": "One-sample t-test",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "None",
        "groups": "1",
        "relation_type": "any",
        "distribution": "Normal",
    },
    {
        "name": "One-sample z-test",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "None",
        "groups": "1",
        "relation_type": "any",
        "distribution": "Normal",
    },
    {
        "name": "One-sample Proportion Test",
        "objective": "Comparison",
        "dep_var": "Binary/Dichotomous" or "Categorical",
        "ind_var": "None",
        "groups": "1",
        "relation_type": "any",
        "distribution": "Normal" or "Non-normal" or "any",
    },
    {
        "name": "Student's t-test (InDependent)",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "InDependent",
        "distribution": "Normal",
    },
    {
        "name": "Paired t-test",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "Dependent",
        "distribution": "Normal",
    },
    {
        "name": "One-way ANOVA",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "InDependent",
        "distribution": "Normal",
    },
    {
        "name": "Wilcoxon Signed-Rank Test",
        "objective": "Comparison",
        "dep_var": "ordinal",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "Dependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Mann-Whitney U Test",
        "objective": "Comparison",
        "dep_var": "ordinal",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "InDependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Kruskal-Wallis Test",
        "objective": "Comparison",
        "dep_var": "ordinal",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "InDependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Repeated Measures ANOVA",
        "objective": "Comparison",
        "dep_var": "Continuous",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "Dependent",
        "distribution": "Normal",
    },
    {
        "name": "MANOVA",
        "objective": "Comparison",
        "dep_var": "Multiple Continuous",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "InDependent",
        "distribution": "Normal",
    },
    {
        "name": "Friedman Test",
        "objective": "Comparison",
        "dep_var": "ordinal",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "Dependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Permutation MANOVA or Non-Parametric MANOVA",
        "objective": "Comparison",
        "dep_var": "Multiple Continuous",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "InDependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Goodness-of-Fit Test",
        "objective": "Comparison",
        "dep_var": "Categorical",
        "ind_var": "None",
        "groups": "1",
        "relation_type": "any",
        "distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Test",
        "objective": "Comparison",
        "dep_var": "Categorical",
        "ind_var": "Categorical",
        "groups": "any",
        "relation_type": "InDependent",
        "distribution": "Non-normal",
    },
    {
        "name": "McNemar's Test",
        "objective": "Comparison",
        "dep_var": "Categorical",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "Dependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Cochen's Q Test",
        "objective": "Comparison",
        "dep_var": "Categorical",
        "ind_var": "Categorical",
        "groups": ">2",
        "relation_type": "Dependent",
        "distribution": "Non-normal",
    },
    {
        "name": "Fisher's Exact Test",
        "objective": "Comparison",
        "dep_var": "Categorical",
        "ind_var": "Categorical",
        "groups": "2",
        "relation_type": "InDependent",
        "distribution": "Non-normal",
    },
    # Association/Correlation Tests
    {
        "name": "Pearson Correlation",
        "objective": "Association/Correlation",
        "dep_var": "Continuous",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "Normal",
    },
    {
        "name": "Spearman Rank Correlation",
        "objective": "Association/Correlation",
        "dep_var": "ordinal",
        "ind_var": "ordinal",
        "groups": "any",
        "relation_type": "any",
        "distribution": "Non-normal",
    },
    {
        "name": "Chi-Square Test of Independence",
        "objective": "Association/Correlation",
        "dep_var": "Categorical",
        "ind_var": "Categorical",
        "groups": "any",
        "relation_type": "any",
        "distribution": "Non-normal",
    },
    {
        "name": "Point-Biserial Correlation",
        "objective": "Association/Correlation",
        "dep_var": "Continuous",
        "ind_var": "Binary/Dichotomous",
        "groups": "2",
        "relation_type": "any",
        "distribution": "Normal",
    },
    # Prediction Tests
    {
        "name": "Simple Linear Regression",
        "objective": "Prediction",
        "dep_var": "Continuous",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "Normal",
    },
    {
        "name": "Multiple Linear Regression",
        "objective": "Prediction",
        "dep_var": "Continuous",
        "ind_var": "Multiple Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "Normal",
    },
    {
        "name": "Logistic Regression",
        "objective": "Prediction",
        "dep_var": "Binary/Dichotomous" or "Categorical",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "any",
    },
    {
        "name": "Multinomial Logistic Regression",
        "objective": "Prediction",
        "dep_var": "Categorical",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "any",
    },
    {
        "name": "Ordinal Logistic Regression",
        "objective": "Prediction",
        "dep_var": "Ordinal",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "any",
    },
    {
        "name": "Poisson Regression",
        "objective": "Prediction",
        "dep_var": "Discrete",
        "ind_var": "Continuous",
        "groups": "any",
        "relation_type": "any",
        "distribution": "any",
    },
]


def find_best_test(user_params):
    """
    Compares user input against the RULES database.
    The 'score' is how many criteria the rule matches.
    """
    matches = []

    for rule in RULES:
        score = 0
        # Check each criterion in the rule
        for key, value in rule.items():
            if key in user_params:
                if value == "any" or value == user_params[key]:
                    score += 1
                else:
                    score = 0  # Mismatch found, invalidate this rule
                    break
            else:
                # If rule requires a parameter the user didn't provide, it's a partial match
                pass

        if score > 0:
            matches.append((rule["name"], score))

    # Sort matches by highest score (most specific match first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def main():
    import streamlit as st  # Assuming you use streamlit for the UI

    st.title("🔬 Statistical Test Finder")
    st.write(
        "Select your variables below to find the most appropriate statistical test."
    )

    st.subheader("1. Research Objective")
    objective = st.selectbox(
        "What is your goal?", ["Comparison", "Association/Correlation", "Predicition"]
    )

    st.subheader("2. Variables")
    dep_var = st.selectbox(
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
    ind_var = st.selectbox(
        "InDependent Variable Type",
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

    st.subheader("3. Experimental Design and Distribution")
    groups = st.selectbox("Number of Groups / Levels", ["1", "2", "More than 2", "any"])
    relation_type = st.selectbox(
        "Relationship Type", ["InDependent", "Dependent", "any"]
    )
    Distribution = st.selectbox("Distribution", ["Normal", "Non-normal", "any"])

    # User's input packaged for the engine
    user_input = {
        "objective": objective,
        "dep_var": dep_var,
        "ind_var": ind_var,
        "groups": groups,
        "relation_type": relation_type,
        "distribution": Distribution,
    }

    if st.button("Find My Test"):
        results = find_best_test(user_input)

        if results:
            st.success(f"Found {len(results)} possible match(es)!")
            for name, score in results:
                st.markdown(f"### 🎯 {name}")
                st.caption(f"Match confidence score: {score}")
        else:
            st.error(
                "No exact match found. Try broadening your criteria (e.g., set a value to 'any')."
            )


if __name__ == "__main__":
    main()
