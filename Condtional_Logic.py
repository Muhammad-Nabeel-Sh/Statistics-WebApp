import streamlit as st


def get_statistical_test(
    objective, outcome_type, predictor_type, groups, relationship, distribution
):
    """
    Logic to determine the appropriate statistical test.
    """
    if objective == "Comparison":
        if outcome_type == "Continuous":
            if predictor_type == "Categorical":
                if relationship == "Independent":
                    if groups == "2 groups":
                        return (
                            "Independent Samples t-test"
                            if distribution == "Normal"
                            else "Mann-Whitney U Test"
                        )
                    else:
                        return (
                            "One-way ANOVA"
                            if distribution == "Normal"
                            else "Kruskal-Wallis Test"
                        )
                elif relationship == "Paired":
                    if groups == "2 groups":
                        return (
                            "Paired Samples t-test"
                            if distribution == "Normal"
                            else "Wilcoxon Signed-Rank Test"
                        )
                    else:
                        return (
                            "Repeated Measures ANOVA"
                            if distribution == "Normal"
                            else "Friedman Test"
                        )
                elif predictor_type == "Continuous":
                    return "Correlation/Regression analysis is more appropriate for continuous predictors."

            elif predictor_type == "Categorical":
                if relationship == "Independent":
                    return "Chi-square Test"
                else:
                    return "McNemar's Test"

        elif outcome_type == "Categorical":
            if predictor_type == "Categorical":
                if relationship == "Independent":
                    return "Chi-square Test of Independence"
                else:
                    return "McNemar's Test"
            else:
                return "Logistic Regression (Categorical outcome, Continuous predictor)"

    elif objective == "Association / Correlation":
        if outcome_type == "Continuous" and predictor_type == "Continuous":
            return (
                "Pearson Correlation"
                if distribution == "Normal"
                else "Spearman's Rank Correlation"
            )
        elif outcome_type == "Categorical" or predictor_type == "Categorical":
            return "Chi-square Test of Independence"
        else:
            return "Point-Biserial Correlation"

    elif objective == "Prediction":
        if outcome_type == "Continuous" and predictor_type == "Continuous":
            return "Simple Linear Regression"
        elif outcome_type == "Continuous" and predictor_type == "Categorical":
            return "Multiple Linear Regression (using dummy variables)"
        elif outcome_type == "Categorical" and predictor_type == "Continuous":
            return "Logistic Regression"
        else:
            return "Generalized Linear Model (GLM)"

    return "No specific test found for the selected parameters. Please refine your selection."


# --- Streamlit UI ---
st.set_page_config(page_title="Biostatistician Assistant", page_icon="🧬")

st.title("🧬 Biostatistical Test Selector")
st.markdown("Identify the correct statistical test for your research design.")

st.header("Step 1: Define Your Study Parameters")
st.subheader("Variables")
outcome_type = st.selectbox(
    "Outcome (Dependent) Variable Type", ["Continuous", "Categorical"]
)
predictor_type = st.selectbox(
    "Predictor (Independent) Variable Type", ["Continuous", "Categorical"]
)

st.divider()

st.header("Step 2: Research Objective")
objective = st.selectbox(
    "Objective", ["Comparison", "Association / Correlation", "Prediction"]
)

st.divider()

st.header("Step 3: Experimental Design & Distribution")

col1, col2 = st.columns(2)

with col1:
    if objective == "Comparison":
        groups = st.selectbox("Number of Groups", ["2 groups", "More than 2 groups"])
        relationship = st.selectbox(
            "Relationship between groups", ["Independent", "Paired"]
        )
    else:
        groups = "N/A"
        relationship = "N/A"
        st.info(
            "Design details (groups/relationship) are primarily used for Comparison objectives."
        )

with col2:
    if objective in ["Comparison", "Association / Correlation"]:
        distribution = st.selectbox("Distribution", ["Normal", "Non-normal"])
    else:
        distribution = "N/A"
        st.info(
            "Distribution is primarily used for Parametric vs Non-parametric tests."
        )

st.divider()

# Final Result Calculation
result = get_statistical_test(
    objective, outcome_type, predictor_type, groups, relationship, distribution
)

st.header("Recommended Test:")
st.success(result)

st.markdown("""
---
**How to use this app:**
1. Define your **Variables** in the sidebar.
2. Select your **Research Objective**.
3. Configure the **Design and Distribution** in the main area.
4. The recommended test will appear below!
""")
