# Statistical Test Finder

🔬 **Statistical Test Finder** is a comprehensive Streamlit-based web application designed to help researchers, students, and data scientists identify the appropriate statistical test for their study design and data characteristics.

## 🚀 Features

- **Interactive Test Matching**: Select your research objective, variable types, and experimental design to receive tailored statistical test recommendations.
- **Two-Column Layout**: Optimized for wide screens, keeping input parameters on the left and results/educational content on the right for a seamless user experience.
- **Comprehensive Test Library**: Covers over 30 statistical tests, including:
    - **Comparison Tests**: t-tests, ANOVA, MANOVA, Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman, Chi-Square, etc.
    - **Association/Correlation Tests**: Pearson, Spearman, Point-Biserial, etc.
    - **Prediction Tests**: Simple/Multiple Linear Regression, Logistic Regression (Multinomial/Ordinal), Poisson Regression, etc.
- **Educational Content**: Each recommended test comes with:
    - Detailed **Explanations** of its purpose and assumptions.
    - Practical **Examples** of use cases.
    - Mathematical **Formulas** rendered in LaTeX.
- **Interactive Visualizations**: Real-time interactive widgets for most tests using synthetic data. Adjust sliders to see how parameters (like mean shift, noise, or correlation) affect test statistics and p-values.
- **Visual Flowchart**: An interactive tree-based flowchart to explore the statistical decision-making process visually.

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Statistics WebApp"
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install streamlit numpy pandas matplotlib plotly scipy statsmodels
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

- `app.py`: The core application containing the matching engine, UI, and interactive widgets.
- `Condtional_Logic.py`: A simplified version of the test selection logic and UI.

## 🧪 How It Works

The app uses a rule-based matching engine defined in `app.py`. Each statistical test is defined as a rule with specific criteria:
- **Objective**: Comparison, Association/Correlation, or Prediction.
- **Dependent Variable**: Continuous, Categorical, Ordinal, etc.
- **Independent Variable**: Type of predictor or grouping variable.
- **Groups**: Number of groups involved (1, 2, or more).
- **Relation**: Whether the samples are independent or dependent (paired).
- **Distribution**: Normal or Non-normal.

When you make selections in the UI, the engine filters the rules to find the most appropriate test(s) for your specific scenario.
