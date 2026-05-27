import os
import streamlit as st
import pandas as pd
import numpy as np


_BUILTIN_DATASETS = {}

# ──────────────────────────────────────────────
# One-sample
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Iris — Petal Width (Setosa)"] = {
    "description": "Petal width measurements for Iris setosa (Fisher 1936). Use for one-sample tests against a hypothesized mean.",
    "source": "Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.",
    "test_types": ["One-sample t-test", "One-sample Wilcoxon Signed-Rank Test", "Sign Test (One-sample)"],
    "csv": """petal_width
0.2
0.2
0.2
0.2
0.2
0.4
0.3
0.2
0.2
0.1
0.2
0.2
0.1
0.1
0.2
0.4
0.4
0.3
0.3
0.3
0.2
0.4
0.2
0.5
0.2
0.2
0.4
0.2
0.2
0.2
0.2
0.4
0.1
0.2
0.2
0.2
0.2
0.1
0.2
0.2
0.3
0.3
0.2
0.6
0.4
0.3
0.2
0.2
0.2
0.2
""",
}

_BUILTIN_DATASETS["Galton — Adult Heights"] = {
    "description": "Adult child heights from Galton's 1885 family dataset on heredity. Good for one-sample tests on height.",
    "source": "Galton, F. (1886). Regression towards mediocrity in hereditary stature.",
    "test_types": ["One-sample t-test", "One-sample Wilcoxon Signed-Rank Test", "One-sample z-test", "Sign Test (One-sample)"],
    "csv": """height
73.2
69.2
69.0
69.0
69.0
68.0
67.8
67.0
66.8
66.0
65.5
65.5
65.0
65.0
64.5
64.0
64.0
63.5
63.5
63.0
62.8
62.5
62.0
62.0
61.5
61.0
60.5
60.0
60.0
59.5
59.0
58.5
58.2
58.0
57.5
57.0
56.5
56.0
55.0
54.0
""",
}

# ──────────────────────────────────────────────
# Two-sample (Independent)
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["PlantGrowth — Control vs Treatment 1"] = {
    "description": "Dried weight of plants under control and treatment 1 conditions (Dobson 1983). Two independent groups.",
    "source": "Dobson, A. J. (1983). An Introduction to Statistical Modelling.",
    "test_types": [
        "Student's t-test (Independent)", "Welch's t-test (Independent)",
        "Mann-Whitney U Test", "F-Test for Two Variances",
    ],
    "csv": """group,weight
Control,4.17
Control,5.58
Control,5.18
Control,6.11
Control,4.50
Control,4.61
Control,5.17
Control,4.53
Control,5.33
Control,5.14
Treatment1,4.81
Treatment1,4.17
Treatment1,4.41
Treatment1,3.59
Treatment1,5.87
Treatment1,3.83
Treatment1,6.03
Treatment1,4.89
Treatment1,4.32
Treatment1,4.69
""",
}

_BUILTIN_DATASETS["Iris — Sepal Length (Setosa vs Versicolor)"] = {
    "description": "Sepal length for two iris species. Classic two-group comparison (Fisher 1936).",
    "source": "Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.",
    "test_types": [
        "Student's t-test (Independent)", "Welch's t-test (Independent)",
        "Mann-Whitney U Test", "F-Test for Two Variances",
    ],
    "csv": """species,sepal_length
setosa,5.1
setosa,4.9
setosa,4.7
setosa,4.6
setosa,5.0
setosa,5.4
setosa,4.6
setosa,5.0
setosa,4.4
setosa,4.9
setosa,5.4
setosa,4.8
setosa,4.8
setosa,4.3
setosa,5.8
setosa,5.7
setosa,5.4
setosa,5.1
setosa,5.7
setosa,5.1
versicolor,7.0
versicolor,6.4
versicolor,6.9
versicolor,5.5
versicolor,6.5
versicolor,5.7
versicolor,6.3
versicolor,4.9
versicolor,6.6
versicolor,5.2
versicolor,5.0
versicolor,5.9
versicolor,6.0
versicolor,6.1
versicolor,5.6
versicolor,6.7
versicolor,5.6
versicolor,5.8
versicolor,6.2
versicolor,5.6
""",
}

# ──────────────────────────────────────────────
# Paired / Dependent
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Sleep — Drug Effect (Paired)"] = {
    "description": "Increase in hours of sleep after two drugs, measured on the same subjects (Cushny & Peebles 1905). Classic paired design.",
    "source": "Cushny, A. R. & Peebles, A. R. (1905). The action of optical isomers.",
    "test_types": [
        "Paired t-test", "Wilcoxon Signed-Rank Test (Paired)",
        "Sign Test (Paired)",
    ],
    "csv": """subject,drug1,drug2
1,0.7,1.9
2,-1.6,0.8
3,-0.2,1.1
4,-1.2,0.1
5,-0.1,-0.1
6,3.4,4.4
7,3.7,5.5
8,0.8,1.6
9,0.0,4.6
10,2.0,3.4
""",
}

_BUILTIN_DATASETS["Before-After Blood Pressure"] = {
    "description": "Simulated systolic blood pressure before and after a treatment. Paired design.",
    "source": "Simulated teaching data.",
    "test_types": [
        "Paired t-test", "Wilcoxon Signed-Rank Test (Paired)",
        "Sign Test (Paired)",
    ],
    "csv": """subject,before,after
1,145,132
2,152,140
3,138,130
4,160,145
5,142,138
6,155,141
7,148,135
8,150,139
9,135,128
10,156,142
11,143,136
12,158,144
""",
}

# ──────────────────────────────────────────────
# Multi-sample (Independent) — ANOVA
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["PlantGrowth — All Three Groups"] = {
    "description": "Dried plant weights under control and two treatments (Dobson 1983). One-way ANOVA classic.",
    "source": "Dobson, A. J. (1983). An Introduction to Statistical Modelling.",
    "test_types": [
        "One-way ANOVA", "Kruskal-Wallis Test", "Mood's Median Test",
    ],
    "csv": """group,weight
Control,4.17
Control,5.58
Control,5.18
Control,6.11
Control,4.50
Control,4.61
Control,5.17
Control,4.53
Control,5.33
Control,5.14
Treatment1,4.81
Treatment1,4.17
Treatment1,4.41
Treatment1,3.59
Treatment1,5.87
Treatment1,3.83
Treatment1,6.03
Treatment1,4.89
Treatment1,4.32
Treatment1,4.69
Treatment2,6.31
Treatment2,5.12
Treatment2,5.54
Treatment2,5.50
Treatment2,5.37
Treatment2,5.29
Treatment2,4.92
Treatment2,6.15
Treatment2,5.80
Treatment2,5.26
""",
}

_BUILTIN_DATASETS["Iris — All Three Species (Sepal Length)"] = {
    "description": "Sepal length across all three iris species (Fisher 1936). Classic one-way ANOVA.",
    "source": "Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.",
    "test_types": [
        "One-way ANOVA", "Kruskal-Wallis Test", "Mood's Median Test",
    ],
    "csv": """species,sepal_length
setosa,5.1
setosa,4.9
setosa,4.7
setosa,4.6
setosa,5.0
setosa,5.4
setosa,4.6
setosa,5.0
setosa,4.4
setosa,4.9
setosa,5.4
setosa,4.8
setosa,4.8
setosa,4.3
setosa,5.8
setosa,5.7
setosa,5.4
setosa,5.1
setosa,5.7
setosa,5.1
versicolor,7.0
versicolor,6.4
versicolor,6.9
versicolor,5.5
versicolor,6.5
versicolor,5.7
versicolor,6.3
versicolor,4.9
versicolor,6.6
versicolor,5.2
versicolor,5.0
versicolor,5.9
versicolor,6.0
versicolor,6.1
versicolor,5.6
versicolor,6.7
versicolor,5.6
versicolor,5.8
versicolor,6.2
versicolor,5.6
virginica,6.3
virginica,5.8
virginica,7.1
virginica,6.3
virginica,6.5
virginica,7.6
virginica,4.9
virginica,7.3
virginica,6.7
virginica,7.2
virginica,6.5
virginica,6.4
virginica,6.8
virginica,5.7
virginica,5.8
virginica,6.4
virginica,6.5
virginica,7.7
virginica,7.7
virginica,6.0
""",
}

# ──────────────────────────────────────────────
# Multi-sample (Dependent) — Repeated Measures
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Repeated Measures — Exam Scores"] = {
    "description": "Simulated exam scores across three time points for the same students. Use for Friedman or repeated measures ANOVA.",
    "source": "Simulated teaching data.",
    "test_types": [
        "Friedman Test", "Repeated Measures ANOVA (One-way)",
    ],
    "csv": """student,midafterm,posttest,final
1,72,75,80
2,65,68,72
3,85,88,91
4,58,60,65
5,90,92,94
6,70,74,78
7,78,80,84
8,62,65,68
9,88,90,92
10,55,58,62
11,82,85,88
12,68,70,74
13,75,78,82
14,60,62,66
15,95,96,97
""",
}

# ──────────────────────────────────────────────
# Correlation
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Cars — Speed vs Stopping Distance"] = {
    "description": "Speed (mph) and stopping distance (ft) from 1920s cars. Classic correlation example.",
    "source": "Ezekiel, M. (1930). Methods of Correlation Analysis.",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
    ],
    "csv": """speed,dist
4,2
4,10
7,4
7,22
8,16
9,10
10,18
10,26
10,34
11,17
11,28
12,14
12,20
12,24
12,28
13,26
13,34
13,34
13,46
14,26
14,36
14,60
14,80
15,20
15,26
15,54
16,32
16,40
17,32
17,40
17,50
18,42
18,56
18,76
18,84
19,36
19,46
19,68
20,32
20,48
20,52
20,56
20,64
22,66
23,54
24,70
24,92
24,93
24,120
25,85
""",
}

_BUILTIN_DATASETS["Iris — Sepal Length vs Sepal Width"] = {
    "description": "Sepal length and sepal width measurements for all iris species (Fisher 1936). Correlation between two continuous variables.",
    "source": "Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
    ],
    "csv": """sepal_length,sepal_width
5.1,3.5
4.9,3.0
4.7,3.2
4.6,3.1
5.0,3.6
5.4,3.9
4.6,3.4
5.0,3.4
4.4,2.9
4.9,3.1
5.4,3.7
4.8,3.4
4.8,3.0
4.3,3.0
5.8,4.0
5.7,4.4
5.4,3.9
5.1,3.5
5.7,3.8
5.1,3.8
7.0,3.2
6.4,3.2
6.9,3.1
5.5,2.3
6.5,2.8
5.7,2.8
6.3,3.3
4.9,2.4
6.6,2.9
5.2,2.7
5.0,2.0
5.9,3.0
6.0,2.2
6.1,2.9
5.6,2.9
6.7,3.1
5.6,3.0
5.8,2.7
6.2,2.2
5.6,2.5
6.3,3.3
5.8,2.7
7.1,3.0
6.3,2.9
6.5,3.0
7.6,3.0
4.9,2.5
7.3,2.9
6.7,2.5
7.2,3.6
""",
}

# ──────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["mtcars — MPG vs Weight"] = {
    "description": "Fuel consumption (mpg) and weight (1000 lbs) for 32 cars (1974 Motor Trend US). Simple linear regression.",
    "source": "Henderson and Velleman (1981). Building multiple regression models interactively.",
    "test_types": [
        "Simple Linear Regression",
    ],
    "csv": """mpg,wt
21.0,2.620
21.0,2.875
22.8,2.320
21.4,3.215
18.7,3.440
18.1,3.460
14.3,3.570
24.4,3.190
22.8,3.150
19.2,3.440
17.8,3.440
16.4,4.070
17.3,3.730
15.2,3.780
10.4,5.250
10.4,5.424
14.7,5.345
32.4,2.200
30.4,1.615
33.9,1.835
21.5,2.465
15.5,3.520
15.2,3.435
13.3,3.840
19.2,3.845
27.3,1.935
26.0,2.140
30.4,1.513
15.8,3.170
19.7,2.770
15.0,3.570
21.4,2.780
""",
}

_BUILTIN_DATASETS["Anscombe's Quartet — Dataset I"] = {
    "description": "Anscombe's quartet dataset I — a linear relationship. Same slope, same R², very different patterns. Teaches why visualization matters.",
    "source": "Anscombe, F. J. (1973). Graphs in statistical analysis.",
    "test_types": [
        "Simple Linear Regression", "Pearson Correlation",
    ],
    "csv": """x,y
10.0,8.04
8.0,6.95
13.0,7.58
9.0,8.81
11.0,8.33
14.0,9.96
6.0,7.24
4.0,4.26
12.0,10.84
7.0,4.82
5.0,5.68
""",
}

# ──────────────────────────────────────────────
# Categorical / Chi-Square
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Titanic — Survival by Passenger Class"] = {
    "description": "Contingency table of Titanic survival by passenger class. Chi-square test of independence.",
    "source": "Hind, P. (1998). Encyclopedia Titanica.",
    "test_types": [
        "Chi-Square Test of Independence",
    ],
    "csv": """class,survived
1st,Yes
1st,Yes
1st,No
1st,No
1st,Yes
1st,Yes
1st,Yes
1st,No
1st,Yes
1st,No
1st,No
1st,No
1st,Yes
1st,No
1st,Yes
1st,Yes
1st,No
1st,No
1st,Yes
1st,No
2nd,No
2nd,No
2nd,Yes
2nd,No
2nd,Yes
2nd,Yes
2nd,No
2nd,No
2nd,Yes
2nd,No
2nd,No
2nd,No
2nd,Yes
2nd,No
2nd,Yes
2nd,Yes
2nd,No
2nd,No
2nd,Yes
2nd,No
3rd,No
3rd,Yes
3rd,No
3rd,No
3rd,No
3rd,Yes
3rd,No
3rd,No
3rd,No
3rd,No
3rd,Yes
3rd,No
3rd,No
3rd,Yes
3rd,No
3rd,No
3rd,No
3rd,No
3rd,No
3rd,No
""",
}

# ──────────────────────────────────────────────
# Agreement / Categorical Association
# ──────────────────────────────────────────────

_BUILTIN_DATASETS["Two Raters — Diagnostic Agreement"] = {
    "description": "Two physicians rating the presence of a disease (Yes/No). Kappa for inter-rater agreement.",
    "source": "Simulated teaching data.",
    "test_types": [
        "Cohen's Kappa", "McNemar's Test",
    ],
    "csv": """rater1,rater2
Yes,Yes
Yes,Yes
No,No
Yes,Yes
No,No
Yes,No
No,No
Yes,Yes
No,No
No,Yes
Yes,Yes
No,No
Yes,Yes
No,No
Yes,Yes
No,No
Yes,No
No,No
No,No
Yes,Yes
""",
}

# ═══════════════════════════════════════════════════
# NEW ADDITIONS — covering more test types
# ═══════════════════════════════════════════════════

_BUILTIN_DATASETS["Palmer Penguins — Body Mass by Species"] = {
    "description": "Body mass (g) of Adelie, Chinstrap, and Gentoo penguins. Excellent for one-way ANOVA, t-tests between species, and correlation with flipper length.",
    "source": "Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data.",
    "source_url": "https://github.com/allisonhorst/palmerpenguins",
    "test_types": [
        "One-way ANOVA", "Kruskal-Wallis Test", "Mood's Median Test",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
    ],
    "csv": """species,body_mass_g,flipper_length_mm,bill_length_mm
Adelie,3750,181,39.1
Adelie,3800,186,39.5
Adelie,3250,195,40.3
Adelie,3450,193,36.7
Adelie,3650,190,39.3
Adelie,3625,181,38.9
Adelie,4675,195,39.2
Adelie,3475,182,34.1
Adelie,4250,191,42.0
Adelie,3300,198,37.8
Adelie,3450,185,37.8
Adelie,3600,195,41.1
Adelie,4500,197,38.6
Adelie,3325,184,34.6
Adelie,4200,194,36.6
Adelie,3400,195,38.7
Adelie,3600,193,42.5
Adelie,3775,195,34.4
Adelie,3800,193,46.0
Adelie,3700,198,37.8
Chinstrap,4100,192,46.5
Chinstrap,3950,196,50.0
Chinstrap,3800,191,51.3
Chinstrap,4350,198,45.4
Chinstrap,3950,197,52.7
Chinstrap,3775,195,45.2
Chinstrap,4100,197,46.1
Chinstrap,3900,190,51.3
Chinstrap,4225,198,46.8
Chinstrap,3800,190,45.7
Chinstrap,3700,192,46.5
Chinstrap,3900,201,50.0
Chinstrap,3775,197,48.4
Chinstrap,4150,195,49.6
Chinstrap,4050,198,50.8
Chinstrap,3800,190,50.2
Chinstrap,3600,192,45.1
Chinstrap,3900,196,50.1
Chinstrap,4450,196,46.4
Chinstrap,4050,193,51.2
Gentoo,5550,222,50.1
Gentoo,5400,215,48.6
Gentoo,5150,220,48.5
Gentoo,5100,215,50.0
Gentoo,4900,222,48.6
Gentoo,5250,215,47.2
Gentoo,5050,213,49.5
Gentoo,4750,216,46.8
Gentoo,5550,230,49.4
Gentoo,5300,222,48.6
Gentoo,5200,220,52.7
Gentoo,5400,218,49.1
Gentoo,5700,230,50.6
Gentoo,5400,220,48.1
Gentoo,4925,210,46.5
Gentoo,5500,215,47.8
Gentoo,4750,215,48.9
Gentoo,5350,225,49.0
Gentoo,5550,220,44.3
Gentoo,5450,215,48.5
""",
}

_BUILTIN_DATASETS["Tips — Restaurant Tips"] = {
    "description": "Restaurant tip amounts by sex, smoker status, day, and time. From a national US restaurant chain (Bryant 2001). Good for t-tests, ANOVA, and regression.",
    "source": "Bryant, P. G. (2001). Tip data from a national restaurant chain.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/tips.csv",
    "test_types": [
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Pearson Correlation", "Spearman Rank Correlation",
        "Simple Linear Regression",
    ],
    "csv": """total_bill,tip,sex,smoker,day,time,size
16.99,1.01,Female,No,Sun,Dinner,2
10.34,1.66,Male,No,Sun,Dinner,3
21.01,3.50,Male,No,Sun,Dinner,3
23.68,3.31,Male,No,Sun,Dinner,2
24.59,3.61,Female,No,Sun,Dinner,4
25.29,4.71,Male,No,Sun,Dinner,4
8.77,2.00,Male,No,Sun,Dinner,2
26.88,3.12,Male,No,Sun,Dinner,4
15.04,1.96,Male,No,Sun,Dinner,2
14.78,3.23,Male,No,Sun,Dinner,2
10.27,1.71,Male,No,Sun,Dinner,2
35.26,5.00,Female,No,Sun,Dinner,4
15.42,1.57,Male,No,Sun,Dinner,2
18.43,3.00,Male,No,Sun,Dinner,4
14.83,3.02,Female,No,Sun,Dinner,2
21.58,3.92,Male,No,Sun,Dinner,2
10.33,1.67,Female,No,Sun,Dinner,3
16.29,3.71,Male,No,Sun,Dinner,3
16.97,3.50,Female,No,Sun,Dinner,3
20.83,3.15,Male,No,Sun,Dinner,3
23.45,3.70,Male,No,Sun,Dinner,3
20.52,2.15,Male,No,Sun,Dinner,2
21.35,3.50,Male,No,Sun,Dinner,3
22.83,2.25,Male,No,Sun,Dinner,2
17.92,3.75,Male,No,Sun,Dinner,2
19.08,2.00,Male,No,Sun,Dinner,2
28.44,5.00,Male,No,Sun,Dinner,4
15.36,3.06,Female,No,Sun,Dinner,2
20.34,2.30,Male,No,Sun,Dinner,2
18.24,3.60,Female,No,Sun,Dinner,3
12.54,2.50,Male,No,Sun,Dinner,2
9.82,1.55,Male,No,Sun,Dinner,2
16.99,2.50,Female,Yes,Sun,Dinner,2
10.59,1.03,Female,No,Sat,Dinner,2
20.49,3.68,Male,No,Sat,Dinner,3
14.65,3.43,Male,No,Sat,Dinner,2
17.42,2.55,Male,No,Sat,Dinner,2
18.43,1.95,Male,No,Sat,Dinner,2
19.59,2.72,Male,No,Sat,Dinner,2
21.13,3.44,Male,No,Sat,Dinner,2
20.37,2.89,Female,Yes,Sat,Dinner,3
23.14,3.43,Male,No,Sat,Dinner,2
25.56,4.31,Male,No,Sat,Dinner,4
13.42,2.75,Female,No,Sat,Dinner,2
19.81,4.00,Male,No,Sat,Dinner,2
15.06,2.00,Female,No,Sat,Dinner,2
15.48,2.02,Male,No,Sat,Dinner,2
21.45,2.66,Male,No,Sat,Dinner,2
20.71,2.80,Male,No,Sat,Dinner,2
""",
}

_BUILTIN_DATASETS["ToothGrowth — Vitamin C and Tooth Length"] = {
    "description": "Length of odontoblasts (teeth) in guinea pigs given vitamin C across two delivery methods and three dose levels (Crampton 1947). Classic factorial ANOVA example.",
    "source": "Crampton, E. W. (1947). The growth of the odontoblast of the incisor tooth as a criterion of vitamin C intake.",
    "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ToothGrowth.html",
    "test_types": [
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
        "Mood's Median Test",
    ],
    "csv": """len,supp,dose
4.2,VC,0.5
11.5,VC,0.5
7.3,VC,0.5
5.8,VC,0.5
6.4,VC,0.5
10.0,VC,0.5
11.2,VC,0.5
11.2,VC,0.5
5.2,VC,0.5
7.0,VC,0.5
16.5,VC,1.0
16.5,VC,1.0
15.2,VC,1.0
17.3,VC,1.0
22.5,VC,1.0
17.3,VC,1.0
13.6,VC,1.0
14.5,VC,1.0
18.8,VC,1.0
15.5,VC,1.0
23.6,VC,2.0
18.5,VC,2.0
33.9,VC,2.0
25.5,VC,2.0
26.4,VC,2.0
32.5,VC,2.0
26.7,VC,2.0
21.5,VC,2.0
23.3,VC,2.0
29.5,VC,2.0
15.2,OJ,0.5
21.5,OJ,0.5
17.6,OJ,0.5
9.7,OJ,0.5
14.5,OJ,0.5
10.0,OJ,0.5
8.2,OJ,0.5
9.4,OJ,0.5
16.5,OJ,0.5
9.7,OJ,0.5
19.7,OJ,1.0
23.3,OJ,1.0
23.6,OJ,1.0
26.4,OJ,1.0
20.0,OJ,1.0
25.2,OJ,1.0
25.8,OJ,1.0
21.2,OJ,1.0
14.5,OJ,1.0
27.3,OJ,1.0
25.5,OJ,2.0
26.4,OJ,2.0
22.4,OJ,2.0
24.5,OJ,2.0
24.8,OJ,2.0
30.9,OJ,2.0
26.4,OJ,2.0
27.3,OJ,2.0
29.4,OJ,2.0
23.0,OJ,2.0
""",
}

_BUILTIN_DATASETS["Diagnostic Test — Disease Detection"] = {
    "description": "Binary diagnostic test results (Positive/Negative) against true disease status. Use for diagnostic accuracy measures: sensitivity, specificity, PPV, NPV, Cohen's Kappa.",
    "source": "Simulated teaching data based on Altman & Bland (1994). Diagnostic tests.",
    "test_types": [
        "Cohen's Kappa", "McNemar's Test",
    ],
    "csv": """test_result,disease_status
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,Yes
Positive,No
Positive,No
Positive,No
Positive,No
Positive,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,No
Negative,Yes
Negative,Yes
Negative,Yes
Negative,Yes
Negative,Yes
""",
}

_BUILTIN_DATASETS["Survival — Lung Cancer (Veterans)"] = {
    "description": "Survival time (days) of male veterans with advanced lung cancer, with treatment type and other covariates. From Kalbfleisch & Prentice (1980).",
    "source": "Kalbfleisch, J. D. & Prentice, R. L. (1980). The Statistical Analysis of Failure Time Data.",
    "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/survival/veteran.html",
    "test_types": [
        "Mann-Whitney U Test", "Student's t-test (Independent)", "Welch's t-test (Independent)",
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Pearson Correlation", "Spearman Rank Correlation",
    ],
    "csv": """time,status,trt,karno,age
72,1,1,60,69
411,1,1,70,64
228,1,1,60,38
126,1,1,60,63
118,1,1,70,65
10,1,1,20,49
82,1,1,40,69
110,1,1,80,68
314,1,1,50,43
100,1,1,70,70
42,1,1,60,66
144,1,1,50,61
30,1,1,70,39
25,1,1,60,64
11,1,1,70,62
63,1,1,30,64
87,1,1,60,56
55,1,1,60,60
28,1,1,60,50
60,1,1,40,67
553,1,2,90,63
223,1,2,80,49
193,1,2,50,57
67,1,2,60,63
128,1,2,50,58
289,1,2,70,57
232,1,2,60,59
186,1,2,70,56
107,1,2,50,50
282,1,2,50,56
224,1,2,60,58
121,1,2,70,47
158,1,2,60,50
86,1,2,60,53
303,1,2,90,53
197,1,2,50,54
51,1,2,40,67
59,1,2,60,51
148,1,2,60,56
182,1,2,90,53
""",
}


# ═══════════════════════════════════════════════════
# FILE-BASED DATASETS (loaded from datasets/ directory)
# ═══════════════════════════════════════════════════

_BUILTIN_DATASETS["Diamonds — Price and Attributes"] = {
    "description": "Prices and physical attributes of ~54,000 round-cut diamonds. Excellent for multiple regression, correlation analysis, and demonstrating large-dataset workflows.",
    "source": "R ggplot2 package (carat, cut, color, clarity, depth, table, price, x, y, z). Originally from H. Wickham.",
    "source_url": "https://ggplot2.tidyverse.org/reference/diamonds.html",
    "file_path": "datasets/diamonds.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "One-way ANOVA", "Kruskal-Wallis Test",
    ],
}

_BUILTIN_DATASETS["Old Faithful — Eruption Duration"] = {
    "description": "Eruption duration (minutes) and waiting time (minutes) for Old Faithful geyser in Yellowstone. Classic bivariate distribution for correlation and regression.",
    "source": "Azzalini, A. & Bowman, A. W. (1990). A look at some data on the Old Faithful geyser.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/geyser.csv",
    "file_path": "datasets/geyser.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
    ],
}

_BUILTIN_DATASETS["Exercises — Pulse Before and After"] = {
    "description": "Pulse rates before and after various exercise types. Repeated measures with two kinds of exercise and a control group.",
    "source": "Buskirk, E. R. et al. (various). Exercise physiology data compiled for teaching.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/exercise.csv",
    "file_path": "datasets/exercise.csv",
    "test_types": [
        "Paired t-test", "Wilcoxon Signed-Rank Test (Paired)", "Sign Test (Paired)",
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Friedman Test", "Repeated Measures ANOVA (One-way)",
    ],
}

_BUILTIN_DATASETS["Titanic — Passenger Survival"] = {
    "description": "Full Titanic passenger manifest with survival, age, sex, fare, and class. The classic dataset for categorical analysis and logistic regression.",
    "source": "Hind, P. (1998). Encyclopedia Titanica. Curated by Seaborn.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv",
    "file_path": "datasets/titanic.csv",
    "test_types": [
        "Chi-Square Test of Independence",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
        "Pearson Correlation", "Spearman Rank Correlation",
        "Cohen's Kappa (Agreement Analysis)", "McNemar's Test",
    ],
}

_BUILTIN_DATASETS["MPG — Fuel Economy"] = {
    "description": "Fuel economy (mpg) for 392 car models from 1970-82, with weight, horsepower, displacement, acceleration, origin, and cylinders. Excellent for multiple regression.",
    "source": "Quinlan, R. (1993). Combining instance-based and model-based learning. UCI Machine Learning Repository.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv",
    "file_path": "datasets/mpg.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
    ],
}

_BUILTIN_DATASETS["US Arrests — Crime Statistics"] = {
    "description": "Violent crime rates per 100,000 residents by US state (1973). Four variables: Murder, Assault, UrbanPop, Rape. Classic PCA example.",
    "source": "McNeil, D. R. (1977). Interactive Data Analysis. Wiley.",
    "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USArrests.html",
    "file_path": "datasets/USArrests.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "One-sample t-test", "One-sample Wilcoxon Signed-Rank Test",
    ],
}

_BUILTIN_DATASETS["Planets — Exoplanet Orbits"] = {
    "description": "Data on exoplanets discovered through various methods up to 2014. Includes orbital period, mass, distance, and discovery year.",
    "source": "Exoplanet Orbit Database / Exoplanet Data Explorer. Curated by Seaborn.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/planets.csv",
    "file_path": "datasets/planets.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "One-way ANOVA", "Kruskal-Wallis Test",
    ],
}

_BUILTIN_DATASETS["Car Crashes — US Accident Data"] = {
    "description": "Car crash statistics per US state (2010-2012), including accident rate, alcohol-impaired driving, and insurance premiums. Good for correlation and regression.",
    "source": "National Highway Traffic Safety Administration. Curated by Seaborn.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/car_crashes.csv",
    "file_path": "datasets/car_crashes.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "One-sample t-test", "One-sample Wilcoxon Signed-Rank Test",
    ],
}

_BUILTIN_DATASETS["Flights — Airline Passengers"] = {
    "description": "Monthly totals of international airline passengers (1949-1960). Classic time series dataset useful for trend and seasonal analysis.",
    "source": "Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (1994). Time Series Analysis.",
    "source_url": "https://github.com/mwaskom/seaborn-data/blob/master/flights.csv",
    "file_path": "datasets/flights.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation",
        "Simple Linear Regression",
        "Friedman Test", "Repeated Measures ANOVA (One-way)",
    ],
}

_BUILTIN_DATASETS["mtcars — Full Motor Trend Data"] = {
    "description": "Full 1974 Motor Trend US magazine data on 32 car models with 11 variables (mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb). Multiple regression classic.",
    "source": "Henderson and Velleman (1981). Building multiple regression models interactively.",
    "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/mtcars.html",
    "file_path": "datasets/mtcars.csv",
    "test_types": [
        "Pearson Correlation", "Spearman Rank Correlation", "Kendall's Tau-b",
        "Simple Linear Regression",
        "One-way ANOVA", "Kruskal-Wallis Test",
        "Student's t-test (Independent)", "Welch's t-test (Independent)", "Mann-Whitney U Test",
    ],
}


def get_builtin_datasets():
    """Return the dict of built-in datasets."""
    return dict(_BUILTIN_DATASETS)


def get_compatible_datasets(test_name):
    """Return datasets compatible with a given test name."""
    return {
        name: info
        for name, info in _BUILTIN_DATASETS.items()
        if test_name in info["test_types"]
    }


def _resolve_path(relative_path):
    """Resolve a path relative to this file's location (features/ → project root)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


def load_builtin_dataset(name):
    """Load a built-in dataset by name. Returns a pandas DataFrame."""
    info = _BUILTIN_DATASETS.get(name)
    if info is None:
        return None
    if "file_path" in info:
        return pd.read_csv(_resolve_path(info["file_path"]))
    import io
    return pd.read_csv(io.StringIO(info["csv"]))


def get_all_dataset_names():
    """Return all built-in dataset names."""
    return list(_BUILTIN_DATASETS.keys())


def get_dataset_info(name):
    """Get metadata for a built-in dataset."""
    return _BUILTIN_DATASETS.get(name)
