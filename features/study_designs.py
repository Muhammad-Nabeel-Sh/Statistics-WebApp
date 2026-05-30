import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.utils import _apa_table

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "study_designs.json")

BIAS_LABELS = {
    "selection_bias": "Selection Bias",
    "information_bias": "Information Bias",
    "confounding": "Confounding",
    "performance_bias": "Performance Bias",
    "detection_bias": "Detection Bias",
    "attrition_bias": "Attrition Bias",
    "reporting_bias": "Reporting Bias",
    "lead_time_bias": "Lead-Time Bias",
    "length_time_bias": "Length-Time Bias",
}

BIAS_DESCRIPTIONS = {
    "selection_bias": "Systematic differences between comparison groups in who is selected or who participates",
    "information_bias": "Systematic differences in how exposure or outcome information is collected",
    "confounding": "A third variable associated with both exposure and outcome that distorts the true relationship",
    "performance_bias": "Systematic differences in care, procedures, or co-interventions between groups",
    "detection_bias": "Systematic differences in how outcomes are assessed between groups",
    "attrition_bias": "Systematic differences in who drops out or is lost to follow-up between groups",
    "reporting_bias": "Selective reporting of outcomes, analyses, or subgroups based on results",
    "lead_time_bias": "Earlier detection creates artificial improvements in survival or outcomes",
    "length_time_bias": "Overrepresentation of less aggressive, slower-progressing disease in screening-detected cases",
}

RISK_ORDER = {"Very High": 0, "High": 1, "Moderate": 2, "Low": 3, "N/A": 4}
RISK_COLORS = {
    "Very High": "rgba(228,87,86,0.85)",
    "High": "rgba(228,87,86,0.55)",
    "Moderate": "rgba(241,197,55,0.65)",
    "Low": "rgba(76,120,168,0.55)",
    "N/A": "rgba(128,128,128,0.25)",
}

SESSION_KEY = "design_section"
DESIGN_KEY = "selected_design"


def _load_designs():
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _section_button(label, key_prefix):
    if st.sidebar.button(
        label,
        key=f"{key_prefix}_btn_{label.replace(' ', '_').replace('/', '_')}",
        use_container_width=True,
        type="primary" if st.session_state.get(SESSION_KEY) == label else "secondary",
    ):
        st.session_state[SESSION_KEY] = label
        st.session_state[DESIGN_KEY] = None
        st.rerun()


def _init_state():
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = "Design Wizard"
    if DESIGN_KEY not in st.session_state:
        st.session_state[DESIGN_KEY] = None


def render_study_designs():
    _init_state()
    designs = _load_designs()

    with st.sidebar:
        st.title("Study Designs")

        st.subheader("Interactive Tools")
        _section_button("Design Wizard", "tool")
        _section_button("Bias Matrix", "tool")
        _section_button("Bias Checklist", "tool")

        st.divider()
        st.subheader("Design Reference")

        for level in range(1, 6):
            level_designs = [d for d in designs if d["evidence_pyramid"] == level]
            if not level_designs:
                continue
            level_labels = {1: "I — RCT / Experimental", 2: "II — Quasi-Experimental", 3: "III — Observational (Analytic)", 4: "IV — Observational (Descriptive)", 5: "V — Case Reports / Expert Opinion"}
            st.markdown(f"**Level {level_labels[level]}**")
            for d in level_designs:
                label = f"{d['icon']} {d['name']}"
                btn_key = f"design_{d['name'].replace(' ', '_').replace('/', '_').replace(',', '')}"
                if st.sidebar.button(
                    label,
                    key=btn_key,
                    use_container_width=True,
                    type="primary" if st.session_state.get(DESIGN_KEY) == d["name"] else "secondary",
                ):
                    st.session_state[SESSION_KEY] = "design_detail"
                    st.session_state[DESIGN_KEY] = d["name"]
                    st.rerun()
            st.sidebar.markdown("<br>", unsafe_allow_html=True)

    section = st.session_state[SESSION_KEY]

    if section == "Design Wizard":
        _render_wizard(designs)
    elif section == "Bias Matrix":
        _render_bias_matrix(designs)
    elif section == "Bias Checklist":
        _render_bias_checklist(designs)
    elif section == "design_detail":
        selected = st.session_state.get(DESIGN_KEY)
        design = next((d for d in designs if d["name"] == selected), None)
        if design:
            _render_design_page(design)
        else:
            st.warning("Design not found. Please select a design from the sidebar.")


def _render_wizard(designs):
    st.title("Interactive Study Design Wizard")
    st.info("""
    Answer a few questions to narrow down which study design is most appropriate 
    for your research question. This tool is for guidance only — consult a 
    biostatistician or epidemiologist for final design decisions.
    """)

    st.markdown("### Your Research Parameters")

    c1, c2 = st.columns(2)
    with c1:
        has_intervention = st.radio(
            "Do you assign an intervention?",
            ["Yes — I control the exposure", "No — I observe exposures as they occur"],
            key="wiz_intervention",
        )
    with c2:
        randomization = st.radio(
            "Can you randomize?",
            ["Yes", "No", "Not applicable (no intervention)"],
            key="wiz_random",
        )

    c3, c4 = st.columns(2)
    with c3:
        timeline = st.radio(
            "When is the outcome measured?",
            ["After the intervention (prospective)", "Looking back (retrospective)", "At the same time (cross-sectional)"],
            key="wiz_timeline",
        )
    with c4:
        comparison = st.radio(
            "Do you have a comparison group?",
            ["Yes — a separate control/comparison group", "No — all participants receive the same treatment", "Each participant serves as their own control"],
            key="wiz_comparison",
        )

    c5, c6 = st.columns(2)
    with c5:
        outcome_rarity = st.radio(
            "Is the outcome rare?",
            ["No — common outcome (>10%)", "Yes — rare outcome (<5%)", "Not sure"],
            key="wiz_rarity",
        )
    with c6:
        cluster = st.radio(
            "Is the intervention delivered at the group level?",
            ["No — individual level", "Yes — group/school/clinic level"],
            key="wiz_cluster",
        )

    st.divider()

    scores = {}
    for d in designs:
        score = 0
        n_criteria = 0

        if has_intervention.startswith("Yes"):
            if randomization == "Yes":
                if cluster == "No — individual level":
                    if timeline == "After the intervention (prospective)":
                        score += 3 if "RCT" in d["name"] else 0
                    else:
                        score += 1 if "RCT" in d["name"] else 0
                else:
                    score += 2 if "Cluster" in d["name"] else 0
                n_criteria += 1
            else:
                if "Case" in d["name"] or "Cohort" in d["name"] or "Stepped" in d["name"]:
                    score += 1
                n_criteria += 1
        else:
            if comparison == "Yes — a separate control/comparison group":
                if timeline == "Looking back (retrospective)":
                    score += 2 if "Case-Control" in d["name"] else 0
                elif timeline == "After the intervention (prospective)":
                    score += 2 if "Cohort" in d["name"] else 0
                n_criteria += 1

            if outcome_rarity.startswith("Yes"):
                score += 2 if "Case-Control" in d["name"] else 0
                n_criteria += 1
            elif outcome_rarity.startswith("No"):
                score += 2 if "Cohort" in d["name"] or "Cross-Sectional" in d["name"] else 0
                n_criteria += 1

            if timeline == "At the same time (cross-sectional)":
                score += 3 if "Cross-Sectional" in d["name"] else 0
                n_criteria += 1

        if comparison == "Each participant serves as their own control":
            score += 3 if "Crossover" in d["name"] else 0
            n_criteria += 1
        elif comparison == "No — all participants receive the same treatment":
            score += 3 if "Single-Arm" in d["name"] else 0
            n_criteria += 1

        scores[d["name"]] = score / max(n_criteria, 1)

    top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_designs = [(name, score) for name, score in top_n if score > 0]

    if not top_designs:
        st.warning("No designs closely match your parameters. Consider consulting a research methodologist.")
        st.markdown("#### All Available Designs")
        for d in designs:
            st.markdown(f"- {d['icon']} **{d['name']}** (Evidence Level {d['evidence_pyramid']})")
        return

    st.markdown("### Recommended Designs")
    st.markdown("Based on your answers, the following designs may be most appropriate:")

    cols = st.columns(len(top_designs))
    for i, (name, score) in enumerate(top_designs):
        design = next(d for d in designs if d["name"] == name)
        with cols[i]:
            st.markdown(f"### {design['icon']}")
            st.markdown(f"**{name}**")
            st.markdown(f"Evidence Level: **{design['evidence_pyramid']}**")
            st.markdown(f"Match: {score:.0%}")
            aliases = design.get("aliases", [])
            if aliases:
                st.caption(f"Also known as: {', '.join(aliases[:3])}")
            if st.button("View Details", key=f"wiz_goto_{design['name']}", use_container_width=True):
                st.session_state[SESSION_KEY] = "design_detail"
                st.session_state[DESIGN_KEY] = name
                st.rerun()

    st.divider()
    st.markdown("#### Understanding Your Results")
    st.markdown("""
    The wizard scores designs based on **matching criteria**, not statistical validity. 
    A high match score means the design fits your structural parameters, not that it is 
    automatically the best choice. Consider:

    - **Feasibility**: Do you have the resources, time, and access to participants?
    - **Ethics**: Is randomization ethical? Is withholding the intervention acceptable?
    - **Bias profile**: What are the key vulnerabilities of the recommended design?
    - **Sample size**: Do you have enough participants for adequate statistical power?
    """)


def _render_bias_matrix(designs):
    st.title("Bias Vulnerability Matrix")
    st.info("""
    This matrix shows which biases pose the greatest threat to each study design. 
    Darker red cells indicate higher vulnerability. Use this to identify which biases 
    you need to address in your study design and analysis.
    """)

    rows = []
    for d in designs:
        bp = d["sections"]["bias_profile"]
        row = {"Design": f"{d['icon']} {d['name']}"}
        for bk in BIAS_LABELS:
            entry = bp.get(bk, {"risk": "N/A"})
            row[BIAS_LABELS[bk]] = entry["risk"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Design")

    risk_order = ["Very High", "High", "Moderate", "Low", "N/A"]
    risk_values = {r: i for i, r in enumerate(risk_order)}

    z = df.map(lambda x: risk_values.get(x, 4))
    bias_cols = list(BIAS_LABELS.values())

    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=bias_cols,
        y=df.index.tolist(),
        text=df.values,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=[
            [0, "rgba(228,87,86,0.85)"],
            [0.25, "rgba(228,87,86,0.55)"],
            [0.5, "rgba(241,197,55,0.65)"],
            [0.75, "rgba(76,120,168,0.55)"],
            [1, "rgba(128,128,128,0.25)"],
        ],
        zmin=0,
        zmax=4,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=60 + 40 * len(designs),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"side": "top", "tickangle": -30},
        yaxis={"tickfont": {"size": 11}},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Legend")
    c = st.columns(4)
    for i, risk in enumerate(risk_order):
        with c[i % 4]:
            st.markdown(
                f"<div style='background-color:{RISK_COLORS[risk]}; padding:6px 12px; "
                f"border-radius:4px; text-align:center; color:white; font-weight:600;'>{risk}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Bias Definitions (click to expand)"):
        for bk, label in BIAS_LABELS.items():
            st.markdown(f"**{label}**: {BIAS_DESCRIPTIONS.get(bk, '')}")


def _render_bias_checklist(designs):
    st.title("Bias Checklist Tool")
    st.info("""
    Select a study design and check which biases you have addressed. 
    The tool will summarize your remaining vulnerabilities and suggest mitigation strategies.
    """)

    design_names = [d["name"] for d in designs]
    selected_name = st.selectbox("Select a study design", design_names, key="bc_design")
    design = next((d for d in designs if d["name"] == selected_name), None)
    if design is None:
        st.warning("Design not found. Please select a valid design.")
        return

    st.markdown(f"### {design['icon']} {design['name']} — Bias Assessment")

    bp = design["sections"]["bias_profile"]
    responses = {}

    st.markdown("For each bias, indicate whether you have addressed it in your study design:")

    for bk in BIAS_LABELS:
        label = BIAS_LABELS[bk]
        entry = bp.get(bk, {"risk": "N/A", "detail": ""})
        risk = entry["risk"]
        detail = entry.get("detail", "")

        if risk == "N/A":
            continue

        color = RISK_COLORS.get(risk, "gray")
        with st.container():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding-left:10px; margin:4px 0;'>"
                    f"<b>{label}</b><br><span style='font-size:0.85em;'>{detail}</span></div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                risk_label = st.markdown(
                    f"<div style='background:{color}; padding:2px 8px; border-radius:4px; "
                    f"text-align:center; color:white; font-weight:600; font-size:0.85em;'>{risk}</div>",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                responses[bk] = st.selectbox(
                    "Status",
                    ["Not Addressed", "Partially", "Fully Addressed", "N/A for my study"],
                    key=f"bc_{bk}",
                    label_visibility="collapsed",
                )

    st.divider()

    not_addressed = [BIAS_LABELS[bk] for bk, v in responses.items() if v == "Not Addressed"]
    partial = [BIAS_LABELS[bk] for bk, v in responses.items() if v == "Partially"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fully Addressed", sum(1 for v in responses.values() if v == "Fully Addressed"))
    with c2:
        st.metric("Partially Addressed", len(partial))
    with c3:
        st.metric("Not Addressed", len(not_addressed))

    if not_addressed:
        st.warning(f"**Vulnerabilities to address**: {', '.join(not_addressed)}")
        st.markdown("**Suggestions:**")
        suggestions = {
            "Selection Bias": "Use random or consecutive sampling; clearly define eligibility criteria; compare participants vs non-participants.",
            "Information Bias": "Use validated measurement instruments; blind assessors; standardize data collection protocols.",
            "Confounding": "Identify confounders via DAGs; use multivariable regression, propensity scores, or instrumental variables; consider matching.",
            "Performance Bias": "Standardize protocols across groups; blind participants and providers when possible; monitor co-interventions.",
            "Detection Bias": "Blind outcome assessors; use objective outcomes when possible; pre-specify outcome definitions.",
            "Attrition Bias": "Minimize loss to follow-up; track reasons for dropout; use multiple imputation or inverse probability weighting.",
            "Reporting Bias": "Pre-register your study; publish a protocol; report all pre-specified outcomes regardless of significance.",
            "Lead-Time Bias": "Use date of outcome (not date of diagnosis) as the endpoint; consider randomized designs for screening evaluation.",
            "Length-Time Bias": "Use incident (not prevalent) cases; avoid comparing screening-detected vs clinically detected cases without adjustment.",
        }
        for bias in not_addressed:
            if bias in suggestions:
                st.markdown(f"- **{bias}**: {suggestions[bias]}")

    if not not_addressed and not partial:
        st.success("All biases have been addressed. Your study design is robust against the identified threats to validity.")
    elif not not_addressed and partial:
        st.info("Most biases are addressed, but some need further attention. Review the suggestions above.")


def _render_design_page(design):
    s = design["sections"]

    with st.container():
        evidence_labels = {1: "I — High (RCT / Experimental)", 2: "II — Moderate-High (Quasi-Experimental)", 3: "III — Moderate (Observational — Analytic)", 4: "IV — Low-Moderate (Observational — Descriptive)", 5: "V — Low (Case Reports / Expert Opinion)"}
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>"
            f"<span style='font-size:2.5rem;'>{design['icon']}</span>"
            f"<span style='font-size:1.8rem; font-weight:700;'>{design['name']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        aliases = design.get("aliases", [])
        if aliases:
            st.caption(f"Also known as: {', '.join(aliases)}")

        ev_level = design["evidence_pyramid"]
        ev_colors = {1: "rgba(76,120,168,0.8)", 2: "rgba(76,120,168,0.6)", 3: "rgba(241,197,55,0.7)", 4: "rgba(228,87,86,0.5)", 5: "rgba(228,87,86,0.7)"}
        st.markdown(
            f"<div style='background:{ev_colors[ev_level]}; padding:6px 14px; border-radius:6px; "
            f"display:inline-block; margin:6px 0 16px; font-weight:600;'>"
            f"Evidence Level {ev_level}</div>",
            unsafe_allow_html=True,
        )

    # Definition
    st.markdown("### What Is This Design?")
    st.markdown(s["definition"])

    # When to use
    st.markdown("### When to Use")
    st.info(s["when_to_use"])

    # Causal inference
    st.markdown("### Causal Inference & Evidence Strength")
    st.markdown(s["causal_inference"])

    # Assumptions
    st.markdown("### Assumptions & Prerequisites")
    for i, assumption in enumerate(s["assumptions"], 1):
        st.markdown(f"{i}. {assumption}")

    # Bias profile
    st.markdown("### Bias Profile")
    bp = s["bias_profile"]
    bias_rows = []
    for bk in BIAS_LABELS:
        entry = bp.get(bk, {"risk": "N/A", "detail": ""})
        risk = entry["risk"]
        detail = entry.get("detail", "")
        color = RISK_COLORS.get(risk, "gray")
        bias_rows.append({
            "Bias": BIAS_LABELS[bk],
            "Risk": risk,
            "Detail": detail,
        })
    bias_df = pd.DataFrame(bias_rows)
    col_map = {"Bias": None, "Risk": None, "Detail": None}
    _apa_table(bias_df, hide_index=True)

    # Worked example
    st.markdown("### Worked Example")
    with st.expander("View worked example", expanded=True):
        st.markdown(s["worked_example"])

    # Reporting guideline
    st.markdown("### Reporting Guideline")
    rg = s.get("reporting_guideline", {})
    if rg:
        name = rg.get("name", "")
        url = rg.get("url", "")
        key_items = rg.get("key_items", [])
        st.markdown(f"**{name}**")
        if url:
            st.markdown(f"[{url}]({url})")
        if key_items:
            st.markdown("**Key reporting items:**")
            for item in key_items:
                st.markdown(f"- {item}")

    # Pitfalls
    st.markdown("### Common Pitfalls")
    for pitfall in s.get("pitfalls", []):
        st.warning(pitfall)

    # Statistical considerations
    st.markdown("### Statistical Considerations")
    st.markdown(s.get("statistical_considerations", ""))

    # Software notes
    st.markdown("### Software Notes & Data Structure")
    with st.expander("Implementation details", expanded=False):
        st.markdown(s.get("software_notes", ""))

    # Further reading
    st.markdown("### Further Reading")
    for ref in s.get("further_reading", []):
        st.markdown(f"- {ref}")

    # Navigation footer
    st.divider()
    if st.button("Back to Design Wizard", use_container_width=True):
        st.session_state[SESSION_KEY] = "Design Wizard"
        st.session_state[DESIGN_KEY] = None
        st.rerun()
