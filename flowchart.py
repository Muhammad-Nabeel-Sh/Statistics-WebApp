import streamlit as st
from data import FIELDS

def build_tree(rule_subset, fields, user_input=None, level=0):

    # No more fields → show tests
    if not fields:

        for rule in rule_subset:
            st.success(f"✅ {rule['name']}")

        return

    current_field = fields[0]
    # Skip meaningless all-any branches
    all_any = all(rule[current_field] == "any" for rule in rule_subset)

    if all_any:
        build_tree(rule_subset, fields[1:], user_input, level + 1)
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

        # Highlighting logic
        is_selected = False
        if user_input and current_field in user_input:
            user_val = user_input[current_field]
            if user_val == value or (user_val == "any" and value == "any"):
                is_selected = True

        display_label = (
            f"🎯 **{current_field}: {label}** (Current Selection)"
            if is_selected
            else f"{current_field}: {label}"
        )

        with st.expander(display_label, expanded=is_selected):

            build_tree(subrules, fields[1:], user_input, level + 1)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
