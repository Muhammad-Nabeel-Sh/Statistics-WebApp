from core.data import CRITERIA_FIELDS, rules


def matches_rule(user_input, rule):

    for key in CRITERIA_FIELDS:

        rule_val = rule.get(key)
        if rule_val is None:
            continue

        user_val = user_input.get(key)

        if rule_val == "any":
            continue

        if isinstance(rule_val, list):
            if user_val not in rule_val:
                return False

        else:
            if user_val != rule_val:
                return False

    return True


def find_matching_tests(user_input):

    matches = []

    for rule in rules:
        if matches_rule(user_input, rule):
            matches.append(rule.name)

    return matches
