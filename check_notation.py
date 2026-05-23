import re

with open('power_calculator.py', encoding='utf-8') as f:
    content = f.read()

in_formula = False
issues = []
for i, line in enumerate(content.split('\n'), 1):
    if 'formula_latex' in line:
        in_formula = True
    if in_formula:
        if re.search(r'z_\{[^1]?\\beta\}', line) and 'z_{1-' not in line:
            issues.append(f'Line {i}: old z_\\beta without 1-')
        if re.search(r'z_\{[^1]?\\alpha\}', line) and 'z_{1-' not in line:
            issues.append(f'Line {i}: old z_\\alpha without 1-')
    if line.strip() == '' or (in_formula and ')' in line and 'formula_latex' not in line):
        in_formula = False

if issues:
    for issue in issues:
        print(issue)
else:
    print('All formula notation is clean')
