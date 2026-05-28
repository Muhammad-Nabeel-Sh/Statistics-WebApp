import re

source_path = "D:\\Projects\\Statistics WebApp\\features\\widgets.py"

with open(source_path, encoding="utf-8") as f:
    source_lines = f.readlines()

# Find function start
func_start = None
for i, line in enumerate(source_lines):
    if line.startswith("def render_test_widget("):
        func_start = i
        print(f"Function found at line {i}: {line.rstrip()}")
        break

# Find else line
else_line = None
for i in range(func_start or 0, len(source_lines)):
    if source_lines[i].lstrip().startswith("else:"):
        else_line = i
        print(f"Else found at line {i}: {source_lines[i].rstrip()}")
        break

# Test the pattern on lines in the function
test_pattern = re.compile(r'\s*(?:if|elif)\s+test_name\s*==\s*"(.+?)"\s*:')
count = 0
for i in range(func_start + 1, else_line):
    line = source_lines[i]
    m = test_pattern.match(line)
    if m:
        count += 1
        if count <= 5:
            print(f"  Line {i}: matched '{m.group(1)}': {line.rstrip()}")

print(f"\nTotal matches: {count}")
print(f"Function spans lines {func_start} to {else_line}")
