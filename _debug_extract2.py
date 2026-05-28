import re
with open("features/widgets.py", encoding="utf-8") as f:
    lines = f.readlines()

pattern = re.compile(r'^    (?:if|elif)\s+test_name\s*==\s*"(.+?)"\s*:\s*$')
count = 0
for i in range(len(lines)):
    m = pattern.match(lines[i])
    if m:
        count += 1
        if count <= 3:
            print(f"  Line {i+1}: matched \"{m.group(1)}\"")
print(f"Total: {count}")

# Also find the else at 4 spaces
for i in range(len(lines)):
    if lines[i] == "    else:\n":
        print(f"else at line {i+1}")
        break
