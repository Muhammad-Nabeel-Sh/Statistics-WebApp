import re
with open("features/graph_explorer/comparison.py", encoding="utf-8") as f:
    c = f.read()
if "raincloud" in c.lower():
    for i, l in enumerate(c.split("\n")):
        if "raincloud" in l.lower():
            print(f"  Line {i+1}: {l[:120]}")
else:
    print("NO raincloud reference in comparison.py")

# Check what GRAPHS entries exist in comparison.py
graphs_match = re.findall(r'"([^"]+)":\s*(\w+)', c)
print(f"\ncomparison.py GRAPHS entries ({len(graphs_match)}):")
for name, func in graphs_match[-5:]:
    print(f"  {name}: {func}")
