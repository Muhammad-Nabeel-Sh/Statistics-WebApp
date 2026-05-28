import re

with open("features/graph_explorer_original.py", encoding="utf-8") as f:
    orig = f.read()

# Find the Post-Hoc Plots section
idx = orig.find('"Post-Hoc Plots"')
if idx >= 0:
    # Find the graph name before this
    before = orig[:idx]
    names = re.findall(r'"([^"]+)"\s*:\s*{', before)
    if names:
        print(f"Graph before Post-Hoc: {names[-1]}")
    print(f"Post-Hoc Plots found at position {idx}")
    print(f"Context: ...{orig[idx-50:idx+50]}...")
else:
    print("Post-Hoc Plots not found!")

# Also check what's in post_hoc.py's GRAPHS
with open("features/graph_explorer/post_hoc.py", encoding="utf-8") as f:
    ph = f.read()
# Extract all GRAPHS entry names
graph_names = re.findall(r'"([^"]+)"\s*:\s*\{', ph)
print(f"\npost_hoc.py GRAPHS entries:")
for n in graph_names:
    print(f"  {n}")
