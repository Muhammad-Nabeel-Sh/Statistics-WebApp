import re

with open("features/graph_explorer_original.py", encoding="utf-8") as f:
    orig = f.read()

# Find sections mentioning Post-Hoc Plots
sections = re.findall(r'"(.+?)":\s*\{.*?"category":\s*"Post-Hoc Plots"', orig, re.DOTALL)
print("Graphs in Post-Hoc Plots:")
for s in sections:
    print(f"  {s}")
if "Histogram" in sections:
    print("ERROR: Histogram is in Post-Hoc Plots!")
elif "Frequency" in sections:
    print("Frequency Polygon is in Post-Hoc Plots - this is the cause!")
