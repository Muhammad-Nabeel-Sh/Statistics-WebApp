import re
from collections import defaultdict

with open("features/graph_explorer.py", encoding="utf-8") as f:
    lines = f.readlines()

# Find graphs dict boundaries
graphs_start = None
graphs_end = None
for i, line in enumerate(lines):
    if line.strip() == "graphs = {":
        graphs_start = i
    if graphs_start is not None and line.strip() == "}":
        graphs_end = i
        break

# Parse graph entries: name -> {category, func_name}
func_refs = defaultdict(list)  # func_name -> [(category, graph_name)]

name_pattern = re.compile(r'^\s*"(.+?)":\s*{')
cat_pattern = re.compile(r'"category":\s*"(.+?)"')
func_pattern = re.compile(r'"widget_function":\s*(\w+)')

current_name = None
current_cat = None
current_func = None
brace_depth = 0
in_graph = False

for i in range(graphs_start, graphs_end + 1):
    line = lines[i]
    if not in_graph:
        m = name_pattern.match(line)
        if m:
            current_name = m.group(1)
            current_cat = None
            current_func = None
            in_graph = True
            brace_depth = line.count("{") - line.count("}")
            cm = re.search(r'"category":\s*"(.+?)"', line)
            if cm: current_cat = cm.group(1)
            fm = re.search(r'"widget_function":\s*(\w+)', line)
            if fm: current_func = fm.group(1)
    else:
        if not current_cat:
            cm = re.search(r'"category":\s*"(.+?)"', line)
            if cm: current_cat = cm.group(1)
        if not current_func:
            fm = re.search(r'"widget_function":\s*(\w+)', line)
            if fm: current_func = fm.group(1)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            if current_func:
                func_refs[current_func].append((current_cat, current_name))
            current_name = None
            in_graph = False

# Check for functions used in multiple categories
multi_cat = {k: v for k, v in func_refs.items() if len(set(c for c, g in v)) > 1}
if multi_cat:
    print("Functions used in multiple categories:")
    for func, refs in sorted(multi_cat.items()):
        cats = set(c for c, g in refs)
        graphs = [g for c, g in refs]
        print(f"  {func}: categories={cats}, graphs={graphs}")
else:
    print("No functions used in multiple categories!")
