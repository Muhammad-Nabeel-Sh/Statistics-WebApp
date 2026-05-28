import re, json, ast

with open("features/graph_explorer.py", encoding="utf-8") as f:
    source = f.read()

# Find the graphs dict boundaries
start = source.index("graphs = {")
brace_count = 0
end = start
for i in range(start, len(source)):
    if source[i] == "{":
        brace_count += 1
    elif source[i] == "}":
        brace_count -= 1
        if brace_count == 0:
            end = i + 1
            break

graph_src = source[start:end]

# Remove widget_function keys
graph_src_clean = re.sub(r',?\s*"widget_function":\s*\w+', "", graph_src)
graph_src_clean = re.sub(r'"widget_function":\s*\w+,?\s*', "", graph_src_clean)

try:
    graphs_meta = ast.literal_eval(graph_src_clean)
    print(f"Parsed {len(graphs_meta)} graph entries")
    with open("assets/graph_metadata.json", "w", encoding="utf-8") as f:
        json.dump(graphs_meta, f, indent=2, ensure_ascii=False)
    print("Written to assets/graph_metadata.json")
except Exception as e:
    print(f"Parse error: {e}")
    # Try more aggressive cleanup
    # Remove lines with widget_function
    lines = graph_src_clean.split("\n")
    clean_lines = [l for l in lines if "widget_function" not in l]
    graph_src_clean2 = "\n".join(clean_lines)
    try:
        graphs_meta = ast.literal_eval(graph_src_clean2)
        print(f"Parsed {len(graphs_meta)} graph entries (fallback)")
        with open("assets/graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(graphs_meta, f, indent=2, ensure_ascii=False)
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
