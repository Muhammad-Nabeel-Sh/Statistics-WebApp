with open("features/graph_explorer_original.py", encoding="utf-8") as f:
    content = f.read()
print(f"Length: {len(content)} chars")
print(f"Has graphs: {'graphs = {' in content}")
print(f"Has render_graph_explorer: {'def render_graph_explorer' in content}")
print(f"First 100: {content[:100]}")
print(f"Last 100: {content[-100:]}")
