import re
with open("features/graph_explorer.py", encoding="utf-8") as f:
    source = f.read()

# Check for duplicate graph names
name_pattern = re.compile(r'^\s*"(.+?)":\s*\{')
names = name_pattern.findall(source)
from collections import Counter
dupes = {k: v for k, v in Counter(names).items() if v > 1}
print(f"Duplicate graph names: {dupes}")

# Check raincloud_widget references
count = source.count("raincloud_widget")
print(f"raincloud_widget references: {count}")
