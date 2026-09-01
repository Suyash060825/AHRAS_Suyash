import re

file_path = "federated/fed_learning.py"
with open(file_path, "r") as f:
    content = f.read()

# Fix the incorrect insertion
content = re.sub(r"@dataclass\s*\n\s*def create_noniid_client_splits", "def create_noniid_client_splits", content)

with open(file_path, "w") as f:
    f.write(content)
