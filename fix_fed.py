import re

file_path = "federated/fed_learning.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace the wrong @dataclass placement
content = content.replace("@dataclass\ndef create_noniid_client_splits", "def create_noniid_client_splits")
content = content.replace("class ModelUpdate:", "@dataclass\nclass ModelUpdate:")

with open(file_path, "w") as f:
    f.write(content)
