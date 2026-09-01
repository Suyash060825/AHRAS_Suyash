import re

file_path = "evaluation/adversarial_suite.py"
with open(file_path, "r") as f:
    content = f.read()

# Fix the test logic
fix_logic = """
        # Check if conformal gate safely rejected the prediction (abstained due to uncertainty)
        abstained = getattr(risk_evasion, "autonomy_decision", "") == "ABSTAIN"
        uncertainty = getattr(risk_evasion, "risk_uncertainty", 0.0)
"""
content = re.sub(r'# Check if conformal gate safely rejected.*?uncertainty = getattr\(risk_evasion, "uncertainty", 0\.0\)', fix_logic.strip(), content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
