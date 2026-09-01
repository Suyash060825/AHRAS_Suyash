import re

file_path = "response/orchestrator.py"
with open(file_path, "r") as f:
    content = f.read()

# Add requests import
if "import requests" not in content:
    content = content.replace("import time", "import time\nimport requests")

soar_code = """
        # REAL_PRODUCTION Execution Adapters
        try:
            # ── TheHive / Shuffle SOAR REST API Integration ──
            # In a true deployment, this routes to a SOAR webhook
            # e.g., Shuffle webhook or TheHive Alert endpoint
            
            soar_webhook_url = os.environ.get("SOAR_WEBHOOK_URL", "")
            soar_api_key = os.environ.get("SOAR_API_KEY", "")
            
            if soar_webhook_url and mode == "REAL_PRODUCTION":
                headers = {"Authorization": f"Bearer {soar_api_key}", "Content-Type": "application/json"}
                payload = {
                    "title": f"AHRAS Action: {action.action_type}",
                    "description": f"Target: {action.target_identifier} | Risk: {action.risk_score}",
                    "severity": action.severity,
                    "action_id": action.action_id,
                    "source": "AHRAS_Conformal_Gate",
                    "artifacts": [
                        {"dataType": "ip", "data": action.target_identifier} if action.action_type == "BLOCK_IP" else {},
                        {"dataType": "other", "data": action.details}
                    ]
                }
                
                try:
                    resp = requests.post(soar_webhook_url, json=payload, headers=headers, timeout=5.0)
                    if resp.status_code >= 400:
                        log.warning(f"[SOAR] Webhook returned {resp.status_code}: {resp.text}")
                    else:
                        log.info(f"[SOAR] Successfully forwarded {action.action_type} to Orchestration Platform.")
                except requests.RequestException as e:
                    log.error(f"[SOAR] Connection to SOAR failed: {e}")
                    # Fall back to local execution adapters below if SOAR is unreachable

            if action.action_type == "ISOLATE_HOST":
"""

# Import os if not imported
if "import os" not in content:
    content = content.replace("import sys", "import sys\nimport os")

content = re.sub(r'        # REAL_PRODUCTION Execution Adapters\n        try:\n            if action.action_type == "ISOLATE_HOST":', soar_code.strip('\n'), content)

with open(file_path, "w") as f:
    f.write(content)
