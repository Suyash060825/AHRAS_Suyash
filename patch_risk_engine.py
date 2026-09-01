import re

file_path = "detection/risk_engine.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Imports
imports = """
from xai.causal_explainer import CausalExplainer, CausalReport
from mitre.mapper import enrich_with_mitre, get_mitre_techniques
from adaptive_learning.ztre import get_ztre
from graph.tgnn import get_tgnn
"""
content = content.replace("from xai.causal_explainer import CausalExplainer, CausalReport", imports.strip())

# 2. MITRE and TGNN
# Inside score_risk, after evidence_ledger...
# Let's find where mitre_techs is appended
# for m in sig_matches:
#     if hasattr(m, 'mitre_technique') and m.mitre_technique:
#         mitre_techs.append(m.mitre_technique)

mitre_tgnn_code = """
        for m in sig_matches:
            S_sig = max(S_sig, getattr(m, 'confidence', 0.0) * getattr(m, 'severity', 0.0) / 5.0)
            if hasattr(m, 'mitre_technique') and m.mitre_technique:
                mitre_techs.append(m.mitre_technique)
            
            # [AHRAS-ZTRE/MITRE ADDITION]
            if hasattr(m, 'rule_name') and m.rule_name:
                mapping = enrich_with_mitre(m.rule_name)
                if mapping and mapping["technique_id"] != "T1000":
                    mitre_techs.append(mapping["technique_id"])
                    
        # [AHRAS-TGNN ADDITION] Online Temporal Graph Scoring
        if cfg.use_graph and evt is not None:
            src_ip = evt.get("src_endpoint", {}).get("ip")
            dst_ip = evt.get("dst_endpoint", {}).get("ip")
            timestamp = evt.get("time", time.time())
            if src_ip and dst_ip:
                tgnn = get_tgnn()
                path_pred = tgnn.record_interaction(src_ip, dst_ip, timestamp, severity=max(S_sig, A_ml))
                if path_pred.developing:
                    G_corr = max(G_corr, path_pred.risk_energy * 0.8) # Weight the TGNN energy
"""

content = re.sub(
    r"        for m in sig_matches:\n            S_sig = max\(S_sig, getattr\(m, 'confidence', 0\.0\) \* getattr\(m, 'severity', 0\.0\) / 5\.0\)\n            if hasattr\(m, 'mitre_technique'\) and m\.mitre_technique:\n                mitre_techs\.append\(m\.mitre_technique\)",
    mitre_tgnn_code.strip('\n'),
    content
)

# 3. ZTRE Integration
# Right before returning RiskResult, we call ZTRE
ztre_code = """
        # [AHRAS-ZTRE ADDITION] Continuous Access Scope Evaluation
        ztre = get_ztre()
        ztre_state = ztre.evaluate_session(entity_key, risk_score)
        
        decision_reason += f" ZTRE Scope: {ztre_state.current_scope.name} (TTL={ztre_state.ttl_seconds}s)."

        return RiskResult(
"""

content = content.replace("        return RiskResult(", ztre_code.strip('\n') + "\n        return RiskResult(")

with open(file_path, "w") as f:
    f.write(content)
