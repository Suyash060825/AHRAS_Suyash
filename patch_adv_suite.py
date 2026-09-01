import re

file_path = "evaluation/adversarial_suite.py"
with open(file_path, "r") as f:
    content = f.read()

# Add test_9 to the run_full_suite dictionary
content = content.replace('"test_8_ti_poisoning": self.test_threat_intel_poisoning(),\n        }', '"test_8_ti_poisoning": self.test_threat_intel_poisoning(),\n            "test_9_conformal_gate_evasion": self.test_conformal_gate_evasion(),\n        }')

# Create the test function
new_test = """
    # ── Test 9: Conformal Gate Evasion ────────────────────────────────────────
    def test_conformal_gate_evasion(self) -> Dict[str, Any]:
        \"\"\"
        Tests if the Conformal Gate safely abstains when an attacker carefully
        perturbs features to stay just below the detection threshold, rather than
        mistakenly giving a high-confidence false negative.
        \"\"\"
        log.info("[RED-TEAM] Executing Test 9: Conformal Gate Evasion")
        
        # 1. Create a clear attack event
        base_attack = _norm_network({
            "src_ip": "10.0.1.55",
            "dst_port": 445,
            "packet_count": 5000,  # Clear flooding/scanning
            "duration_sec": 1.0,
        })
        
        # Process unperturbed attack
        res_base = self.combiner.process(base_attack)
        sig_matches = res_base.signature_matches if res_base else []
        anomaly_res = res_base.anomaly_result if res_base else None
        stat_res = res_base.stat_result if res_base else None
        
        from detection.risk_engine import RiskConfig
        cfg = RiskConfig(use_selective_gate=True)
        
        # Risk engine uses conformal gate implicitly if configured
        risk_base = self.risk_engine.score_risk(
            "10.0.1.55", sig_matches, anomaly_res, stat_res, evt=base_attack, override_config=cfg
        )
        
        # 2. Perturb event carefully to sit on the boundary (evasion attempt)
        evasion_attack = _norm_network({
            "src_ip": "10.0.1.55",
            "dst_port": 445,
            "packet_count": 850,  # Just below typical threshold, attempting to blend in
            "duration_sec": 10.0, # Slowed down
        })
        
        res_evasion = self.combiner.process(evasion_attack)
        sig_matches_evasion = res_evasion.signature_matches if res_evasion else []
        anomaly_res_evasion = res_evasion.anomaly_result if res_evasion else None
        stat_res_evasion = res_evasion.stat_result if res_evasion else None
        
        risk_evasion = self.risk_engine.score_risk(
            "10.0.1.55", sig_matches_evasion, anomaly_res_evasion, stat_res_evasion, 
            evt=evasion_attack, override_config=cfg
        )
        
        # Check if conformal gate safely rejected the prediction (abstained due to uncertainty)
        # In AHRAS, abstention sets abstained=True or forces risk to default/intervention
        abstained = getattr(risk_evasion, "abstained", False)
        uncertainty = getattr(risk_evasion, "uncertainty", 0.0)
        
        # Passed if the gate identified high uncertainty or abstained during the evasion attempt
        passed = abstained or (uncertainty > 0.4)
        
        return {
            "passed": passed,
            "base_risk": risk_base.risk_score,
            "evasion_risk": risk_evasion.risk_score,
            "evasion_uncertainty": uncertainty,
            "abstained": abstained,
            "description": "Gate correctly identifies boundary perturbation as highly uncertain" if passed else "Gate failed to abstain on boundary perturbation"
        }
"""

content = content.replace("    # ── Test 1: Feature Manipulation & Evasion ────────────────────────────────", new_test + "\n    # ── Test 1: Feature Manipulation & Evasion ────────────────────────────────")

with open(file_path, "w") as f:
    f.write(content)
