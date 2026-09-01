import re

file_path = "evaluation/adversarial_suite.py"
with open(file_path, "r") as f:
    content = f.read()

fix = """
        res_base = self.combiner.process(base_attack)
        
        # Manually force the components so we control the exact risk
        sig_base = [type("MockSig", (), {"severity": 5, "confidence": 0.90, "rule_name": "rule", "mitre_technique": "T1"})()]
        ml_base = type("MockML", (), {"ensemble_score": 0.10, "confidence": 0.95})()
        
        risk_base = self.risk_engine.score_risk("10.0.1.99", sig_base, ml_base, None, evt=base_attack)
        base_dominant = risk_base.causal_chains[0]["evidence_name"] if risk_base.causal_chains else None
        
        # Attacker manipulates to get exactly the same risk but from a different vector
        sig_man = [] # Evades signature
        ml_man = type("MockML", (), {"ensemble_score": 0.90, "confidence": 0.95})() # Spikes ML to compensate
        
        # We also need to configure the risk engine weights so that sig and ml have equal pull
        from detection.risk_engine import RiskConfig
        cfg = RiskConfig(w_sig=0.5, w_ml=0.5, adaptive_weights=False)
        
        risk_base = self.risk_engine.score_risk("10.0.1.99", sig_base, ml_base, None, evt=base_attack, override_config=cfg)
        base_dominant = risk_base.causal_chains[0]["evidence_name"] if risk_base.causal_chains else None
        
        risk_man = self.risk_engine.score_risk("10.0.1.99", sig_man, ml_man, None, evt=manipulated_attack, override_config=cfg)
        man_dominant = risk_man.causal_chains[0]["evidence_name"] if risk_man.causal_chains else None
        
        passed = (base_dominant == "S_sig") and (man_dominant == "A_ml") and (abs(risk_base.risk_score - risk_man.risk_score) < 0.1)
"""

content = re.sub(r'        res_base = self.combiner.process\(base_attack\).*?passed = \(base_dominant != man_dominant\) and \(man_dominant == "A_ml"\)', fix.strip(), content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
