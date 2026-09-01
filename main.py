import os
import time
import json
import logging
from typing import Dict, Any

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("AHRAS-Main")

# Import all core modules
from detection.hybrid_engine import get_combiner
from detection.risk_engine import get_risk_engine, RiskConfig
from response.orchestrator import ResponseOrchestrator
from adaptive_learning.ztre import get_ztre
from graph.tgnn import get_tgnn

class AHRASPipeline:
    """
    The Single Unified AHRAS Pipeline.
    No research modes, no dev modes. Just the final production pipeline.
    """
    def __init__(self, soar_webhook_url: str = None):
        log.info("Initializing Final AHRAS Pipeline...")
        
        # 1. Threat Combiner (ML, Stat, Signatures)
        self.combiner = get_combiner()
        
        # 2. Risk Engine (Dempster-Shafer, TGNN, ZTRE, MITRE)
        # Using the optimal configuration established in Phase 2
        self.risk_config = RiskConfig(
            adaptive_weights=True,
            use_selective_gate=True,
            use_graph=True
        )
        self.risk_engine = get_risk_engine()
        self.risk_engine.config = self.risk_config
        
        # 3. Temporal Graph & ZTRE (Singletons)
        self.tgnn = get_tgnn()
        self.ztre = get_ztre()
        
        # 4. SOAR Orchestrator
        self.soar = ResponseOrchestrator()
        # If a URL is provided, enable real webhooks, otherwise simulate
        if soar_webhook_url:
            self.soar.execution_mode = "REAL_PRODUCTION"
            self.soar._dry_run = False
            os.environ["SOAR_WEBHOOK_URL"] = soar_webhook_url
        else:
            self.soar.execution_mode = "SIMULATED"
            self.soar._dry_run = True

        log.info("AHRAS Pipeline Ready.")

    def process_telemetry(self, entity_key: str, event: Dict[str, Any]):
        """
        End-to-End processing of a single OCSF-normalized telemetry event.
        """
        log.info(f"--- Processing New Event for {entity_key} ---")
        
        # Step 1: Base Threat Detection (Combiner)
        detection_res = self.combiner.process(event)
        
        sig_matches = detection_res.signature_matches if detection_res else []
        ml_res = detection_res.anomaly_result if detection_res else None
        stat_res = detection_res.stat_result if detection_res else None
        
        # Injecting simulation rule hit based on the event payload for demonstration
        if "rule_name" in event:
            class MockSig:
                def __init__(self, rule_name):
                    self.rule_name = rule_name
                    self.severity = 5
                    self.confidence = 0.95
            sig_matches.append(MockSig(event["rule_name"]))

        # Step 2: Risk Scoring, TGNN Lateral Movement, ZTRE Scoping, & Conformal Gating
        risk_result = self.risk_engine.score_risk(
            entity_key=entity_key,
            sig_matches=sig_matches,
            ml_res=ml_res,
            stat_res=stat_res,
            evt=event
        )
        
        log.info(f"Final Risk Score: {risk_result.risk_score:.3f} | Severity: {risk_result.severity}")
        log.info(f"MITRE Tactics: {risk_result.mitre_techniques}")
        
        # Step 3: Explanation & State
        log.info(f"Causal Explanation: {risk_result.explanation}")
        
        # Step 4: Autonomous SOAR Response (Governed by Conformal Gate)
        log.info(f"Conformal Gate Decision: {risk_result.autonomy_decision}")
        self.soar.evaluate_and_respond(risk_result)
        
        return risk_result


if __name__ == "__main__":
    # Example Usage of the Single Working Copy
    pipeline = AHRASPipeline()
    
    # Simulate a realistic OCSF Network Activity Event (e.g. Lateral Movement / Bruteforce)
    sample_event = {
        "class_name": "Network Activity",
        "time": time.time(),
        "src_endpoint": {"ip": "10.0.1.99"},
        "dst_endpoint": {"ip": "192.168.1.10", "port": 22}, # SSH
        "network_traffic": {
            "packets": 5000, 
            "bytes": 250000
        },
        "rule_name": "SSH Bruteforce" # Triggers MITRE T1110.001
    }
    
    # Process the event
    pipeline.process_telemetry(entity_key="10.0.1.99", event=sample_event)
    
    # Simulate a follow-up event to trigger TGNN & ZTRE restriction
    time.sleep(1)
    follow_up_event = {
        "class_name": "Network Activity",
        "time": time.time(),
        "src_endpoint": {"ip": "192.168.1.10"}, # Lateral Movement hop
        "dst_endpoint": {"ip": "10.0.1.50", "port": 445}, # SMB
        "network_traffic": {
            "packets": 200, 
            "bytes": 50000
        },
        "rule_name": "Lateral Movement" # Triggers MITRE T1021
    }
    
    pipeline.process_telemetry(entity_key="10.0.1.99", event=follow_up_event)
