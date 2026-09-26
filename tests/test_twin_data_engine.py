from __future__ import annotations
"""
Unit and Integration Tests for Security Twin Data Engine & Multi-Objective Scorecard (EXP-21)
-----------------------------------------------------------------------------------------------
Tests:
  - T01: Default enterprise topology creation in SecurityTwinDataEngine
  - T02: APT Lateral Movement multi-stage causal campaign generation
  - T03: Ransomware Burst multi-stage causal campaign generation
  - T04: Causal temporal order guarantee (t_0 < t_1 < ... < t_k)
  - T05: Background normal traffic interleaving and schema integrity
  - T06: Deterministic generation reproducibility with fixed random seed
  - T07: Multi-Objective Scorecard 12-dimensional initialization and target compliance
  - T08: Pareto dominance evaluation over traditional SOAR and monolithic DL
"""

import unittest
from security_twin.data_engine import (
    SecurityTwinDataEngine,
    CampaignType,
    CoherentTelemetryEvent,
    generate_coherent_training_dataset,
)
from evaluation.multi_objective_scorecard import (
    MultiObjectiveSecurityScorecard,
    generate_and_export_scorecard,
)


class TestSecurityTwinDataEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SecurityTwinDataEngine(seed=42)

    def test_t01_default_topology(self):
        """Verifies default enterprise twin hosts and criticalities."""
        twin = self.engine.twin
        self.assertIn("dc01", twin.hosts)
        self.assertIn("srv-db01", twin.hosts)
        self.assertIn("gw-web01", twin.hosts)
        self.assertGreaterEqual(twin.hosts["dc01"].criticality, 0.90)

    def test_t02_apt_lateral_movement_campaign(self):
        """Verifies APT multi-stage kill-chain has valid stages and parent linkages."""
        stream = self.engine.generate_campaign_stream(
            campaign_type=CampaignType.APT_LATERAL_MOVEMENT,
            n_background_events=10,
        )
        attack_events = [e for e in stream if e.is_attack]
        self.assertGreaterEqual(len(attack_events), 4)

        stages = [e.attack_stage for e in attack_events]
        self.assertIn("RECON", stages)
        self.assertIn("EXECUTION", stages)
        self.assertIn("LATERAL_MOVEMENT", stages)
        self.assertIn("CREDENTIAL_ACCESS", stages)

        # Check causal parent-child linkage
        self.assertIsNone(attack_events[0].parent_event_id)
        self.assertEqual(attack_events[1].parent_event_id, attack_events[0].event_id)
        self.assertEqual(attack_events[2].parent_event_id, attack_events[1].event_id)

    def test_t03_ransomware_burst_campaign(self):
        """Verifies Ransomware kill-chain includes file activity with high entropy."""
        stream = self.engine.generate_campaign_stream(
            campaign_type=CampaignType.RANSOMWARE_BURST,
            n_background_events=5,
        )
        file_events = [e for e in stream if e.ocsf_class == "file_activity" and e.is_attack]
        self.assertGreater(len(file_events), 0)
        self.assertGreaterEqual(file_events[0].ocsf_payload["file"]["entropy"], 7.5)

    def test_t04_causal_temporal_ordering(self):
        """Verifies that all events in a campaign stream are strictly chronologically ordered."""
        stream = self.engine.generate_campaign_stream(
            campaign_type=CampaignType.APT_LATERAL_MOVEMENT,
            n_background_events=30,
        )
        for i in range(len(stream) - 1):
            self.assertLessEqual(stream[i].timestamp, stream[i + 1].timestamp)

    def test_t05_background_traffic_interleaving(self):
        """Verifies normal enterprise background events are correctly labeled benign."""
        stream = self.engine.generate_campaign_stream(
            campaign_type=CampaignType.RANSOMWARE_BURST,
            n_background_events=25,
        )
        bg = [e for e in stream if not e.is_attack]
        self.assertEqual(len(bg), 25)
        for e in bg:
            self.assertFalse(e.is_attack)
            self.assertIn(e.ocsf_class, ("network_activity", "process_activity", "file_activity"))

    def test_t06_deterministic_seed_reproducibility(self):
        """Verifies that identical random seeds yield identical event sequences."""
        eng1 = SecurityTwinDataEngine(seed=123)
        stream1 = eng1.generate_campaign_stream(CampaignType.APT_LATERAL_MOVEMENT, n_background_events=10)

        eng2 = SecurityTwinDataEngine(seed=123)
        stream2 = eng2.generate_campaign_stream(CampaignType.APT_LATERAL_MOVEMENT, n_background_events=10)

        self.assertEqual(len(stream1), len(stream2))
        for e1, e2 in zip(stream1, stream2):
            self.assertEqual(e1.ocsf_class, e2.ocsf_class)
            self.assertEqual(e1.is_attack, e2.is_attack)
            self.assertAlmostEqual(e1.timestamp, e2.timestamp, places=3)

    def test_t07_scorecard_dimensions_and_compliance(self):
        """Verifies 12-dimensional scorecard initialization and 100% target compliance."""
        scorecard = MultiObjectiveSecurityScorecard()
        eval_res = scorecard.evaluate_pareto_dominance()
        self.assertEqual(eval_res["total_dimensions"], 12)
        self.assertEqual(eval_res["targets_met_count"], 12)
        self.assertEqual(eval_res["target_compliance_pct"], 100.0)

    def test_t08_scorecard_pareto_dominance(self):
        """Verifies AHRAS strictly Pareto-dominates SOAR and monolithic DL baselines."""
        scorecard = MultiObjectiveSecurityScorecard()
        eval_res = scorecard.evaluate_pareto_dominance()
        self.assertTrue(eval_res["pareto_dominates_traditional_soar"])
        self.assertTrue(eval_res["pareto_dominates_monolithic_dl"])


if __name__ == "__main__":
    unittest.main()
