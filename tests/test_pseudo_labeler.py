from __future__ import annotations
"""
Unit and Integration Tests for AHRAS Confidence-Gated Pseudo-Label Learning (EXP-20)
--------------------------------------------------------------------------------------
Verifies:
  - T01: Confidence extremity gating for attack and benign predictions
  - T02: Strict out-of-distribution (OOD) rejection
  - T03: Epistemic uncertainty rejection and routing to human queue
  - T04: Temporal volatility and instability rejection
  - T05: Cross-modal sensor disagreement rejection
  - T06: Ambiguous decision margin rejection
  - T07: Explicit provenance tracking and sample weight discounting (0.50 vs 1.00)
  - T08: Generation-based quarantine and purge rollback
  - T09: Hybrid ActiveLearner stream triage and query budget reduction
  - T10: Continual learning MultiMemoryReplayBuffer pseudo-compartment isolation
"""

import unittest
from collections import deque

from adaptive_learning.pseudo_labeler import (
    PseudoLabelEngine,
    PseudoLabelRecord,
    PseudoLabelDecision,
    LabelProvenance,
    get_pseudo_label_engine,
)
from adaptive_learning.active_learner import ActiveLearner
from adaptive_learning.weight_learner import MultiMemoryReplayBuffer, FeedbackSample


class TestPseudoLabelEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PseudoLabelEngine(
            conf_high=0.95,
            conf_low=0.05,
            max_uncertainty=0.10,
            max_ood=0.20,
            sample_weight=0.50,
            buffer_capacity=100,
        )

    def test_t01_confidence_gating_attack_and_benign(self):
        """Verifies ultra-confident in-distribution samples receive valid pseudo-labels."""
        # 1. High risk attack
        rec_attack = self.engine.evaluate_sample(
            event_id="EVT-ATTACK-01",
            entity_key="10.0.1.5",
            predicted_risk=0.98,
            confidence=0.95,
            epistemic_uncertainty=0.04,
            ood_score=0.05,
            temporal_instability=0.0,
            cross_modal_consistency=1.0,
            features={"pkt_rate": 1500.0, "entropy": 7.8},
        )
        self.assertEqual(rec_attack.assigned_label, 1)
        self.assertEqual(rec_attack.decision, PseudoLabelDecision.PSEUDO_LABEL_ATTACK.value)
        self.assertEqual(rec_attack.provenance, LabelProvenance.PSEUDO_VALIDATED.value)
        self.assertEqual(rec_attack.sample_weight, 0.50)

        # 2. Low risk benign
        rec_benign = self.engine.evaluate_sample(
            event_id="EVT-BENIGN-01",
            entity_key="10.0.1.12",
            predicted_risk=0.02,
            confidence=0.97,
            epistemic_uncertainty=0.03,
            ood_score=0.04,
            temporal_instability=0.0,
            cross_modal_consistency=1.0,
            features={"pkt_rate": 12.0, "entropy": 3.4},
        )
        self.assertEqual(rec_benign.assigned_label, 0)
        self.assertEqual(rec_benign.decision, PseudoLabelDecision.PSEUDO_LABEL_BENIGN.value)
        self.assertEqual(rec_benign.provenance, LabelProvenance.PSEUDO_VALIDATED.value)

    def test_t02_ood_rejection(self):
        """Verifies that out-of-distribution samples are strictly rejected from pseudo-labeling."""
        rec = self.engine.evaluate_sample(
            event_id="EVT-OOD-01",
            entity_key="10.0.2.1",
            predicted_risk=0.99,  # High risk, but OOD!
            confidence=0.96,
            epistemic_uncertainty=0.05,
            ood_score=0.65,  # Exceeds max_ood 0.20
            temporal_instability=0.0,
            cross_modal_consistency=1.0,
        )
        self.assertIsNone(rec.assigned_label)
        self.assertEqual(rec.decision, PseudoLabelDecision.REJECT_OOD.value)
        self.assertEqual(rec.provenance, LabelProvenance.QUARANTINED.value)
        self.assertIn("OOD score", rec.rejection_reason)

    def test_t03_epistemic_uncertainty_rejection(self):
        """Verifies that high epistemic uncertainty rejects pseudo-label and routes to human queue."""
        rec = self.engine.evaluate_sample(
            event_id="EVT-UNC-01",
            entity_key="10.0.3.4",
            predicted_risk=0.96,
            confidence=0.90,
            epistemic_uncertainty=0.35,  # Exceeds max_uncertainty 0.10
            ood_score=0.08,
            temporal_instability=0.0,
            cross_modal_consistency=1.0,
        )
        self.assertIsNone(rec.assigned_label)
        self.assertEqual(rec.decision, PseudoLabelDecision.ROUTE_HUMAN_ACTIVE_LEARNING.value)
        self.assertIn("Epistemic uncertainty", rec.rejection_reason)

    def test_t04_temporal_instability_rejection(self):
        """Verifies that oscillating predictions over time are rejected."""
        rec = self.engine.evaluate_sample(
            event_id="EVT-OSC-01",
            entity_key="10.0.4.9",
            predicted_risk=0.97,
            confidence=0.92,
            epistemic_uncertainty=0.06,
            ood_score=0.05,
            temporal_instability=0.45,  # Oscillating
            cross_modal_consistency=1.0,
        )
        self.assertIsNone(rec.assigned_label)
        self.assertEqual(rec.decision, PseudoLabelDecision.REJECT_UNSTABLE.value)

    def test_t05_cross_modal_inconsistency_rejection(self):
        """Verifies that conflicting sensor modalities reject pseudo-labeling."""
        rec = self.engine.evaluate_sample(
            event_id="EVT-CONFLICT-01",
            entity_key="10.0.5.1",
            predicted_risk=0.96,
            confidence=0.91,
            epistemic_uncertainty=0.05,
            ood_score=0.05,
            temporal_instability=0.0,
            cross_modal_consistency=0.40,  # Telemetry plane disagreement
        )
        self.assertIsNone(rec.assigned_label)
        self.assertEqual(rec.decision, PseudoLabelDecision.REJECT_INCONSISTENT.value)

    def test_t06_ambiguous_decision_margin_rejection(self):
        """Verifies that mid-range risk (0.05 < R < 0.95) routes to human active learning."""
        rec = self.engine.evaluate_sample(
            event_id="EVT-MARGIN-01",
            entity_key="10.0.6.2",
            predicted_risk=0.52,  # Ambiguous decision boundary
            confidence=0.88,
            epistemic_uncertainty=0.05,
            ood_score=0.05,
            temporal_instability=0.0,
            cross_modal_consistency=1.0,
        )
        self.assertIsNone(rec.assigned_label)
        self.assertEqual(rec.decision, PseudoLabelDecision.ROUTE_HUMAN_ACTIVE_LEARNING.value)

    def test_t07_provenance_tracking_and_sample_weights(self):
        """Verifies explicit provenance and training dataset generation with proper weighting."""
        # 1. Add pseudo-label
        self.engine.evaluate_sample(
            event_id="EVT-P1",
            entity_key="10.0.1.1",
            predicted_risk=0.97,
            confidence=0.95,
            epistemic_uncertainty=0.04,
            ood_score=0.05,
            features={"f1": 1.0},
        )
        # 2. Add human ground truth
        self.engine.add_human_verified_sample(
            event_id="EVT-H1",
            entity_key="10.0.1.2",
            ground_truth_label=1,
            predicted_risk=0.60,
            features={"f1": 0.8},
        )

        dataset = self.engine.get_training_dataset()
        self.assertEqual(len(dataset), 2)

        # Human sample has weight 1.00
        human_sample = [d for d in dataset if d[3] == LabelProvenance.HUMAN_VERIFIED.value][0]
        self.assertEqual(human_sample[2], 1.00)

        # Pseudo sample has discounted weight 0.50
        pseudo_sample = [d for d in dataset if d[3] == LabelProvenance.PSEUDO_VALIDATED.value][0]
        self.assertEqual(pseudo_sample[2], 0.50)

    def test_t08_generation_quarantine_and_purge(self):
        """Verifies atomic quarantine and rollback of pseudo-labels without losing human data."""
        # Gen 1 pseudo label
        self.engine.evaluate_sample(
            event_id="EVT-G1",
            entity_key="10.0.1.1",
            predicted_risk=0.97,
            confidence=0.95,
            epistemic_uncertainty=0.04,
            ood_score=0.05,
        )
        # Advance to Gen 2
        gen2 = self.engine.advance_generation()
        self.assertEqual(gen2, 2)

        # Gen 2 pseudo label
        self.engine.evaluate_sample(
            event_id="EVT-G2",
            entity_key="10.0.1.2",
            predicted_risk=0.98,
            confidence=0.96,
            epistemic_uncertainty=0.03,
            ood_score=0.04,
        )
        # Human ground truth in Gen 2
        self.engine.add_human_verified_sample(
            event_id="EVT-HG2",
            entity_key="10.0.1.3",
            ground_truth_label=0,
        )

        # Drift detected in Gen 2! Quarantine Gen 2 pseudo-labels
        quarantined = self.engine.quarantine_generation(generation=2)
        self.assertEqual(quarantined, 1)

        # Gen 1 pseudo label and Human label should survive
        dataset = self.engine.get_training_dataset()
        self.assertEqual(len(dataset), 2)
        provenances = [d[3] for d in dataset]
        self.assertIn(LabelProvenance.HUMAN_VERIFIED.value, provenances)
        self.assertIn(LabelProvenance.PSEUDO_VALIDATED.value, provenances)

        # Purge quarantined
        purged = self.engine.purge_quarantined()
        self.assertEqual(purged, 1)

    def test_t09_active_learner_hybrid_triage(self):
        """Verifies ActiveLearner dual-tier triage saves human query budget."""
        al = ActiveLearner(budget_per_window=5, window_sec=3600.0, pseudo_label_engine=self.engine)

        # 1. Ultra-confident sample -> Auto accepted as pseudo-label (no human budget spent)
        res1 = al.triage_stream_sample(
            event_id="EVT-AL-01",
            entity_key="10.0.1.1",
            risk_score=0.98,
            confidence=0.96,
            uncertainty=0.03,
            ood_score=0.04,
        )
        self.assertTrue(res1["auto_accepted_pseudo"])
        self.assertFalse(res1["human_queried"])
        self.assertEqual(len(al.get_pending()), 0)

        # 2. Ambiguous sample -> Enqueued for human active learning
        res2 = al.triage_stream_sample(
            event_id="EVT-AL-02",
            entity_key="10.0.1.2",
            risk_score=0.55,
            confidence=0.70,
            uncertainty=0.45,
            ood_score=0.15,
        )
        self.assertFalse(res2["auto_accepted_pseudo"])
        self.assertTrue(res2["human_queried"])
        self.assertEqual(len(al.get_pending()), 1)

    def test_t10_multimemory_replay_buffer_integration(self):
        """Verifies MultiMemoryReplayBuffer correctly routes and balances pseudo-labeled samples."""
        buffer = MultiMemoryReplayBuffer(pseudo_cap=50)

        # Add human attack
        s_human = FeedbackSample(
            src_ip="10.0.1.1",
            label=1,
            components={"sig": 0.9, "ml": 0.8},
            predicted_risk=0.85,
            provenance="HUMAN_VERIFIED",
            sample_weight=1.0,
            generation=1,
        )
        buffer.add_sample(s_human, loss=0.1)

        # Add validated pseudo-label
        s_pseudo = FeedbackSample(
            src_ip="10.0.1.2",
            label=1,
            components={"sig": 0.95, "ml": 0.92},
            predicted_risk=0.96,
            provenance="PSEUDO_VALIDATED",
            sample_weight=0.5,
            generation=1,
        )
        buffer.add_sample(s_pseudo, loss=0.05)

        self.assertEqual(len(buffer.attack_memory), 1)
        self.assertEqual(len(buffer.pseudo_memory), 1)

        # Sample batch
        batch = buffer.sample_balanced_batch(batch_size=4)
        self.assertGreater(len(batch), 0)

        # Quarantine pseudo generation 1
        purged = buffer.quarantine_pseudo_generation(generation=1)
        self.assertEqual(purged, 1)
        self.assertEqual(len(buffer.pseudo_memory), 0)
        # Human memory preserved
        self.assertEqual(len(buffer.attack_memory), 1)


if __name__ == "__main__":
    unittest.main()
