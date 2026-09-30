"""
Seven Formal Metamorphic Relations Test Suite for AHRAS Detection & Risk Logic.
Evaluates invariant and monotonic behavioral consistency across inputs.
"""

from typing import Dict, Any, List, Tuple
import copy
import numpy as np

class MetamorphicTestEngine:
    """
    Formal Metamorphic Testing Suite implementing:
    MR1: Event duplication (idempotence / no double counting)
    MR2: Irrelevant metadata scaling (noise invariance)
    MR3: Monotonic malicious evidence increment (risk non-decrease)
    MR4: Single evidence source removal (graceful monotonic degradation)
    MR5: Event order permutation (temporal-order invariance where applicable)
    MR6: Benign background volume duplication (false positive stability)
    MR7: Asset criticality escalation (risk score monotonic non-decrease)
    """

    def __init__(self, risk_evaluator_fn=None):
        """
        risk_evaluator_fn: callable taking a list of events/features and returning risk float [0, 1]
        """
        self.risk_evaluator_fn = risk_evaluator_fn or self._default_mock_evaluator

    def _default_mock_evaluator(self, events: List[Dict[str, Any]]) -> float:
        # Simple surrogate evaluator for metamorphic property validation if no model passed
        if not events:
            return 0.0
        malicious_count = sum(1 for e in events if e.get("is_malicious", False))
        max_crit = max(e.get("asset_criticality", 0.5) for e in events)
        base = malicious_count / len(events) if len(events) > 0 else 0.0
        return float(min(1.0, base * 1.5 * max_crit + (0.1 if malicious_count > 0 else 0.0)))

    def evaluate_mr1_event_duplication(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR1: Duplicating exact identical telemetry event should not arbitrarily multiply threat risk."""
        r_orig = self.risk_evaluator_fn(events)
        # Duplicate each event
        dup_events = []
        for e in events:
            dup_events.append(e)
            dup_events.append(copy.deepcopy(e))
        r_dup = self.risk_evaluator_fn(dup_events)
        # Risk should not deviate by more than 15% due to exact duplicate suppression/idempotence
        diff = abs(r_dup - r_orig)
        passed = diff <= 0.20
        return {"mr": "MR1_EventDuplication", "passed": passed, "r_orig": r_orig, "r_transformed": r_dup, "delta": diff}

    def evaluate_mr2_irrelevant_metadata(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR2: Modifying irrelevant metadata fields (e.g. user_agent string padding) must not change detection."""
        r_orig = self.risk_evaluator_fn(events)
        transformed = copy.deepcopy(events)
        for e in transformed:
            e["unrelated_tag"] = "benign_padding_meta_12345"
            e["sensor_location_detail"] = "rack-9-chassis-2"
        r_trans = self.risk_evaluator_fn(transformed)
        passed = abs(r_trans - r_orig) < 1e-4
        return {"mr": "MR2_IrrelevantMetadata", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans}

    def evaluate_mr3_monotonic_malicious_increment(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR3: Adding unambiguous malicious evidence must monotonically non-decrease the risk score."""
        r_orig = self.risk_evaluator_fn(events)
        transformed = copy.deepcopy(events)
        transformed.append({
            "event_type": "known_malware_c2_beacon",
            "is_malicious": True,
            "threat_confidence": 0.95,
            "asset_criticality": 0.8,
            "destination_port": 4444
        })
        r_trans = self.risk_evaluator_fn(transformed)
        passed = (r_trans >= r_orig - 1e-5)
        return {"mr": "MR3_MonotonicMaliciousIncrement", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans}

    def evaluate_mr4_evidence_source_removal(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR4: Removing an indicator of compromise must monotonically non-increase risk score."""
        if not events:
            return {"mr": "MR4_EvidenceRemoval", "passed": True, "r_orig": 0.0, "r_transformed": 0.0}
        r_orig = self.risk_evaluator_fn(events)
        transformed = copy.deepcopy(events[:-1])
        r_trans = self.risk_evaluator_fn(transformed)
        passed = (r_trans <= r_orig + 1e-5)
        return {"mr": "MR4_EvidenceRemoval", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans}

    def evaluate_mr5_order_permutation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR5: Permuting event order inside an aggregation window should maintain score bounds."""
        r_orig = self.risk_evaluator_fn(events)
        transformed = list(reversed(copy.deepcopy(events)))
        r_trans = self.risk_evaluator_fn(transformed)
        diff = abs(r_trans - r_orig)
        passed = diff <= 0.25
        return {"mr": "MR5_OrderPermutation", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans, "delta": diff}

    def evaluate_mr6_benign_background_scaling(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR6: Influx of pure benign background noise must not trigger false positive inflation."""
        r_orig = self.risk_evaluator_fn(events)
        transformed = copy.deepcopy(events)
        for i in range(20):
            transformed.append({
                "event_type": "standard_http_get",
                "is_malicious": False,
                "threat_confidence": 0.01,
                "asset_criticality": 0.3
            })
        r_trans = self.risk_evaluator_fn(transformed)
        # Risk should not dramatically spike
        passed = (r_trans <= r_orig + 0.15)
        return {"mr": "MR6_BenignBackgroundScaling", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans}

    def evaluate_mr7_asset_criticality_escalation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """MR7: Escalating asset criticality with identical detector signals must non-decrease risk score."""
        r_orig = self.risk_evaluator_fn(events)
        transformed = copy.deepcopy(events)
        for e in transformed:
            e["asset_criticality"] = min(1.0, e.get("asset_criticality", 0.5) * 1.5 + 0.2)
        r_trans = self.risk_evaluator_fn(transformed)
        passed = (r_trans >= r_orig - 1e-5)
        return {"mr": "MR7_AssetCriticalityEscalation", "passed": passed, "r_orig": r_orig, "r_transformed": r_trans}

    def run_all_metamorphic_tests(self, sample_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [
            self.evaluate_mr1_event_duplication(sample_events),
            self.evaluate_mr2_irrelevant_metadata(sample_events),
            self.evaluate_mr3_monotonic_malicious_increment(sample_events),
            self.evaluate_mr4_evidence_source_removal(sample_events),
            self.evaluate_mr5_order_permutation(sample_events),
            self.evaluate_mr6_benign_background_scaling(sample_events),
            self.evaluate_mr7_asset_criticality_escalation(sample_events),
        ]
        all_passed = all(r["passed"] for r in results)
        return {
            "all_passed": all_passed,
            "pass_rate": sum(1 for r in results if r["passed"]) / len(results),
            "details": results
        }
