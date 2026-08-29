from __future__ import annotations
"""
AHRAS Dependency & Module Import Verification Test
---------------------------------------------------
Verifies that all production modules can be cleanly imported without syntax or import errors.
"""

import unittest
import importlib


class TestModuleImports(unittest.TestCase):
    MODULES_TO_TEST = [
        "config.settings",
        "auth.manager",
        "auth.dependencies",
        "rbac.permissions",
        "rbac.middleware",
        "storage.store",
        "pipeline.bus",
        "normalizer.ocsf_normalizer",
        "normalizer.enrichment",
        "detection.feature_extractor",
        "detection.signature_engine.rules",
        "detection.anomaly_engine.ml_engine",
        "detection.statistical_engine.stat_engine",
        "detection.statistical_engine.peer_group",
        "detection.statistical_engine.trend_engine",
        "detection.statistical_engine.entity_report",
        "detection.hybrid_engine",
        "detection.risk_engine",
        "detection.gnn_engine",
        "detection.xai_explainer",
        "detection.dataset_generator",
        "detection.pipeline",
        "detection.evaluator",
        "ahras.evidence.models",
        "ahras.evidence.ledger",
        "response.orchestrator",
        "threat_intel.intel",
        "threat_intel.stix_ingestor",
        "deception.honeypot_manager",
        "forecast.predictor",
        "adaptive_learning.weight_learner",
        "federated.fed_learning",
        "historical_risk.engine",
        "xai.fidelity_ledger",
        "xai.llm_narrator",
        "evaluation.metrics",
        "evaluation.dataset_loader",
        "evaluation.leakage_audit",
        "evaluation.adversarial_suite",
        "sensors.host_agent",
        "sensors.network_sensor",
        "sensors.cloud_adapter",
        "api.server",
    ]

    def test_clean_imports_across_all_modules(self):
        for mod_name in self.MODULES_TO_TEST:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except Exception as e:
                    self.fail(f"Failed to import production module '{mod_name}': {e}")


if __name__ == "__main__":
    unittest.main()
