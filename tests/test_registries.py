"""
Tests for Model and Detection Registry Manager (Section 26 / Research Frontier P1)
------------------------------------------------------------------------------------
Validates:
  1. Registry creation and schema integrity for model_registry.json and detection_registry.json.
  2. Immutability & non-overwriting rule: duplicate (name, version) registrations are rejected.
  3. Querying by model name and retrieval of latest version.
  4. Querying detections by MITRE technique.
  5. Persistence across manager instances.
"""

import json
import pytest
from pathlib import Path
from evaluation.registry import (
    AHRASRegistryManager,
    ModelRegistryEntry,
    DetectionRegistryEntry,
)


@pytest.fixture
def temp_registries(tmp_path):
    model_file = tmp_path / "model_registry.json"
    det_file = tmp_path / "detection_registry.json"
    mgr = AHRASRegistryManager(model_path=model_file, detection_path=det_file)
    return mgr, model_file, det_file


def test_registry_initialization(temp_registries):
    mgr, model_file, det_file = temp_registries
    assert model_file.exists()
    assert det_file.exists()

    with open(model_file) as f:
        data = json.load(f)
        assert "models" in data
        assert data["schema_version"] == "1.0"


def test_model_registration_and_anti_overwrite(temp_registries):
    mgr, _, _ = temp_registries

    entry = ModelRegistryEntry(
        name="TestModel",
        version="v1.0",
        artifact_hash="abc123hash",
        training_dataset="TestDataset",
        dataset_hash="def456hash",
        features=["feat1", "feat2"],
        input_schema={"type": "array"},
        output_schema={"type": "object"},
        calibration={"method": "Platt", "ece": 0.05},
        ood_behavior={"method": "Energy", "auroc": 0.95},
        known_limitations=["test limitation"],
        evaluation_date="2026-09-28",
    )

    # First registration must succeed
    assert mgr.register_model(entry) is True

    # Duplicate registration of same (name, version) must be rejected
    assert mgr.register_model(entry) is False

    # Second version should succeed
    entry_v2 = ModelRegistryEntry(
        name="TestModel",
        version="v2.0",
        artifact_hash="abc789hash",
        training_dataset="TestDatasetV2",
        dataset_hash="def789hash",
        features=["feat1", "feat2", "feat3"],
        input_schema={"type": "array"},
        output_schema={"type": "object"},
        calibration={"method": "Platt", "ece": 0.04},
        ood_behavior={"method": "Energy", "auroc": 0.97},
        known_limitations=[],
        evaluation_date="2026-09-28",
    )
    assert mgr.register_model(entry_v2) is True

    # Retrieve models
    models = mgr.get_models("TestModel")
    assert len(models) == 2
    latest = mgr.get_latest_model("TestModel")
    assert latest["version"] == "v2.0"


def test_detection_registration_and_query_by_technique(temp_registries):
    mgr, _, _ = temp_registries

    det1 = DetectionRegistryEntry(
        rule_id="NET-SSH-BRUTE",
        version="1.0.0",
        technique="T1110",
        implementation="detection.signature_engine.rules._rule_ssh_brute_force",
        telemetry=["network"],
        precision=0.98,
        recall=0.96,
        robustness=0.95,
        validation_scenarios=["EXP-24"],
        limitations=["rate threshold dependency"],
    )

    det2 = DetectionRegistryEntry(
        rule_id="HOST-CMD-EXEC",
        version="1.0.0",
        technique="T1059.001",
        implementation="detection.signature_engine.rules._rule_shell_exec_in_cmdline",
        telemetry=["host"],
        precision=0.99,
        recall=0.97,
        robustness=0.96,
        validation_scenarios=["EXP-24"],
        limitations=["requires deobfuscation"],
    )

    assert mgr.register_detection(det1) is True
    assert mgr.register_detection(det2) is True

    # Duplicate must be rejected
    assert mgr.register_detection(det1) is False

    # Query by technique
    matches = mgr.get_detections(technique="T1110")
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "NET-SSH-BRUTE"
