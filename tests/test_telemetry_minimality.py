"""
Tests for Telemetry Adequacy & Data Minimality Engine (Research Frontier B / EXP-23)
-----------------------------------------------------------------------------------
Validates:
- TelemetryRequirementsRegistry and field partitioning
- Cross-tactic feature importance and indispensability scoring
- Multi-dimensional adequacy scoring (C_field, C_time, C_entity, C_causal)
- Epistemic uncertainty penalties on incomplete telemetry
- Degraded telemetry stress testing (100% -> 75% -> 50% -> 25%)
- Fleet-wide data volume reduction calculation and Pareto frontier
- End-to-end artifact generation and JSON/LaTeX export
"""

import json
from pathlib import Path
import pytest

from telemetry.telemetry_requirements import (
    TelemetryRequirementProfile,
    TelemetryVolumeMetric,
    TelemetryRequirementsRegistry,
    get_default_requirements_registry,
)
from telemetry.feature_importance_mapper import (
    FeatureImportanceMapper,
    FieldImportanceSummary,
)
from telemetry.telemetry_analyzer import (
    TelemetryAdequacyScore,
    TelemetryAdequacyAuditor,
    DegradedTelemetryStageResult,
)
from telemetry.data_minimality import (
    DataMinimalityOptimizer,
    MinimalityProfileResult,
    ParetoFrontierPoint,
)
from evaluation.run_telemetry_minimality import run_benchmark


class TestTelemetryRequirements:
    def test_requirements_registry_coverage(self):
        reg = get_default_requirements_registry()
        profiles = reg.get_all_profiles()
        assert len(profiles) >= 15

        # Check tactics represented
        tactics = {p.tactic for p in profiles}
        assert "Execution" in tactics
        assert "Credential Access" in tactics
        assert "Discovery" in tactics
        assert "Lateral Movement" in tactics
        assert "Impact" in tactics

    def test_volume_metrics_consistency(self):
        reg = get_default_requirements_registry()
        for p in reg.get_all_profiles():
            vol = p.volume
            # raw >= normalized > minimal
            assert vol.raw_bytes_per_event >= vol.normalized_bytes_per_event
            assert vol.normalized_bytes_per_event > vol.minimal_bytes_per_event
            assert 0.0 < vol.volume_reduction_ratio < 100.0


class TestFeatureImportanceMapper:
    def test_field_importance_calculation(self):
        mapper = FeatureImportanceMapper()
        summaries = mapper.compute_field_importance()
        assert len(summaries) > 0

        # Check classification types
        classifications = {s.classification for s in summaries}
        assert "INDISPENSABLE" in classifications
        assert "OPTIONAL_REDUNDANT" in classifications

        # Check top indispensable fields
        top = mapper.get_top_indispensable_fields(5)
        assert len(top) == 5
        top_names = [s.field_name for s in top]
        # Common indispensable fields must be present
        assert any("cmd_line" in n or "port" in n or "packets" in n for n in top_names)

        for s in summaries:
            assert 0.0 <= s.importance_score <= 1.0
            assert 0.0 <= s.failure_rate_if_removed <= 100.0


class TestTelemetryAdequacyAuditor:
    def test_audit_event_adequacy_dimensions(self):
        auditor = TelemetryAdequacyAuditor()
        reg = auditor.registry
        profile = reg.get_profile("T1059.001-IMPL-01")

        # Full event
        event_full = {
            "event_id": "evt-1234",
            "time": 1700000000.0,
            "device": {"hostname": "corp-host-01"},
            "src_endpoint": {"ip": "10.0.0.5"},
            "actor": {
                "user": {"name": "suyash"},
                "process": {
                    "name": "powershell.exe",
                    "cmd_line": "powershell.exe -enc ...",
                    "pid": 4120,
                }
            },
            "process": {"parent_name": "explorer.exe", "parent_pid": 1000},
        }

        score = auditor.audit_event_adequacy(event_full, profile)
        assert 0.0 <= score.field_completeness <= 1.0
        assert score.temporal_completeness == 1.0
        assert score.entity_completeness == 1.0
        assert score.causal_completeness == 1.0
        assert score.composite_adequacy > 0.80
        assert score.observability_score >= 0.80
        assert score.is_incomplete is False
        assert score.uncertainty_penalty == 0.0

    def test_incomplete_telemetry_penalization(self):
        auditor = TelemetryAdequacyAuditor()
        reg = auditor.registry
        profile = reg.get_profile("T1059.001-IMPL-01")

        # Event missing mandatory cmd_line
        event_stripped = {
            "time": 1700000000.0,
            "device": {"hostname": "corp-host-01"},
            "actor": {"process": {"pid": 4120}},
        }

        score = auditor.audit_event_adequacy(event_stripped, profile)
        assert score.is_incomplete is True
        assert score.uncertainty_penalty > 0.0
        assert score.observability_score < 0.80

    def test_degraded_stress_testing_monotonicity(self):
        auditor = TelemetryAdequacyAuditor()
        stages = auditor.run_degraded_telemetry_stress_test()
        assert len(stages) == 4

        # As retained fields decrease, recall and F1 decrease, uncertainty increases
        for i in range(1, len(stages)):
            assert stages[i].retained_fields_pct < stages[i-1].retained_fields_pct
            assert stages[i].detection_recall_pct <= stages[i-1].detection_recall_pct
            assert stages[i].mean_uncertainty >= stages[i-1].mean_uncertainty


class TestDataMinimalityOptimizer:
    def test_fleet_optimization_math(self):
        optimizer = DataMinimalityOptimizer()
        res = optimizer.optimize_fleet_minimality()

        # Volume reduction should exceed 60%
        assert res["overall_volume_reduction_pct"] > 60.0
        assert res["mean_full_schema_bytes"] > res["mean_minimal_schema_bytes"]

        # All vectors retain detection capability under minimal schema
        for v in res["vector_results"]:
            assert v.detection_preserved is True
            assert v.volume_reduction_pct > 50.0

    def test_pareto_frontier_properties(self):
        optimizer = DataMinimalityOptimizer()
        res = optimizer.optimize_fleet_minimality()
        pareto = res["pareto_frontier"]

        assert len(pareto) >= 3
        # Must have at least one strictly optimal operating point
        optimal_points = [p for p in pareto if p.is_pareto_optimal]
        assert len(optimal_points) >= 2


class TestMinimalityArtifacts:
    def test_end_to_end_benchmark_run(self, tmp_path):
        report = run_benchmark()
        assert report["experiment_id"] == "EXP-23"
        assert "executive_answers" in report
        assert report["executive_answers"]["volume_reduction_without_detection_loss_pct"] > 60.0

        json_path = Path("evaluation/results/TELEMETRY_MINIMALITY_REPORT.json")
        latex_path = Path("publication/tables/telemetry_minimality.tex")

        assert json_path.exists()
        assert latex_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiment_id"] == "EXP-23"
        assert len(data["pareto_frontier"]) > 0

        with open(latex_path, "r", encoding="utf-8") as f:
            latex = f.read()
        assert r"\begin{table*}" in latex
        assert r"Pareto Frontier" in latex
