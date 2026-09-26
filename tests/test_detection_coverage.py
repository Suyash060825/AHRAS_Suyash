"""
Tests for Threat-Informed Detection Coverage Engine (Research Frontier A / EXP-22)
----------------------------------------------------------------------------------
Validates:
- ATT&CK Implementation Catalog structure and 10 tactics
- Telemetry adequacy and observability formula O(i) = 1(R_req ⊆ R_avail)
- Mathematical invariant: IC(t) <= TC(t) <= 1.0
- 5-Tier Coverage Depth level assignment (L0 to L4)
- Evasion and mutation robustness analysis
- End-to-end JSON and LaTeX report generation and schema compliance
"""

import json
from pathlib import Path
import pytest

from coverage.implementation_catalog import (
    TechniqueImplementation,
    TechniqueDefinition,
    ImplementationCatalog,
    get_default_catalog,
)
from coverage.telemetry_mapper import (
    TelemetryMapper,
    TelemetryObservationResult,
    DEFAULT_AHRAS_AVAILABLE_TELEMETRY,
)
from coverage.technique_mapper import (
    TechniqueMapper,
    MappedDetector,
)
from coverage.detection_analyzer import (
    DetectionAnalyzer,
    ImplementationAnalysisResult,
)
from coverage.coverage_calculator import (
    CoverageCalculator,
    TechniqueCoverageSummary,
    SystemCoverageReport,
)
from coverage.coverage_report import (
    CoverageReporter,
    generate_coverage_artifacts,
)


class TestImplementationCatalog:
    def test_default_catalog_breadth(self):
        catalog = get_default_catalog()
        tactics = catalog.get_tactics()
        # Must cover all 10 core enterprise tactics
        expected_tactics = {
            "Execution", "Persistence", "Privilege Escalation", "Defense Evasion",
            "Credential Access", "Discovery", "Lateral Movement", "Collection",
            "Command and Control", "Impact"
        }
        for t in expected_tactics:
            assert t in tactics, f"Tactic {t} missing from catalog"

        assert catalog.count_techniques() >= 15
        assert catalog.count_implementations() >= 45

    def test_implementation_contracts(self):
        catalog = get_default_catalog()
        for impl in catalog.get_all_implementations():
            assert impl.technique_id.startswith("T")
            assert impl.implementation_id.startswith(impl.technique_id)
            assert impl.execution_modality in {
                "host_cli", "host_api", "registry", "network_flow",
                "file_system", "cloud_api", "encrypted_session", "memory_injection"
            }
            assert len(impl.required_telemetry_fields) > 0
            assert isinstance(impl.sample_event, dict)
            assert isinstance(impl.mutations, list)


class TestTelemetryMapper:
    def test_observability_logic(self):
        mapper = TelemetryMapper(available_telemetry={"process.cmd", "process.name"})
        
        # All required present -> observable
        assert mapper.is_observable(["process.cmd"]) is True
        assert mapper.is_observable(["process.cmd", "process.name"]) is True
        
        # Missing field -> unobservable
        assert mapper.is_observable(["process.cmd", "registry.key_path"]) is False
        assert mapper.is_observable(["unknown.field"]) is False
        assert mapper.is_observable([]) is False

    def test_evaluate_implementation(self):
        mapper = TelemetryMapper(available_telemetry={"process.cmd", "process.name"})
        impl = TechniqueImplementation(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="Execution",
            implementation_id="T1059.001-IMPL-TEST",
            vector_name="test_vector",
            description="test description",
            execution_modality="host_cli",
            required_telemetry_fields=["process.cmd", "registry.key_path"],
        )
        res = mapper.evaluate_implementation(impl)
        assert res.observable is False
        assert res.available_fields == ["process.cmd"]
        assert res.missing_fields == ["registry.key_path"]
        assert res.adequacy_ratio == 0.50

    def test_telemetry_mutation_and_gaps(self):
        catalog = get_default_catalog()
        mapper = TelemetryMapper()
        gaps = mapper.get_telemetry_gaps(catalog)
        assert isinstance(gaps, dict)
        # Should identify known gaps like registry, kernel ETW, etc.
        all_missing = [field for sub in gaps.values() for field in sub]
        assert any("registry" in f or "kernel" in f or "etw" in f for f in all_missing)


class TestTechniqueMapper:
    def test_default_mappings(self):
        mapper = TechniqueMapper()
        mapped_techs = mapper.get_all_mapped_techniques()
        assert "T1059.001" in mapped_techs
        assert "T1046" in mapped_techs
        assert "T1486" in mapped_techs
        assert "T1498.001" in mapped_techs

        # Look up by implementation
        detectors = mapper.get_detectors_for_implementation("T1046-IMPL-01")
        assert len(detectors) >= 1
        detector_ids = [d.detector_id for d in detectors]
        assert "NET-001" in detector_ids or "StatEngine" in detector_ids


class TestDetectionAnalyzer:
    def test_unobservable_implementation_returns_zero(self):
        mapper = TelemetryMapper(available_telemetry=set())  # Sensor blackout
        analyzer = DetectionAnalyzer(telemetry_mapper=mapper)
        catalog = get_default_catalog()
        impl = catalog.get_all_implementations()[0]

        res = analyzer.analyze_implementation(impl)
        assert res.observable is False
        assert res.detected is False
        assert res.empirical_precision == 0.0
        assert res.empirical_recall == 0.0
        assert res.evasion_robustness == 0.0

    def test_observable_detection_and_mutations(self):
        analyzer = DetectionAnalyzer()
        catalog = get_default_catalog()
        # Find port scan implementation
        impl = [i for i in catalog.get_all_implementations() if i.implementation_id == "T1046-IMPL-01"][0]
        res = analyzer.analyze_implementation(impl)
        assert res.observable is True
        assert res.detected is True
        assert "NET-001" in res.rule_matches
        assert res.empirical_precision >= 0.80
        assert res.empirical_recall == 1.0
        assert res.evasion_robustness > 0.0


class TestCoverageCalculator:
    def test_depth_hierarchy_assignment(self):
        calc = CoverageCalculator()
        
        # L0: unobservable
        r0 = ImplementationAnalysisResult(
            implementation_id="T-01", technique_id="T", tactic="Tac", vector_name="v",
            observable=False, detected=False, detecting_engines=[], rule_matches=[],
            empirical_precision=0.0, empirical_recall=0.0, evasion_robustness=0.0
        )
        assert calc.assign_implementation_depth(r0).depth_level == 0

        # L1: observable but not detected
        r1 = ImplementationAnalysisResult(
            implementation_id="T-02", technique_id="T", tactic="Tac", vector_name="v",
            observable=True, detected=False, detecting_engines=[], rule_matches=[],
            empirical_precision=0.0, empirical_recall=0.0, evasion_robustness=0.0
        )
        assert calc.assign_implementation_depth(r1).depth_level == 1

        # L2: detected but fragile (low robustness)
        r2 = ImplementationAnalysisResult(
            implementation_id="T-03", technique_id="T", tactic="Tac", vector_name="v",
            observable=True, detected=True, detecting_engines=["signature"], rule_matches=["NET-001"],
            empirical_precision=0.85, empirical_recall=1.0, evasion_robustness=0.30
        )
        assert calc.assign_implementation_depth(r2).depth_level == 2

        # L3: validated detection
        r3 = ImplementationAnalysisResult(
            implementation_id="T-04", technique_id="T", tactic="Tac", vector_name="v",
            observable=True, detected=True, detecting_engines=["signature"], rule_matches=["NET-001"],
            empirical_precision=0.85, empirical_recall=1.0, evasion_robustness=0.90
        )
        assert calc.assign_implementation_depth(r3).depth_level == 3

    def test_mathematical_invariants(self):
        reporter = CoverageReporter()
        data = reporter.generate_report_data()
        
        sum_data = data["summary"]
        # Fundamental invariant: IC <= TC <= 1.0
        assert sum_data["macro_implementation_coverage"] <= sum_data["macro_telemetry_coverage"]
        assert sum_data["micro_implementation_coverage"] <= sum_data["micro_telemetry_coverage"]
        assert 0.0 <= sum_data["macro_telemetry_coverage"] <= 1.0
        assert 0.0 <= sum_data["macro_implementation_coverage"] <= 1.0

        for tech in data["techniques"]:
            assert tech["implementation_coverage"] <= tech["telemetry_coverage"] + 1e-6
            assert 0 <= tech["depth_level"] <= 4


class TestCoverageReportAndArtifacts:
    def test_artifact_generation(self, tmp_path):
        json_path = tmp_path / "DETECTION_COVERAGE_REPORT.json"
        latex_path = tmp_path / "detection_coverage.tex"

        data = generate_coverage_artifacts(
            output_json_path=json_path,
            output_latex_path=latex_path,
        )

        assert json_path.exists()
        assert latex_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        assert loaded_json["experiment_id"] == "EXP-22"
        assert len(loaded_json["techniques"]) >= 15
        assert len(loaded_json["tactics"]) >= 10
        assert len(loaded_json["system_actionable_recommendations"]) > 0

        with open(latex_path, "r", encoding="utf-8") as f:
            latex_content = f.read()

        assert r"\begin{table*}" in latex_content
        assert r"\caption{\textbf{Threat-Informed Detection Coverage (EXP-22)" in latex_content
        assert r"\end{table*}" in latex_content
