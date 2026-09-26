"""
AHRAS Coverage Calculator
-------------------------
Calculates Implementation Coverage (IC), Telemetry Coverage (TC),
and assigns the 5-Tier Coverage Depth (Level 0 through Level 4)
at both implementation and technique levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coverage.implementation_catalog import ImplementationCatalog, TechniqueDefinition
from coverage.detection_analyzer import ImplementationAnalysisResult


@dataclass
class ImplementationDepthResult:
    """
    Assigned depth level and analysis metrics for a single implementation.
    """
    implementation_id: str
    technique_id: str
    tactic: str
    vector_name: str
    depth_level: int              # 0 to 4
    depth_label: str              # Level 0 (Blind) ... Level 4 (Resilient)
    observable: bool
    detected: bool
    empirical_precision: float
    empirical_recall: float
    evasion_robustness: float
    detecting_engines: List[str]
    rule_matches: List[str]


@dataclass
class TechniqueCoverageSummary:
    """
    Summary metrics for a MITRE ATT&CK technique.
    """
    technique_id: str
    technique_name: str
    tactic: str
    total_implementations: int
    observable_implementations: int
    detected_implementations: int
    implementation_coverage: float   # IC(t)
    telemetry_coverage: float        # TC(t)
    depth_level: int                 # Highest verified tier for this technique (0 to 4)
    depth_label: str
    covered_vectors: List[str]
    uncovered_vectors: List[str]
    implementation_results: List[ImplementationDepthResult] = field(default_factory=list)


@dataclass
class TacticCoverageSummary:
    """
    Aggregated coverage metrics for an entire ATT&CK tactic.
    """
    tactic: str
    technique_count: int
    total_implementations: int
    observable_implementations: int
    detected_implementations: int
    implementation_coverage: float
    telemetry_coverage: float
    depth_distribution: Dict[int, int]  # count of impls at L0..L4


@dataclass
class SystemCoverageReport:
    """
    Full enterprise system coverage report across all evaluated techniques.
    """
    total_techniques: int
    total_implementations: int
    observable_implementations: int
    detected_implementations: int
    micro_implementation_coverage: float
    macro_implementation_coverage: float
    micro_telemetry_coverage: float
    macro_telemetry_coverage: float
    depth_distribution_implementations: Dict[int, int]
    depth_distribution_techniques: Dict[int, int]
    technique_summaries: List[TechniqueCoverageSummary]
    tactic_summaries: List[TacticCoverageSummary]
    weakest_tactics: List[str]


DEPTH_LABELS = {
    0: "Level 0: Blind (Missing Telemetry)",
    1: "Level 1: Telemetry Observable (No Detector)",
    2: "Level 2: Fragile Detection (Low Robustness / Precision)",
    3: "Level 3: Validated Implementation (Verified & Tested)",
    4: "Level 4: Resilient Multi-Vector Defense",
}


class CoverageCalculator:
    """
    Computes mathematical coverage metrics and categorizes 5-tier depth levels.
    """
    def assign_implementation_depth(self, result: ImplementationAnalysisResult) -> ImplementationDepthResult:
        """
        Assigns coverage depth level (0 to 3) for an individual implementation vector.
        Level 4 is assigned at the technique level upon verifying multi-vector resilience.
        """
        if not result.observable:
            level = 0
        elif not result.detected:
            level = 1
        elif result.evasion_robustness < 0.50 or result.empirical_precision < 0.60:
            level = 2
        else:
            level = 3

        return ImplementationDepthResult(
            implementation_id=result.implementation_id,
            technique_id=result.technique_id,
            tactic=result.tactic,
            vector_name=result.vector_name,
            depth_level=level,
            depth_label=DEPTH_LABELS[level],
            observable=result.observable,
            detected=result.detected,
            empirical_precision=result.empirical_precision,
            empirical_recall=result.empirical_recall,
            evasion_robustness=result.evasion_robustness,
            detecting_engines=result.detecting_engines,
            rule_matches=result.rule_matches,
        )

    def evaluate_technique(
        self,
        tech: TechniqueDefinition,
        analysis_results: Dict[str, ImplementationAnalysisResult],
    ) -> TechniqueCoverageSummary:
        """
        Computes technique-level IC, TC, and evaluates whether technique reaches Level 4.
        """
        total = len(tech.implementations)
        if total == 0:
            return TechniqueCoverageSummary(
                technique_id=tech.technique_id,
                technique_name=tech.technique_name,
                tactic=tech.tactic,
                total_implementations=0,
                observable_implementations=0,
                detected_implementations=0,
                implementation_coverage=0.0,
                telemetry_coverage=0.0,
                depth_level=0,
                depth_label=DEPTH_LABELS[0],
                covered_vectors=[],
                uncovered_vectors=[],
                implementation_results=[],
            )

        impl_depths: List[ImplementationDepthResult] = []
        covered_vectors: List[str] = []
        uncovered_vectors: List[str] = []

        obs_count = 0
        det_count = 0
        level_3_count = 0
        distinct_modalities_detected = set()

        for impl in tech.implementations:
            res = analysis_results.get(impl.implementation_id)
            if res is None:
                continue
            depth_res = self.assign_implementation_depth(res)
            impl_depths.append(depth_res)

            if depth_res.observable:
                obs_count += 1
            if depth_res.detected:
                det_count += 1
                covered_vectors.append(impl.vector_name)
                distinct_modalities_detected.add(impl.execution_modality)
            else:
                uncovered_vectors.append(impl.vector_name)

            if depth_res.depth_level >= 3:
                level_3_count += 1

        ic = round(det_count / total, 4)
        tc = round(obs_count / total, 4)

        # Technique depth level assignment
        # Level 4: At least 2 implementations at Level 3 AND multiple modalities or robust multi-vector coverage
        if level_3_count >= 2 and (len(distinct_modalities_detected) >= 2 or ic >= 0.50):
            tech_level = 4
        elif level_3_count >= 1:
            tech_level = 3
        elif det_count >= 1:
            tech_level = 2
        elif obs_count >= 1:
            tech_level = 1
        else:
            tech_level = 0

        return TechniqueCoverageSummary(
            technique_id=tech.technique_id,
            technique_name=tech.technique_name,
            tactic=tech.tactic,
            total_implementations=total,
            observable_implementations=obs_count,
            detected_implementations=det_count,
            implementation_coverage=ic,
            telemetry_coverage=tc,
            depth_level=tech_level,
            depth_label=DEPTH_LABELS[tech_level],
            covered_vectors=covered_vectors,
            uncovered_vectors=uncovered_vectors,
            implementation_results=impl_depths,
        )

    def calculate_system_report(
        self,
        catalog: ImplementationCatalog,
        analysis_results: Dict[str, ImplementationAnalysisResult],
    ) -> SystemCoverageReport:
        """
        Aggregates technique and tactic metrics into a comprehensive system-wide report.
        """
        tech_summaries: List[TechniqueCoverageSummary] = []
        for tech in catalog.get_all_techniques():
            tech_summaries.append(self.evaluate_technique(tech, analysis_results))

        total_techs = len(tech_summaries)
        total_impls = sum(t.total_implementations for t in tech_summaries)
        obs_impls = sum(t.observable_implementations for t in tech_summaries)
        det_impls = sum(t.detected_implementations for t in tech_summaries)

        micro_ic = round(det_impls / total_impls, 4) if total_impls else 0.0
        macro_ic = round(sum(t.implementation_coverage for t in tech_summaries) / total_techs, 4) if total_techs else 0.0

        micro_tc = round(obs_impls / total_impls, 4) if total_impls else 0.0
        macro_tc = round(sum(t.telemetry_coverage for t in tech_summaries) / total_techs, 4) if total_techs else 0.0

        # Distribution of depth levels across implementations
        dist_impl: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for t in tech_summaries:
            for ir in t.implementation_results:
                dist_impl[ir.depth_level] = dist_impl.get(ir.depth_level, 0) + 1

        # Distribution of depth levels across techniques
        dist_tech: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for t in tech_summaries:
            dist_tech[t.depth_level] = dist_tech.get(t.depth_level, 0) + 1

        # Tactic summaries
        tactic_map: Dict[str, List[TechniqueCoverageSummary]] = {}
        for t in tech_summaries:
            tactic_map.setdefault(t.tactic, []).append(t)

        tactic_summaries: List[TacticCoverageSummary] = []
        for tactic_name, t_list in tactic_map.items():
            t_total = sum(t.total_implementations for t in t_list)
            t_obs = sum(t.observable_implementations for t in t_list)
            t_det = sum(t.detected_implementations for t in t_list)
            t_ic = round(t_det / t_total, 4) if t_total else 0.0
            t_tc = round(t_obs / t_total, 4) if t_total else 0.0

            t_depth_dist: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for t in t_list:
                for ir in t.implementation_results:
                    t_depth_dist[ir.depth_level] = t_depth_dist.get(ir.depth_level, 0) + 1

            tactic_summaries.append(TacticCoverageSummary(
                tactic=tactic_name,
                technique_count=len(t_list),
                total_implementations=t_total,
                observable_implementations=t_obs,
                detected_implementations=t_det,
                implementation_coverage=t_ic,
                telemetry_coverage=t_tc,
                depth_distribution=t_depth_dist,
            ))

        # Rank weakest tactics by IC ascending
        sorted_tactics = sorted(tactic_summaries, key=lambda x: (x.implementation_coverage, x.telemetry_coverage))
        weakest_tactics = [ts.tactic for ts in sorted_tactics[:3]]

        return SystemCoverageReport(
            total_techniques=total_techs,
            total_implementations=total_impls,
            observable_implementations=obs_impls,
            detected_implementations=det_impls,
            micro_implementation_coverage=micro_ic,
            macro_implementation_coverage=macro_ic,
            micro_telemetry_coverage=micro_tc,
            macro_telemetry_coverage=macro_tc,
            depth_distribution_implementations=dist_impl,
            depth_distribution_techniques=dist_tech,
            technique_summaries=tech_summaries,
            tactic_summaries=tactic_summaries,
            weakest_tactics=weakest_tactics,
        )
