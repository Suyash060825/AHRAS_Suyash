"""
AHRAS Coverage Report Generator
-------------------------------
Generates:
1. Structured detection gap JSON report: evaluation/results/DETECTION_COVERAGE_REPORT.json
2. Publication LaTeX summary table: publication/tables/detection_coverage.tex
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from coverage.implementation_catalog import ImplementationCatalog
from coverage.telemetry_mapper import TelemetryMapper
from coverage.technique_mapper import TechniqueMapper
from coverage.detection_analyzer import DetectionAnalyzer
from coverage.coverage_calculator import CoverageCalculator, SystemCoverageReport, TechniqueCoverageSummary

log = logging.getLogger(__name__)


RECOMMENDATION_TEMPLATES: Dict[str, List[str]] = {
    "T1059.001": [
        "Deploy Microsoft-Windows-DotNETRuntime ETW tracing to detect in-process C# PowerShell runspaces.",
        "Add memory inspection for System.Management.Automation assemblies loaded without powershell.exe.",
    ],
    "T1059.004": [
        "Instrument eBPF sched_process_exec tracepoints to monitor execution inside ephemeral container namespaces.",
    ],
    "T1053.005": [
        "Deploy registry auditing on HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Schedule\\TaskCache.",
        "Enable file integrity monitoring (FIM) on %SystemRoot%\\System32\\Tasks directory.",
    ],
    "T1543.002": [
        "Add auditd rules for file writes targeting /etc/systemd/system and user-level ~/.config/systemd/user.",
    ],
    "T1548.001": [
        "Audit dynamic linker environment variables (LD_PRELOAD, LD_LIBRARY_PATH) on setuid binary execution.",
    ],
    "T1078.004": [
        "Integrate cloud identity provider (IdP) risk scoring and impossible travel anomaly detection into cloud_api normalizer.",
    ],
    "T1070.004": [
        "Deploy raw block device driver monitoring to intercept IOCTL_VOLSNAP shadow storage volume destruction.",
    ],
    "T1027": [
        "Integrate steganographic analysis and entropy variance filters for secondary file drops.",
    ],
    "T1003.001": [
        "Enable LSA protection (RunAsPPL) and deploy kernel object handle auditing on lsass.exe OpenProcess calls.",
    ],
    "T1110.001": [
        "Implement cross-source entity sliding-window authentication failure accumulator to catch low-and-slow sprays.",
    ],
    "T1046": [
        "Deploy 24-hour time-decay stateful IP tracking to capture distributed port-per-hour scanning patterns.",
    ],
    "T1526": [
        "Build baseline behavioral profile of cloud service enumeration APIs (Describe*, Get*) to catch low-rate reconnaissance.",
    ],
    "T1021.002": [
        "Instrument Windows Named Pipe monitoring to intercept \\pipe\\psexecsvc and lateral administrative IPC$ sessions.",
    ],
    "T1021.001": [
        "Correlate SSH/SOCKS port forwarding tunnels with localhost RDP connections to detect proxied lateral movement.",
    ],
    "T1005": [
        "Enable process file handle lock monitoring to detect unauthorized read access to locked browser SQLite cookie databases.",
    ],
    "T1074": [
        "Enable NTFS alternate data stream (ADS) creation telemetry (:stream_name).",
    ],
    "T1071.001": [
        "Implement DNS query correlation and JA3/JA4 TLS client fingerprinting to detect domain-fronted C2 channels.",
    ],
    "T1573.002": [
        "Deploy obfs4/packet-shaping entropy detection filters on edge proxy interfaces.",
    ],
    "T1486": [
        "Implement partial file entropy delta monitoring to catch in-place database page encryption that preserves file headers.",
    ],
    "T1498.001": [
        "Deploy hardware NIC offload telemetry for sub-millisecond SYN/UDP flood surge detection.",
    ],
    "T1499": [
        "Configure reverse proxy worker thread monitoring and connection keep-alive timeout triggers against Slowloris.",
    ],
}


class CoverageReporter:
    """
    Produces formatted JSON gap analysis reports and LaTeX publication tables.
    """
    def __init__(
        self,
        catalog: Optional[ImplementationCatalog] = None,
        telemetry_mapper: Optional[TelemetryMapper] = None,
        technique_mapper: Optional[TechniqueMapper] = None,
        analyzer: Optional[DetectionAnalyzer] = None,
        calculator: Optional[CoverageCalculator] = None,
    ) -> None:
        from coverage.implementation_catalog import get_default_catalog
        self.catalog = catalog or get_default_catalog()
        self.telemetry_mapper = telemetry_mapper or TelemetryMapper()
        self.technique_mapper = technique_mapper or TechniqueMapper()
        self.analyzer = analyzer or DetectionAnalyzer(self.telemetry_mapper, self.technique_mapper)
        self.calculator = calculator or CoverageCalculator()

    def generate_report_data(self) -> Dict[str, Any]:
        """
        Executes analysis and formats full reporting dictionary.
        """
        analysis_results = self.analyzer.analyze_catalog(self.catalog)
        system_report = self.calculator.calculate_system_report(self.catalog, analysis_results)
        telemetry_gaps = self.telemetry_mapper.get_telemetry_gaps(self.catalog)

        tech_data: List[Dict[str, Any]] = []
        for ts in system_report.technique_summaries:
            gaps = telemetry_gaps.get(ts.technique_id, [])
            recs = RECOMMENDATION_TEMPLATES.get(ts.technique_id, [
                f"Enhance telemetry and detection logic for uncovered vectors of {ts.technique_name}."
            ])
            
            # Map detecting rules and engines across covered vectors
            detecting_engines = set()
            rule_hits = set()
            for ir in ts.implementation_results:
                if ir.detected:
                    detecting_engines.update(ir.detecting_engines)
                    rule_hits.update(ir.rule_matches)

            tech_data.append({
                "technique_id": ts.technique_id,
                "technique_name": ts.technique_name,
                "tactic": ts.tactic,
                "total_implementations": ts.total_implementations,
                "observable_implementations": ts.observable_implementations,
                "detected_implementations": ts.detected_implementations,
                "implementation_coverage": ts.implementation_coverage,
                "telemetry_coverage": ts.telemetry_coverage,
                "depth_level": ts.depth_level,
                "depth_label": ts.depth_label,
                "covered_vectors": ts.covered_vectors,
                "uncovered_vectors": ts.uncovered_vectors,
                "detecting_engines": sorted(list(detecting_engines)),
                "rule_hits": sorted(list(rule_hits)),
                "telemetry_gaps": gaps,
                "actionable_recommendations": recs,
            })

        tactic_data: List[Dict[str, Any]] = []
        for tac in system_report.tactic_summaries:
            tactic_data.append({
                "tactic": tac.tactic,
                "technique_count": tac.technique_count,
                "total_implementations": tac.total_implementations,
                "observable_implementations": tac.observable_implementations,
                "detected_implementations": tac.detected_implementations,
                "implementation_coverage": tac.implementation_coverage,
                "telemetry_coverage": tac.telemetry_coverage,
                "depth_distribution": tac.depth_distribution,
            })

        # Aggregated system-wide recommendations
        system_recommendations: List[str] = [
            "Bridge Persistence Telemetry Gap: Integrate Windows TaskCache registry auditing and systemd unit file integrity monitoring.",
            "Bridge Execution Memory Gap: Deploy CLR assembly load tracing (ETW) to detect in-process unmanaged PowerShell runspaces.",
            "Bridge Lateral Movement IPC Gap: Capture Windows Named Pipe connection events to expose PsExec and remote service control execution.",
            "Bridge Low-and-Slow Credential Gap: Add sliding-window cross-host authentication failure aggregators to detect distributed credential sprays.",
        ]

        report = {
            "experiment_id": "EXP-22",
            "benchmark_name": "Threat-Informed Detection Coverage Benchmark",
            "summary": {
                "total_techniques": system_report.total_techniques,
                "total_implementations": system_report.total_implementations,
                "observable_implementations": system_report.observable_implementations,
                "detected_implementations": system_report.detected_implementations,
                "micro_implementation_coverage": system_report.micro_implementation_coverage,
                "macro_implementation_coverage": system_report.macro_implementation_coverage,
                "micro_telemetry_coverage": system_report.micro_telemetry_coverage,
                "macro_telemetry_coverage": system_report.macro_telemetry_coverage,
                "weakest_tactics": system_report.weakest_tactics,
                "depth_distribution_implementations": system_report.depth_distribution_implementations,
                "depth_distribution_techniques": system_report.depth_distribution_techniques,
            },
            "techniques": tech_data,
            "tactics": tactic_data,
            "system_actionable_recommendations": system_recommendations,
        }
        return report

    def generate_latex_table(self, report_data: Dict[str, Any]) -> str:
        """
        Formats report into publication-quality LaTeX table.
        """
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Threat-Informed Detection Coverage (EXP-22): Concrete Implementation vs Telemetry Observability Depth}}",
            r"\label{tab:threat_informed_coverage}",
            r"\begin{tabular}{llrcccc}",
            r"\toprule",
            r"\textbf{Technique ID} & \textbf{Tactic} & \textbf{Impls} & \textbf{TC} & \textbf{IC} & \textbf{Depth} & \textbf{Primary Detectors} \\",
            r"\midrule",
        ]

        for t in report_data.get("techniques", []):
            tid = t["technique_id"]
            tactic = t["tactic"]
            impls = f"{t['detected_implementations']}/{t['total_implementations']}"
            tc = f"{t['telemetry_coverage'] * 100:.1f}\\%"
            ic = f"{t['implementation_coverage'] * 100:.1f}\\%"
            depth = f"L{t['depth_level']}"
            detectors = ", ".join(t.get("rule_hits", [])[:2]) or "None"
            lines.append(f"{tid} & {tactic} & {impls} & {tc} & {ic} & {depth} & \\texttt{{{detectors}}} \\\\")

        sum_data = report_data.get("summary", {})
        macro_tc = f"{sum_data.get('macro_telemetry_coverage', 0) * 100:.1f}\\%"
        macro_ic = f"{sum_data.get('macro_implementation_coverage', 0) * 100:.1f}\\%"
        tot_impls = f"{sum_data.get('detected_implementations', 0)}/{sum_data.get('total_implementations', 0)}"

        lines.extend([
            r"\midrule",
            f"\\textbf{{Macro Aggregate}} & \\textbf{{10 Tactics}} & {tot_impls} & \\textbf{{{macro_tc}}} & \\textbf{{{macro_ic}}} & \\textbf{{L3/L4}} & \\textbf{{Hybrid Defense}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        return "\n".join(lines)


def generate_coverage_artifacts(
    output_json_path: Path | str = "evaluation/results/DETECTION_COVERAGE_REPORT.json",
    output_latex_path: Path | str = "publication/tables/detection_coverage.tex",
) -> Dict[str, Any]:
    """
    Executes coverage assessment and serializes both JSON and LaTeX artifacts.
    """
    reporter = CoverageReporter()
    data = reporter.generate_report_data()
    latex = reporter.generate_latex_table(data)

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    out_latex = Path(output_latex_path)
    out_latex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_latex, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    return data
