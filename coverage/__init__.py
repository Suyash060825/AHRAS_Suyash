"""
AHRAS Threat-Informed Detection Coverage Engine
-----------------------------------------------
Moves detection coverage beyond superficial ATT&CK heatmaps toward:
- Behaviorally distinct concrete execution vectors
- Rigorous telemetry adequacy and observability mapping
- Quantified 5-tier detection depth (L0 to L4)
- Evasion mutation robustness analysis
- Actionable detection gap reporting
"""

from coverage.implementation_catalog import (
    TechniqueImplementation,
    TechniqueDefinition,
    ImplementationCatalog,
    get_default_catalog,
)
from coverage.telemetry_mapper import (
    TelemetryMapper,
    TelemetryObservationResult,
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

__all__ = [
    "TechniqueImplementation",
    "TechniqueDefinition",
    "ImplementationCatalog",
    "get_default_catalog",
    "TelemetryMapper",
    "TelemetryObservationResult",
    "TechniqueMapper",
    "MappedDetector",
    "DetectionAnalyzer",
    "ImplementationAnalysisResult",
    "CoverageCalculator",
    "TechniqueCoverageSummary",
    "SystemCoverageReport",
    "CoverageReporter",
    "generate_coverage_artifacts",
]
