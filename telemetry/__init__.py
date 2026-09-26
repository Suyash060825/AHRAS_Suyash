"""
AHRAS Telemetry Adequacy & Data Minimality Engine
-------------------------------------------------
Quantifies telemetry requirements, field indispensability, adequacy scoring,
and data volume reduction without compromising threat detection capability.
"""

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
)
from telemetry.data_minimality import (
    DataMinimalityOptimizer,
    MinimalityProfileResult,
    ParetoFrontierPoint,
)

__all__ = [
    "TelemetryRequirementProfile",
    "TelemetryVolumeMetric",
    "TelemetryRequirementsRegistry",
    "get_default_requirements_registry",
    "FeatureImportanceMapper",
    "FieldImportanceSummary",
    "TelemetryAdequacyScore",
    "TelemetryAdequacyAuditor",
    "DataMinimalityOptimizer",
    "MinimalityProfileResult",
    "ParetoFrontierPoint",
]
