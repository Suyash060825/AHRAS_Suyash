"""
AHRAS Streaming Prequential Evaluation & Concept Drift Harness
--------------------------------------------------------------
Implements strict test-then-train streaming evaluation without future data leakage,
online statistical drift detection (Page-Hinkley, ADWIN), verification delay queues,
and longitudinal memory profiling under sudden, gradual, and recurring concept drift.
"""

from streaming.temporal_split import (
    StreamingTemporalEvent,
    TemporalStreamGenerator,
    DriftRegimeType,
)
from streaming.drift_detector import (
    PageHinkleyDriftDetector,
    ADWINDriftDetector,
    StreamingDriftSignal,
)
from streaming.drift_metrics import (
    PrequentialWindowMetrics,
    PrequentialMetricsAccumulator,
)
from streaming.prequential_runner import (
    PrequentialEvaluationRunner,
    PrequentialSimulationResult,
    ModelAdaptationStrategy,
)

__all__ = [
    "StreamingTemporalEvent",
    "TemporalStreamGenerator",
    "DriftRegimeType",
    "PageHinkleyDriftDetector",
    "ADWINDriftDetector",
    "StreamingDriftSignal",
    "PrequentialWindowMetrics",
    "PrequentialMetricsAccumulator",
    "PrequentialEvaluationRunner",
    "PrequentialSimulationResult",
    "ModelAdaptationStrategy",
]
