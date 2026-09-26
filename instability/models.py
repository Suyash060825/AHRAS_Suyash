from __future__ import annotations
"""
AHRAS Module / Instability — Data Models & Metric Contracts
-----------------------------------------------------------
Defines data structures for temporal prediction tracking, class-flip detection,
trajectory volatility, and calibrated epistemic instability signals.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PredictionRecord:
    """Historical snapshot of an individual prediction and confidence for an entity."""
    timestamp: float
    risk_score: float                   # Risk probability p_t in [0.0, 1.0]
    confidence: float                   # Confidence score c_t in [0.0, 1.0]
    class_label: int                    # Binary classification (0 = benign, 1 = malicious)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstabilityMetrics:
    """
    Multidimensional temporal epistemic instability metrics for an entity over a sliding window.
    """
    confidence_variance: float          # Variance in model confidence Var(c)
    class_flip_count: int               # Count of classification state transitions 0 <-> 1
    class_flip_rate: float              # Flips per observation step in [0.0, 1.0]
    trajectory_instability: float       # Second-order rate of change |Δp_{t+1} - Δp_t|
    prediction_entropy: float           # Mean binary Shannon entropy H(p) in [0.0, 1.0]
    temporal_disagreement: float        # Divergence between short-term and long-term trends
    instability_score: float            # Normalized composite uncertainty signal in [0.0, 1.0]
    recommended_action: str             # "AUTONOMOUS", "MONITOR", "ABSTAIN", "ESCALATE_ANALYST"
    human_review_priority: float        # Calibrated analyst triage priority in [0.0, 1.0]
    autonomy_gated: bool                # True if high instability blocks autonomous actuation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_variance": round(self.confidence_variance, 4),
            "class_flip_count": self.class_flip_count,
            "class_flip_rate": round(self.class_flip_rate, 4),
            "trajectory_instability": round(self.trajectory_instability, 4),
            "prediction_entropy": round(self.prediction_entropy, 4),
            "temporal_disagreement": round(self.temporal_disagreement, 4),
            "instability_score": round(self.instability_score, 4),
            "recommended_action": self.recommended_action,
            "human_review_priority": round(self.human_review_priority, 4),
            "autonomy_gated": self.autonomy_gated,
        }


@dataclass
class EntityStabilityProfile:
    """Complete temporal stability profile for a monitored entity."""
    entity_key: str
    history_length: int
    window_seconds: float
    metrics: InstabilityMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_key": self.entity_key,
            "history_length": self.history_length,
            "window_seconds": round(self.window_seconds, 2),
            "metrics": self.metrics.to_dict(),
        }
