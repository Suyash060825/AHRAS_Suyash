"""
AHRAS Temporal Split & Chronological Stream Generator
-----------------------------------------------------
Generates strictly chronological, non-shuffled streaming event sequences
modeling canonical concept drift regimes:
1. Baseline Stationary
2. Sudden (Abrupt) Drift
3. Gradual Covariate Shift
4. Recurring Seasonality Drift
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class DriftRegimeType(str, enum.Enum):
    BASELINE_STATIONARY = "baseline_stationary"
    SUDDEN_DRIFT = "sudden_drift"
    GRADUAL_DRIFT = "gradual_drift"
    RECURRING_DRIFT = "recurring_drift"


@dataclass
class StreamingTemporalEvent:
    """
    A single event in a chronological streaming evaluation sequence.
    """
    step_index: int
    timestamp: float                    # Strictly monotonic epoch timestamp
    features: np.ndarray                # Normalized feature vector in R^d
    ground_truth_label: int             # 0 = Benign, 1 = Attack
    attack_family: str                  # e.g., "port_scan", "dos_flood", "c2_beacon", "benign"
    regime: DriftRegimeType
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalStreamGenerator:
    """
    Synthesizes long-horizon chronological event streams exhibiting realistic
    cybersecurity non-stationarity without lookahead leakage.
    """
    def __init__(self, feature_dim: int = 16, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)

        # Baseline centroids
        self._benign_mean = np.zeros(feature_dim)
        self._attack_mean = np.ones(feature_dim) * 1.5

    def generate_stream(
        self,
        n_steps: int = 1000,
        regimes: Optional[List[Tuple[DriftRegimeType, int]]] = None,
        base_timestamp: float = 1700000000.0,
        time_step_sec: float = 60.0,
    ) -> List[StreamingTemporalEvent]:
        """
        Produces a chronological list of events traversing designated drift regimes.
        """
        if regimes is None:
            # Default sequence: 250 baseline, 250 sudden drift, 250 gradual drift, 250 recurring
            q = n_steps // 4
            regimes = [
                (DriftRegimeType.BASELINE_STATIONARY, q),
                (DriftRegimeType.SUDDEN_DRIFT, q),
                (DriftRegimeType.GRADUAL_DRIFT, q),
                (DriftRegimeType.RECURRING_DRIFT, n_steps - 3 * q),
            ]

        stream: List[StreamingTemporalEvent] = []
        cur_time = base_timestamp
        cur_step = 0

        # State tracking for gradual and recurring drift
        drift_offset = np.zeros(self.feature_dim)

        for regime, count in regimes:
            for i in range(count):
                cur_step += 1
                cur_time += time_step_sec + self.py_rng.uniform(-5.0, 5.0)

                # Determine if attack (30% attack base rate)
                is_attack = (self.py_rng.random() < 0.30)
                label = 1 if is_attack else 0

                # Compute feature vector based on regime
                if regime == DriftRegimeType.BASELINE_STATIONARY:
                    base_mu = self._attack_mean if is_attack else self._benign_mean
                    feat = base_mu + self.rng.normal(0, 0.4, size=self.feature_dim)
                    family = "known_dos_scan" if is_attack else "normal_traffic"

                elif regime == DriftRegimeType.SUDDEN_DRIFT:
                    # Sudden jump: attack mean shifts radically, benign shifts slightly
                    sudden_shift = np.array([2.5 if j % 2 == 0 else -1.5 for j in range(self.feature_dim)])
                    base_mu = (self._attack_mean + sudden_shift) if is_attack else (self._benign_mean + 0.5)
                    feat = base_mu + self.rng.normal(0, 0.5, size=self.feature_dim)
                    family = "zero_day_evasion" if is_attack else "workload_spike"

                elif regime == DriftRegimeType.GRADUAL_DRIFT:
                    # Linear accumulation of covariate drift
                    progress = float(i) / max(1, count)
                    drift_offset = np.linspace(0, 2.0, self.feature_dim) * progress
                    base_mu = (self._attack_mean + drift_offset) if is_attack else (self._benign_mean + drift_offset * 0.5)
                    feat = base_mu + self.rng.normal(0, 0.4, size=self.feature_dim)
                    family = "gradual_polymorphic" if is_attack else "infrastructure_migration"

                elif regime == DriftRegimeType.RECURRING_DRIFT:
                    # Diurnal oscillation (sinusoidal recurring pattern)
                    phase = math.sin(2.0 * math.pi * float(i) / 50.0)
                    seasonal_offset = np.ones(self.feature_dim) * phase * 1.2
                    base_mu = (self._attack_mean + seasonal_offset) if is_attack else (self._benign_mean + seasonal_offset * 0.3)
                    feat = base_mu + self.rng.normal(0, 0.35, size=self.feature_dim)
                    family = "recurring_c2_burst" if is_attack else "nightly_backup"

                stream.append(StreamingTemporalEvent(
                    step_index=cur_step,
                    timestamp=round(cur_time, 2),
                    features=feat,
                    ground_truth_label=label,
                    attack_family=family,
                    regime=regime,
                ))

        return stream
