"""
AHRAS Open-Set & Unknown Attack Classifier
-----------------------------------------
Unifies OpenMax extreme value re-weighting, Helmholtz Free Energy scoring,
and Mahalanobis latent distance into an open-world intrusion classification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openset.open_max import OpenMaxEngine, OpenMaxOutput
from openset.energy_detector import EnergyBasedOODDetector, EnergyDetectionOutput


@dataclass
class OpenSetVerdict:
    """
    Open-world classification verdict distinguishing known attack classes from unseen zero-days.
    """
    predicted_label: str            # e.g. "BENIGN", "DOS_ATTACK", "UNKNOWN_ZERO_DAY"
    is_known: bool
    is_zero_day: bool
    known_class_idx: Optional[int]
    open_set_anomaly_score: float   # in [0, 1]
    openmax_unknown_prob: float
    free_energy: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "is_known": self.is_known,
            "is_zero_day": self.is_zero_day,
            "known_class_idx": self.known_class_idx,
            "open_set_anomaly_score": round(self.open_set_anomaly_score, 4),
            "openmax_unknown_prob": round(self.openmax_unknown_prob, 4),
            "free_energy": round(self.free_energy, 4),
            "details": self.details,
        }


class OpenSetClassifier:
    """
    Hybrid open-world classifier integrating OpenMax and Energy-based detection.
    """
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        unknown_prob_threshold: float = 0.40,
        energy_temperature: float = 1.0,
    ) -> None:
        self.class_names = class_names or []
        self.unknown_prob_threshold = unknown_prob_threshold
        self.openmax = OpenMaxEngine()
        self.energy_detector = EnergyBasedOODDetector(temperature=energy_temperature)
        self.is_fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits both OpenMax Weibull models and energy thresholds on in-distribution training data.
        """
        self.openmax.fit(logits, labels)
        self.energy_detector.calibrate(logits)
        self.is_fitted = True

    def predict_single(self, logit_vector: np.ndarray) -> OpenSetVerdict:
        """
        Evaluates a single instance logit vector across OpenMax and Energy detectors.
        """
        om_res = self.openmax.predict(logit_vector, unknown_threshold=self.unknown_prob_threshold)
        energy_res = self.energy_detector.evaluate(logit_vector)

        # Composite open-set anomaly score: combination of OpenMax unknown prob and energy
        # Normalize energy into [0, 1]
        energy_norm = 1.0 - energy_res.confidence
        composite_score = 0.60 * om_res.unknown_probability + 0.40 * energy_norm

        is_zero_day = om_res.is_unknown or energy_res.is_out_of_distribution or composite_score >= 0.50
        is_known = not is_zero_day

        if is_zero_day:
            pred_label = "UNKNOWN_ZERO_DAY"
            known_idx = None
        else:
            known_idx = om_res.predicted_class
            if self.class_names and known_idx < len(self.class_names):
                pred_label = self.class_names[known_idx]
            else:
                pred_label = f"KNOWN_CLASS_{known_idx}"

        return OpenSetVerdict(
            predicted_label=pred_label,
            is_known=is_known,
            is_zero_day=is_zero_day,
            known_class_idx=known_idx,
            open_set_anomaly_score=float(composite_score),
            openmax_unknown_prob=om_res.unknown_probability,
            free_energy=energy_res.free_energy,
            details={
                "openmax": om_res.to_dict(),
                "energy": energy_res.to_dict(),
            },
        )

    def predict(self, logits_batch: np.ndarray) -> List[OpenSetVerdict]:
        """Evaluates a batch of logit vectors."""
        return [self.predict_single(vec) for vec in logits_batch]
