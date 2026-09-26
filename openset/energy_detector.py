"""
AHRAS Energy-Based Out-Of-Distribution & Unknown Attack Detector
---------------------------------------------------------------
Implements Free Energy score formulation (Liu et al., NeurIPS 2020)
for robust zero-day attack identification without softmax overconfidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnergyDetectionOutput:
    """
    Result of free energy OOD detection.
    """
    free_energy: float
    is_out_of_distribution: bool
    temperature: float
    energy_threshold: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "free_energy": round(float(self.free_energy), 4),
            "is_out_of_distribution": self.is_out_of_distribution,
            "temperature": self.temperature,
            "energy_threshold": round(float(self.energy_threshold), 4),
            "confidence": round(float(self.confidence), 4),
        }


class EnergyBasedOODDetector:
    """
    Energy scoring function: E(x; T) = -T * log( sum_k exp(f_k(x) / T) )
    Higher energy indicates out-of-distribution / unseen zero-day instance.
    """
    def __init__(self, temperature: float = 1.0, percentile_threshold: float = 95.0) -> None:
        self.temperature = temperature
        self.percentile_threshold = percentile_threshold
        self.energy_threshold: float = 0.0
        self.is_calibrated: bool = False

    def compute_energy(self, logits: np.ndarray) -> float:
        """
        Computes negative log-sum-exp free energy.
        """
        T = self.temperature
        scaled_logits = logits / T
        # Log-sum-exp trick for numerical stability
        max_val = np.max(scaled_logits)
        lse = max_val + np.log(np.sum(np.exp(scaled_logits - max_val)))
        return float(-T * lse)

    def calibrate(self, in_distribution_logits: np.ndarray) -> float:
        """
        Calibrates the energy threshold on known training/validation logits.
        """
        energies = [self.compute_energy(l) for l in in_distribution_logits]
        self.energy_threshold = float(np.percentile(energies, self.percentile_threshold))
        self.is_calibrated = True
        return self.energy_threshold

    def evaluate(self, logits: np.ndarray) -> EnergyDetectionOutput:
        """
        Evaluates whether a test instance exceeds the energy threshold.
        """
        e = self.compute_energy(logits)
        # Samples with higher energy than threshold are OOD
        is_ood = e > self.energy_threshold

        # Normalized sigmoid confidence of being in-distribution
        delta = self.energy_threshold - e
        conf = 1.0 / (1.0 + np.exp(-delta))

        return EnergyDetectionOutput(
            free_energy=e,
            is_out_of_distribution=is_ood,
            temperature=self.temperature,
            energy_threshold=self.energy_threshold,
            confidence=float(conf),
        )
