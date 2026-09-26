"""
AHRAS Counterfactual Verifier
-----------------------------
Validates counterfactual explanations against formal safety and operational criteria:
  1. Target Validity: Confirms the intervention successfully reduces risk below threshold.
  2. Sparsity (L0 norm): Quantifies minimal intervention footprint.
  3. Proximity (L1/L2 distance): Measures perturbation cost.
  4. Actionability & Immutability: Enforces that non-actionable features are never modified.
  5. Plausibility / Physical Realizability: Bounds features within empirical domain domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CounterfactualVerificationResult:
    """
    Evaluation verdict for a counterfactual intervention.
    """
    is_valid: bool
    is_actionable: bool
    l0_sparsity: int
    l1_proximity: float
    l2_proximity: float
    risk_factual: float
    risk_counterfactual: float
    risk_reduction: float
    immutable_feature_violations: List[str]
    out_of_bounds_violations: List[str]
    efficiency_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "is_actionable": self.is_actionable,
            "l0_sparsity": self.l0_sparsity,
            "l1_proximity": round(self.l1_proximity, 4),
            "l2_proximity": round(self.l2_proximity, 4),
            "risk_factual": round(self.risk_factual, 4),
            "risk_counterfactual": round(self.risk_counterfactual, 4),
            "risk_reduction": round(self.risk_reduction, 4),
            "immutable_feature_violations": self.immutable_feature_violations,
            "out_of_bounds_violations": self.out_of_bounds_violations,
            "efficiency_ratio": round(self.efficiency_ratio, 4),
        }


class CounterfactualVerifier:
    """
    Automated verification engine for counterfactual explanations.
    """
    def __init__(
        self,
        immutable_feature_indices: Optional[Set[int]] = None,
        feature_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.immutable_indices = immutable_feature_indices or set()
        self.feature_bounds = feature_bounds or {}
        self.feature_names = feature_names or []

    def verify_counterfactual(
        self,
        x_factual: np.ndarray,
        x_counterfactual: np.ndarray,
        predict_func: Callable[[np.ndarray], float],
        target_risk_threshold: float = 0.40,
    ) -> CounterfactualVerificationResult:
        """
        Verifies validity, sparsity, proximity, and actionability of a counterfactual intervention.
        """
        delta = x_counterfactual - x_factual
        modified_indices = np.where(np.abs(delta) > 1e-6)[0]
        l0 = len(modified_indices)
        l1 = float(np.sum(np.abs(delta)))
        l2 = float(np.linalg.norm(delta))

        # 1. Target Validity
        r_factual = float(predict_func(x_factual))
        r_counterfactual = float(predict_func(x_counterfactual))
        is_valid = r_counterfactual <= target_risk_threshold
        risk_reduction = max(0.0, r_factual - r_counterfactual)

        # 2. Immutability checks
        immutable_violations = []
        for idx in modified_indices:
            if idx in self.immutable_indices:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                immutable_violations.append(name)

        # 3. Domain Bounds checks
        bounds_violations = []
        for idx in modified_indices:
            if idx in self.feature_bounds:
                low, high = self.feature_bounds[idx]
                val = x_counterfactual[idx]
                if val < low or val > high:
                    name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                    bounds_violations.append(f"{name} ({val:.2f} not in [{low}, {high}])")

        is_actionable = len(immutable_violations) == 0 and len(bounds_violations) == 0

        # Efficiency ratio: Risk reduction per unit L1 distance
        eff = risk_reduction / max(1e-4, l1)

        return CounterfactualVerificationResult(
            is_valid=is_valid,
            is_actionable=is_actionable,
            l0_sparsity=l0,
            l1_proximity=l1,
            l2_proximity=l2,
            risk_factual=r_factual,
            risk_counterfactual=r_counterfactual,
            risk_reduction=risk_reduction,
            immutable_feature_violations=immutable_violations,
            out_of_bounds_violations=bounds_violations,
            efficiency_ratio=eff,
        )
