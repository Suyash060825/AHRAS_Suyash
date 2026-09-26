"""
AHRAS Faithfulness Evaluator
----------------------------
Quantifies explanation faithfulness across Sufficiency, Comprehensiveness,
and Monotonicity under systematic feature masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FaithfulnessMetrics:
    """
    Quantifies explanation faithfulness and monotonicity.
    """
    top_k: int
    sufficiency_score: float        # Risk preserved using only top-k features
    comprehensiveness_score: float  # Risk drop when top-k features are removed
    monotonicity_score: float       # Percentage of monotonic transitions [0, 1]
    is_faithful: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "sufficiency_score": round(self.sufficiency_score, 4),
            "comprehensiveness_score": round(self.comprehensiveness_score, 4),
            "monotonicity_score": round(self.monotonicity_score, 4),
            "is_faithful": self.is_faithful,
        }


class FaithfulnessEvaluator:
    """
    Evaluates Sufficiency, Comprehensiveness, and Monotonicity of feature attributions.
    """
    def __init__(self, baseline_val: float = 0.0) -> None:
        self.baseline_val = baseline_val

    def evaluate_faithfulness(
        self,
        x: np.ndarray,
        attributions: np.ndarray,
        predict_func: Callable[[np.ndarray], float],
        top_k: int = 5,
    ) -> FaithfulnessMetrics:
        """
        Computes sufficiency, comprehensiveness, and step-by-step monotonicity.
        """
        r_full = float(predict_func(x))
        ranked_indices = np.argsort(-np.abs(attributions))
        topk_indices = set(ranked_indices[:top_k])

        # 1. Sufficiency: Keep ONLY top-k features, replace others with baseline
        x_suff = np.full_like(x, self.baseline_val)
        for idx in topk_indices:
            x_suff[idx] = x[idx]
        r_suff = float(predict_func(x_suff))
        sufficiency = r_suff / max(1e-4, r_full)

        # 2. Comprehensiveness: Remove top-k features (replace with baseline), keep others
        x_comp = x.copy()
        for idx in topk_indices:
            x_comp[idx] = self.baseline_val
        r_comp = float(predict_func(x_comp))
        comprehensiveness = max(0.0, r_full - r_comp)

        # 3. Monotonicity: Sequentially remove features 1 by 1 and verify decreasing risk
        x_seq = x.copy()
        prev_risk = r_full
        monotonic_steps = 0
        total_steps = min(top_k, len(ranked_indices))

        for step in range(total_steps):
            idx = ranked_indices[step]
            x_seq[idx] = self.baseline_val
            curr_risk = float(predict_func(x_seq))
            if curr_risk <= prev_risk + 1e-4:  # Small numerical tolerance
                monotonic_steps += 1
            prev_risk = curr_risk

        monotonicity = float(monotonic_steps / max(1, total_steps))
        is_faithful = sufficiency >= 0.70 and comprehensiveness >= 0.20 and monotonicity >= 0.80

        return FaithfulnessMetrics(
            top_k=top_k,
            sufficiency_score=sufficiency,
            comprehensiveness_score=comprehensiveness,
            monotonicity_score=monotonicity,
            is_faithful=is_faithful,
        )
