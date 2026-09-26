from __future__ import annotations
"""
AHRAS Module / Instability — Temporal Epistemic Instability Tracker
-------------------------------------------------------------------
Calculates 5 distinct temporal volatility signals over sliding prediction windows:
  1. Confidence Variance: Var(c_t)
  2. Class-Flip Count & Rate: Frequency of threshold crossings (0 <-> 1)
  3. Trajectory Instability: Second-order acceleration |Δp_{t+1} - Δp_t|
  4. Prediction Entropy: Mean Shannon binary entropy H(p_t)
  5. Temporal Disagreement: Short-term vs. long-term moving average divergence

IMPORTANT OPERATIONAL GUARANTEE:
  Instability modulates epistemic uncertainty, monitoring levels, human-review priority,
  and autonomous gating. It NEVER arbitrarily inflates baseline risk R_t.
"""

from collections import defaultdict, deque
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from instability.models import (
    PredictionRecord, InstabilityMetrics, EntityStabilityProfile
)


class TemporalInstabilityTracker:
    """
    Tracks streaming predictions per entity and computes calibrated
    temporal epistemic instability metrics for selective autonomy gating.
    """

    def __init__(
        self,
        max_window_size: int = 50,
        temporal_window_sec: float = 86400.0,
        abstain_threshold: float = 0.40,
        escalate_threshold: float = 0.65,
    ):
        self.max_window_size = max_window_size
        self.temporal_window_sec = temporal_window_sec
        self.abstain_threshold = abstain_threshold
        self.escalate_threshold = escalate_threshold
        self._history: Dict[str, deque[PredictionRecord]] = defaultdict(lambda: deque(maxlen=self.max_window_size))

    def record_prediction(
        self,
        entity_key: str,
        timestamp: float,
        risk_score: float,
        confidence: float,
        threshold: float = 0.50,
    ) -> None:
        """Appends a new prediction and confidence observation for entity_key."""
        p_clamped = min(1.0, max(0.0, float(risk_score)))
        c_clamped = min(1.0, max(0.0, float(confidence)))
        label = 1 if p_clamped >= threshold else 0

        record = PredictionRecord(
            timestamp=float(timestamp),
            risk_score=p_clamped,
            confidence=c_clamped,
            class_label=label,
        )

        q = self._history[entity_key]
        q.append(record)

        # Prune stale records outside temporal window
        cutoff = timestamp - self.temporal_window_sec
        while q and q[0].timestamp < cutoff:
            q.popleft()

    def get_history_length(self, entity_key: str) -> int:
        return len(self._history.get(entity_key, []))

    def compute_instability(
        self,
        entity_key: str,
        threshold: float = 0.50,
    ) -> InstabilityMetrics:
        """
        Computes the complete 5-component epistemic instability metrics for an entity.
        Returns normalized instability score in [0.0, 1.0].
        """
        records = list(self._history.get(entity_key, []))
        n = len(records)

        # Base case: insufficient history
        if n < 2:
            return InstabilityMetrics(
                confidence_variance=0.0,
                class_flip_count=0,
                class_flip_rate=0.0,
                trajectory_instability=0.0,
                prediction_entropy=0.0,
                temporal_disagreement=0.0,
                instability_score=0.0,
                recommended_action="AUTONOMOUS",
                human_review_priority=0.1,
                autonomy_gated=False,
            )

        risks = np.array([r.risk_score for r in records], dtype=np.float64)
        confs = np.array([r.confidence for r in records], dtype=np.float64)

        # ── 1. Confidence Variance ──
        # Normalized by max possible variance of [0, 1] bounded variable (0.25)
        conf_var = float(np.var(confs))
        norm_conf_var = min(1.0, 4.0 * conf_var)

        # ── 2. Class Flip Count & Rate ──
        binary_labels = np.array([1 if r >= threshold else 0 for r in risks], dtype=np.int32)
        flips = int(np.sum(binary_labels[:-1] != binary_labels[1:]))
        flip_rate = float(flips / (n - 1)) if n > 1 else 0.0

        # ── 3. Trajectory Instability ──
        # Evaluates second-order rate of change: |(p_{t+1} - p_t) - (p_t - p_{t-1})|
        delta_p = np.diff(risks)
        if len(delta_p) >= 2:
            second_diff = np.abs(np.diff(delta_p))
            mean_second_diff = float(np.mean(second_diff))
            traj_instability = min(1.0, 2.0 * mean_second_diff)
        else:
            traj_instability = min(1.0, float(np.abs(delta_p[0]))) if len(delta_p) > 0 else 0.0

        # ── 4. Prediction Entropy ──
        # Mean binary Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)
        eps = 1e-7
        p_safe = np.clip(risks, eps, 1.0 - eps)
        entropies = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
        mean_entropy = float(np.mean(entropies))

        # ── 5. Temporal Disagreement ──
        # Divergence between short-term EWMA (last 5 or 25%) and long-term mean
        k_short = max(2, min(5, n // 2))
        short_mean = float(np.mean(risks[-k_short:]))
        long_mean = float(np.mean(risks))
        ewma_divergence = abs(short_mean - long_mean)
        consecutive_variance = float(np.mean(np.abs(delta_p))) if len(delta_p) > 0 else 0.0
        temporal_disagree = min(1.0, 0.5 * ewma_divergence + 0.5 * consecutive_variance)

        # ── 6. Calibrated Composite Instability Score ──
        # Multi-factor uncertainty formulation:
        instability = (
            0.30 * flip_rate +
            0.25 * mean_entropy +
            0.20 * traj_instability +
            0.15 * norm_conf_var +
            0.10 * temporal_disagree
        )
        instability_score = min(1.0, max(0.0, float(instability)))

        # ── 7. Decision Recommendations & Autonomy Gating ──
        if instability_score >= self.escalate_threshold:
            action = "ESCALATE_ANALYST"
            gated = True
            priority = instability_score
        elif instability_score >= self.abstain_threshold:
            action = "ABSTAIN"
            gated = True
            priority = 0.50 + 0.50 * instability_score
        elif instability_score >= 0.20:
            action = "MONITOR"
            gated = False
            priority = 0.30
        else:
            action = "AUTONOMOUS"
            gated = False
            priority = 0.10

        return InstabilityMetrics(
            confidence_variance=round(conf_var, 6),
            class_flip_count=flips,
            class_flip_rate=round(flip_rate, 4),
            trajectory_instability=round(traj_instability, 4),
            prediction_entropy=round(mean_entropy, 4),
            temporal_disagreement=round(temporal_disagree, 4),
            instability_score=round(instability_score, 4),
            recommended_action=action,
            human_review_priority=round(priority, 4),
            autonomy_gated=gated,
        )

    def get_entity_profile(self, entity_key: str, threshold: float = 0.50) -> EntityStabilityProfile:
        """Returns the full EntityStabilityProfile including window span and metrics."""
        records = list(self._history.get(entity_key, []))
        metrics = self.compute_instability(entity_key, threshold=threshold)
        span_sec = (records[-1].timestamp - records[0].timestamp) if len(records) >= 2 else 0.0
        return EntityStabilityProfile(
            entity_key=entity_key,
            history_length=len(records),
            window_seconds=span_sec,
            metrics=metrics,
        )

    # ── Operational Integration Helpers ───────────────────────────────────────

    def modulate_uncertainty(self, base_uncertainty: float, instability_score: float) -> float:
        """
        Inflates epistemic uncertainty proportionally to temporal instability:
          U_mod = U_base + (1.0 - U_base) * 0.65 * Instability
        Leaves baseline risk score R_t strictly untouched.
        """
        u_clamped = min(1.0, max(0.0, float(base_uncertainty)))
        inst_clamped = min(1.0, max(0.0, float(instability_score)))
        delta = (1.0 - u_clamped) * 0.65 * inst_clamped
        return round(min(1.0, max(0.0, u_clamped + delta)), 4)

    def modulate_selective_autonomy(
        self,
        candidate_action: str,
        instability_score: float,
    ) -> Tuple[str, bool, str]:
        """
        Inspects candidate action and enforces safety overrides when instability is elevated.
        Returns: (final_action, autonomy_gated, explanation).
        """
        inst = float(instability_score)

        # 1. Extreme instability: route to human analyst
        if inst >= self.escalate_threshold:
            gated = candidate_action in ("AUTONOMOUS_CONTAINMENT", "STAGED_CONTAINMENT")
            return (
                "ESCALATE_ANALYST",
                gated,
                f"Inhibited autonomous containment: Overrode {candidate_action} -> ESCALATE_ANALYST (Extreme temporal instability {inst:.2f} >= {self.escalate_threshold}).",
            )

        # 2. High instability: route to selective abstention
        if inst >= self.abstain_threshold:
            if candidate_action in ("AUTONOMOUS_CONTAINMENT", "STAGED_CONTAINMENT"):
                return (
                    "ABSTAIN",
                    True,
                    f"Inhibited autonomous containment: Overrode {candidate_action} -> ABSTAIN (High temporal oscillation {inst:.2f} >= {self.abstain_threshold}).",
                )
            elif candidate_action == "MONITOR":
                return (
                    "ABSTAIN",
                    False,
                    f"Elevated MONITOR -> ABSTAIN: High temporal oscillation ({inst:.2f} >= {self.abstain_threshold}) warrants selective abstention.",
                )

        # 3. Premature auto-pass gating
        if candidate_action == "AUTONOMOUS_PASS" and inst >= 0.35:
            return (
                "MONITOR",
                False,
                f"Elevated {candidate_action} -> MONITOR: Moderate temporal volatility ({inst:.2f} >= 0.35) warrants proactive monitoring.",
            )

        return candidate_action, False, "Candidate action verified by temporal stability filter."
