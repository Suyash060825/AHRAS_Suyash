from __future__ import annotations
"""
AHRAS XAI Explainer
--------------------
Generates human-readable explanations for every detection.

Approach: Local feature importance via perturbation analysis.
  For each feature, measure how much the anomaly score changes
  when that feature is replaced with its baseline (mean) value.
  Larger change = higher contribution = more important feature.

  This is equivalent to a simplified SHAP (SHapley Additive exPlanations)
  for models where we don't have direct access to SHAP TreeExplainer.

  For Isolation Forest, we additionally extract the average tree
  path depth per feature as a proxy for importance.

Output per detection:
  {
    "top_features": [
      {"feature": "log_packet_count", "value": 10.82,
       "baseline": 2.31, "deviation_pct": 368, "contribution": 0.45},
      ...
    ],
    "explanation_text": "Flagged primarily due to packet_count (+368% above
                          baseline) and unique_dst_ports (+412% above baseline)",
    "confidence_breakdown": {
      "signature": 0.90,
      "anomaly":   0.72,
      "statistical": 0.61,
    }
  }
"""

import math
import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# How many top features to include in explanations
_TOP_N = 5


# ─────────────────────────────────────────────────────────────────────────────
# Core: perturbation-based feature importance
# ─────────────────────────────────────────────────────────────────────────────

def _perturbation_importance(
        vec: np.ndarray,
        baseline: np.ndarray,
        score_fn,
        n_samples: int = 20,
) -> np.ndarray:
    """
    Estimate feature importance by measuring score change when each
    feature is replaced with its baseline value.

    Returns importance array of same length as vec.
    Higher value = more important to the anomaly detection.
    """
    original_score = score_fn(vec.reshape(1, -1))
    importances    = np.zeros(len(vec))

    for i in range(len(vec)):
        vec_masked     = vec.copy()
        vec_masked[i]  = baseline[i]
        masked_score   = score_fn(vec_masked.reshape(1, -1))
        importances[i] = abs(original_score - masked_score)

    # Normalize to [0,1]
    total = importances.sum()
    if total > 1e-9:
        importances = importances / total

    return importances


def _if_feature_importance(
        if_pipe, vec: np.ndarray
) -> np.ndarray:
    """
    Isolation Forest feature importance via mean tree path depth.
    Features that appear near the root (short path) contribute more
    to the isolation = more anomalous contribution.
    """
    try:
        scaler = if_pipe.named_steps["scaler"]
        clf    = if_pipe.named_steps["clf"]
        X_sc   = scaler.transform(vec.reshape(1, -1))

        # Collect depth of each feature split across all trees
        n_feat = vec.shape[0]
        depths = np.zeros(n_feat)
        counts = np.zeros(n_feat)

        for tree in clf.estimators_:
            node   = 0
            depth  = 0
            while tree.tree_.feature[node] != -2:   # -2 = leaf
                feat_idx = tree.tree_.feature[node]
                if 0 <= feat_idx < n_feat:
                    depths[feat_idx] += depth
                    counts[feat_idx] += 1
                # Follow the path for this sample
                thresh = tree.tree_.threshold[node]
                if X_sc[0, feat_idx if feat_idx < n_feat else 0] <= thresh:
                    node = tree.tree_.children_left[node]
                else:
                    node = tree.tree_.children_right[node]
                depth += 1

        # Invert: shorter path = more important
        with np.errstate(divide="ignore", invalid="ignore"):
            importance = np.where(counts > 0, 1.0 / (depths / counts + 1), 0)
        total = importance.sum()
        if total > 1e-9:
            importance /= total
        return importance

    except Exception as e:
        log.debug(f"[XAI] IF importance failed: {e}")
        return np.ones(len(vec)) / len(vec)


# ─────────────────────────────────────────────────────────────────────────────
# Feature deviation analysis
# ─────────────────────────────────────────────────────────────────────────────

def _feature_deviations(
        vec: np.ndarray,
        baseline_mean: np.ndarray,
        feature_names: list[str],
) -> list[dict]:
    """
    Compute percentage deviation of each feature from its baseline mean.
    Returns sorted list (highest deviation first).
    """
    deviations = []
    for i, name in enumerate(feature_names):
        val   = float(vec[i])
        base  = float(baseline_mean[i]) if baseline_mean is not None else 0.0
        if abs(base) > 1e-9:
            dev_pct = ((val - base) / abs(base)) * 100
        else:
            dev_pct = val * 100   # if baseline is 0, any value is infinite deviation
        deviations.append({
            "feature":       name,
            "value":         round(val, 4),
            "baseline":      round(base, 4),
            "deviation_pct": round(dev_pct, 1),
            "abs_deviation": abs(dev_pct),
        })
    return sorted(deviations, key=lambda x: x["abs_deviation"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Natural language generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_explanation_text(
        top_features: list[dict],
        attack_type:  str,
        sig_conf:     float,
        ml_conf:      float,
        stat_conf:    float,
) -> str:
    """Generate a one-sentence human-readable explanation."""
    if not top_features:
        return "Anomaly detected with no dominant feature contribution."

    parts = []
    for f in top_features[:3]:
        name  = f["feature"].replace("_", " ")
        dev   = f["deviation_pct"]
        if abs(dev) > 10:
            direction = "above" if dev > 0 else "below"
            parts.append(f"{name} ({dev:+.0f}% {direction} baseline)")
        else:
            parts.append(f"{name} (value={f['value']:.3f})")

    engines = []
    if sig_conf  > 0.5: engines.append(f"signature ({sig_conf:.0%})")
    if ml_conf   > 0.3: engines.append(f"ML ({ml_conf:.0%})")
    if stat_conf > 0.3: engines.append(f"statistical ({stat_conf:.0%})")

    attack_label = attack_type.replace("_", " ").title() if attack_type else "anomaly"
    feature_str  = "; ".join(parts)
    engine_str   = " + ".join(engines) if engines else "ensemble"

    return (
        f"Detected {attack_label} via {engine_str}. "
        f"Primary drivers: {feature_str}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def explain(
        vec:           np.ndarray,
        feature_names: list[str],
        ocsf_class:    str,
        attack_type:   str,
        sig_confidence:  float = 0.0,
        ml_confidence:   float = 0.0,
        stat_confidence: float = 0.0,
        if_pipe=None,          # optional: Isolation Forest pipeline for IF-importance
        baseline_mean: Optional[np.ndarray] = None,
) -> dict:
    """
    Generate a full XAI explanation for a detection.

    Returns:
      {
        "top_features": [...],           # sorted by contribution
        "explanation_text": "...",       # one-sentence summary
        "confidence_breakdown": {...},   # per-engine confidence
        "feature_importances": [...],    # raw importance scores
      }
    """
    n = len(feature_names)

    # ── Feature importance ────────────────────────────────────────────────────
    if if_pipe is not None:
        importances = _if_feature_importance(if_pipe, vec)
    else:
        # Fallback: use absolute feature value as rough importance proxy
        total = float(vec.sum()) or 1.0
        importances = np.abs(vec) / total

    importances = importances[:n]   # align lengths

    # ── Baseline for deviation calculation ────────────────────────────────────
    if baseline_mean is None:
        # Use zero vector as baseline if no history
        baseline_mean = np.zeros(n)

    # ── Feature deviation table ───────────────────────────────────────────────
    deviations = _feature_deviations(vec[:n], baseline_mean[:n], feature_names)

    # ── Merge: score = 0.6 × deviation rank + 0.4 × importance ───────────────
    imp_map = {feature_names[i]: float(importances[i]) for i in range(min(n, len(importances)))}
    for d in deviations:
        imp = imp_map.get(d["feature"], 0.0)
        d["contribution"] = round(
            0.6 * (d["abs_deviation"] / max(deviations[0]["abs_deviation"], 1))
            + 0.4 * imp,
            4
        )
    deviations.sort(key=lambda x: x["contribution"], reverse=True)

    top_features = []
    for d in deviations[:_TOP_N]:
        top_features.append({
            "feature":       d["feature"],
            "value":         d["value"],
            "baseline":      d["baseline"],
            "deviation_pct": d["deviation_pct"],
            "contribution":  d["contribution"],
        })

    explanation_text = _build_explanation_text(
        top_features, attack_type,
        sig_confidence, ml_confidence, stat_confidence,
    )

    return {
        "top_features": top_features,
        "explanation_text": explanation_text,
        "confidence_breakdown": {
            "signature":   round(sig_confidence,  4),
            "anomaly":     round(ml_confidence,   4),
            "statistical": round(stat_confidence, 4),
        },
        "feature_importances": [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
            for i in range(min(n, len(importances)))
        ],
    }
