from __future__ import annotations
"""
AHRAS Evaluation Metrics Engine
---------------------------------
Computes comprehensive intrusion detection, probability calibration, XAI,
and computational performance metrics:
  1. Accuracy, Precision, Recall, F1-Score
  2. ROC Curve & Area Under Curve (ROC-AUC)
  3. False Positive Rate (FPR) & Detection Rate (DR)
  4. Brier Score & Expected Calibration Error (ECE)
  5. Confusion Matrix (TP, FP, TN, FN)
  6. Latency statistics (mean, p95, p99, throughput eps)
  7. Peak Memory (MB) and CPU (%)
  8. Per-class attack category breakdown
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    # Classification metrics
    accuracy:            float = 0.0
    precision:           float = 0.0
    recall:              float = 0.0
    f1:                  float = 0.0
    auc:                 Optional[float] = None
    false_positive_rate: float = 0.0
    detection_rate:      float = 0.0
    
    # Calibration metrics
    brier_score:         Optional[float] = None
    ece:                 Optional[float] = None

    # Confusion matrix
    true_positives:      int = 0
    false_positives:     int = 0
    true_negatives:      int = 0
    false_negatives:     int = 0
    total_samples:       int = 0
    num_benign:          int = 0
    num_attack:          int = 0

    # Computational Performance
    mean_latency_ms:     float = 0.0
    p95_latency_ms:      float = 0.0
    p99_latency_ms:      float = 0.0
    throughput_eps:      float = 0.0
    peak_memory_mb:      float = 0.0
    mean_cpu_pct:        float = 0.0

    # ROC / Breakdown
    roc_fpr:             List[float] = field(default_factory=list)
    roc_tpr:             Optional[List[float]] = field(default_factory=list)
    per_class:           Dict[str, dict] = field(default_factory=dict)
    dataset_name:        str = ""
    threshold_used:      float = 0.50

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "samples": {
                "total": self.total_samples,
                "benign": self.num_benign,
                "attack": self.num_attack,
            },
            "classification": {
                "accuracy":            round(self.accuracy, 4),
                "precision":           round(self.precision, 4),
                "recall":              round(self.recall, 4),
                "f1":                  round(self.f1, 4),
                "auc":                 round(self.auc, 4) if self.auc is not None else None,
                "false_positive_rate": round(self.false_positive_rate, 4),
                "detection_rate":      round(self.detection_rate, 4),
                "brier_score":         round(self.brier_score, 4) if self.brier_score is not None else None,
                "ece":                 round(self.ece, 4) if self.ece is not None else None,
            },
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "TN": self.true_negatives,
                "FN": self.false_negatives,
            },
            "performance": {
                "mean_latency_ms": round(self.mean_latency_ms, 3),
                "p95_latency_ms":  round(self.p95_latency_ms, 3),
                "p99_latency_ms":  round(self.p99_latency_ms, 3),
                "throughput_eps":  round(self.throughput_eps, 1),
                "peak_memory_mb":  round(self.peak_memory_mb, 1),
                "mean_cpu_pct":    round(self.mean_cpu_pct, 1),
            },
            "per_class": self.per_class,
            "threshold_used": self.threshold_used,
        }

    def summary_line(self) -> str:
        auc_str = f"{self.auc:.3f}" if self.auc is not None else "n/a"
        brier_str = f"{self.brier_score:.4f}" if self.brier_score is not None else "n/a"
        return (
            f"[{self.dataset_name}] "
            f"Acc={self.accuracy:.3f} P={self.precision:.3f} "
            f"R={self.recall:.3f} F1={self.f1:.3f} AUC={auc_str} "
            f"FPR={self.false_positive_rate:.3f} Brier={brier_str} "
            f"Lat={self.mean_latency_ms:.1f}ms Thru={self.throughput_eps:.0f}eps"
        )


class MetricsCalculator:
    """
    Computes all standard detection and operational metrics.
    """

    def compute(
        self,
        y_true: List[int],
        y_score: List[float],
        latencies_ms: Optional[List[float]] = None,
        peak_memory_mb: float = 0.0,
        mean_cpu_pct: float = 0.0,
        dataset_name: str = "evaluation_run",
        threshold: float = 0.50,
        attack_categories: Optional[List[str]] = None,
    ) -> MetricsReport:
        if not y_true:
            return MetricsReport(dataset_name=dataset_name)

        latencies_ms = latencies_ms or [0.0] * len(y_true)
        n = len(y_true)

        # Scale detection: if scores are 0-100 scale, normalize to [0, 1] for thresholding
        max_s = max(y_score) if y_score else 0.0
        normalized_scores = [s / 100.0 if max_s > 1.0 else float(s) for s in y_score]
        norm_threshold = threshold / 100.0 if threshold > 1.0 else float(threshold)

        y_pred = [1 if s >= norm_threshold else 0 for s in normalized_scores]

        # Confusion Matrix
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        accuracy  = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) else 0.0

        # ROC / AUC
        roc_fpr, roc_tpr, auc_val = self._roc_auc(y_true, normalized_scores)

        # Calibration: Brier Score & ECE
        brier = float(np.mean([(s - yt) ** 2 for s, yt in zip(normalized_scores, y_true)]))
        ece = self._compute_ece(y_true, normalized_scores)

        # Latency
        mean_lat, p95_lat, p99_lat, throughput = self._latency_stats(latencies_ms)

        # Per-class breakdown
        per_class = {}
        if attack_categories:
            cats: Dict[str, Dict[str, int]] = {}
            for yt, yp, cat in zip(y_true, y_pred, attack_categories):
                c = (cat or "Normal").strip()
                if c not in cats:
                    cats[c] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
                key = ("tp" if yt == 1 and yp == 1 else
                       "fp" if yt == 0 and yp == 1 else
                       "tn" if yt == 0 and yp == 0 else "fn")
                cats[c][key] += 1
                cats[c]["total"] += 1

            for c, counts in cats.items():
                c_tp, c_fp, c_tn, c_fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
                c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
                c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
                c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) else 0.0
                per_class[c] = {
                    "count": counts["total"],
                    "precision": round(c_p, 4),
                    "recall": round(c_r, 4),
                    "f1": round(c_f1, 4),
                    "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
                }

        return MetricsReport(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc_val,
            false_positive_rate=fpr,
            detection_rate=recall,
            brier_score=brier,
            ece=ece,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_samples=n,
            num_benign=sum(1 for y in y_true if y == 0),
            num_attack=sum(1 for y in y_true if y == 1),
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            throughput_eps=throughput,
            peak_memory_mb=peak_memory_mb,
            mean_cpu_pct=mean_cpu_pct,
            roc_fpr=roc_fpr,
            roc_tpr=roc_tpr,
            per_class=per_class,
            dataset_name=dataset_name,
            threshold_used=norm_threshold,
        )

    def _compute_ece(self, y_true: List[int], y_score: List[float], n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        if n == 0:
            return 0.0

        scores = np.array(y_score)
        labels = np.array(y_true)

        for i in range(n_bins):
            bin_lower, bin_upper = bins[i], bins[i + 1]
            in_bin = (scores >= bin_lower) & (scores < bin_upper if i < n_bins - 1 else scores <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(scores[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def _roc_auc(self, y_true: List[int], y_score: List[float]) -> Tuple[List[float], Optional[List[float]], Optional[float]]:
        if len(set(y_true)) < 2:
            return [], None, None
        try:
            from sklearn.metrics import roc_curve, auc
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
            auc_val = float(auc(fpr_arr, tpr_arr))
            step = max(1, len(fpr_arr) // 50)
            return (
                [round(float(x), 4) for x in fpr_arr[::step]],
                [round(float(x), 4) for x in tpr_arr[::step]],
                round(auc_val, 4),
            )
        except Exception as e:
            log.warning(f"[METRICS] ROC/AUC computation exception: {e}")
            return [], None, None

    def _latency_stats(self, latencies_ms: List[float]) -> Tuple[float, float, float, float]:
        if not latencies_ms:
            return 0.0, 0.0, 0.0, 0.0
        s = sorted(latencies_ms)
        n = len(s)
        mean_l = sum(s) / n
        p95_l = s[int(n * 0.95)] if n > 1 else s[0]
        p99_l = s[min(int(n * 0.99), n - 1)] if n > 1 else s[0]
        total_s = sum(latencies_ms) / 1000.0
        throughput = n / total_s if total_s > 0 else 0.0
        return round(mean_l, 3), round(p95_l, 3), round(p99_l, 3), round(throughput, 1)
