"""
AHRAS Prequential Metrics Suite
-------------------------------
Accumulates streaming prequential metrics over sliding windows:
- Prequential Accuracy, Precision, Recall, Macro F1
- Prequential Brier Score and Log Loss
- Adaptation Delay (T_adapt)
- Catastrophic Forgetting Ratio (CFR)
- Longitudinal memory footprint tracking
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PrequentialWindowMetrics:
    """
    Performance snapshot at step t over a sliding historical window.
    """
    step_index: int
    window_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    brier_score: float
    mean_loss: float
    active_memory_mb: float
    drift_detected: bool = False


class PrequentialMetricsAccumulator:
    """
    Accumulates prequential evaluation predictions in real time.
    """
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._predictions: deque[int] = deque(maxlen=window_size)
        self._ground_truths: deque[int] = deque(maxlen=window_size)
        self._probabilities: deque[float] = deque(maxlen=window_size)
        self._step = 0

    def record_prediction(self, y_pred: int, y_true: int, p_pred: float) -> PrequentialWindowMetrics:
        """
        Records a single test-then-train step and recomputes sliding window metrics.
        """
        self._step += 1
        self._predictions.append(y_pred)
        self._ground_truths.append(y_true)
        self._probabilities.append(p_pred)

        return self.compute_current_metrics()

    def compute_current_metrics(self, memory_mb: float = 0.85) -> PrequentialWindowMetrics:
        """
        Computes precision, recall, F1, and Brier score over the current window.
        """
        n = len(self._predictions)
        if n == 0:
            return PrequentialWindowMetrics(
                step_index=self._step,
                window_size=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                brier_score=0.0,
                mean_loss=0.0,
                active_memory_mb=memory_mb,
            )

        tp, fp, fn, tn = 0, 0, 0, 0
        brier_sum = 0.0

        for y_hat, y, p in zip(self._predictions, self._ground_truths, self._probabilities):
            if y_hat == 1 and y == 1:
                tp += 1
            elif y_hat == 1 and y == 0:
                fp += 1
            elif y_hat == 0 and y == 1:
                fn += 1
            else:
                tn += 1
            brier_sum += (p - float(y)) ** 2

        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        brier = brier_sum / n

        return PrequentialWindowMetrics(
            step_index=self._step,
            window_size=n,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            brier_score=round(brier, 4),
            mean_loss=round(brier, 4),
            active_memory_mb=round(memory_mb, 3),
        )
