"""
AHRAS Streaming Drift Detectors
-------------------------------
Implements online sequential change-point and concept drift detectors:
1. Page-Hinkley Test (Cumulative deviation monitoring)
2. ADWIN (Adaptive Windowing statistical mean divergence test)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass
class StreamingDriftSignal:
    """
    Alert signal emitted when distribution drift is detected in real time.
    """
    detected: bool
    detector_name: str
    step_index: int
    test_statistic: float
    threshold: float
    description: str


class PageHinkleyDriftDetector:
    """
    Page-Hinkley cumulative deviation drift detector for streaming error / loss series.
    """
    def __init__(self, delta: float = 0.05, threshold: float = 25.0, alpha: float = 0.99) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.step = 0
        self.mean = 0.0
        self.sum_val = 0.0
        self.min_val = 0.0

    def reset(self) -> None:
        self.step = 0
        self.mean = 0.0
        self.sum_val = 0.0
        self.min_val = 0.0

    def update(self, x: float) -> StreamingDriftSignal:
        """
        Updates cumulative sum and tests whether PH_stat exceeds threshold.
        """
        self.step += 1
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        self.sum_val += (x - self.mean - self.delta)
        if self.sum_val < self.min_val:
            self.min_val = self.sum_val

        ph_stat = self.sum_val - self.min_val
        is_drift = ph_stat > self.threshold

        signal = StreamingDriftSignal(
            detected=is_drift,
            detector_name="PageHinkley",
            step_index=self.step,
            test_statistic=round(ph_stat, 4),
            threshold=self.threshold,
            description=f"Page-Hinkley statistic {ph_stat:.2f} > {self.threshold:.2f}" if is_drift else "Stable",
        )

        if is_drift:
            # Auto-reset accumulator after drift alert to enable subsequent alerts
            self.sum_val = 0.0
            self.min_val = 0.0

        return signal


class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) algorithm: adjusts window length dynamically
    and signals drift when subwindow means diverge statistically.
    """
    def __init__(self, delta_p: float = 0.002, max_window: int = 500) -> None:
        self.delta_p = delta_p
        self.max_window = max_window
        self.window: deque[float] = deque(maxlen=max_window)
        self.step = 0

    def reset(self) -> None:
        self.window.clear()
        self.step = 0

    def update(self, x: float) -> StreamingDriftSignal:
        self.step += 1
        self.window.append(x)

        n = len(self.window)
        if n < 30:
            return StreamingDriftSignal(
                detected=False,
                detector_name="ADWIN",
                step_index=self.step,
                test_statistic=0.0,
                threshold=0.0,
                description="Warming up",
            )

        # Split window into two halves W0 and W1
        mid = n // 2
        w0 = list(self.window)[:mid]
        w1 = list(self.window)[mid:]

        mu0 = sum(w0) / len(w0)
        mu1 = sum(w1) / len(w1)
        diff = abs(mu0 - mu1)

        # Hoeffding bound threshold
        m = 1.0 / (1.0 / len(w0) + 1.0 / len(w1))
        eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta_p))

        is_drift = diff > eps_cut
        if is_drift:
            # Shrink window by dropping older half
            for _ in range(mid):
                self.window.popleft()

        return StreamingDriftSignal(
            detected=is_drift,
            detector_name="ADWIN",
            step_index=self.step,
            test_statistic=round(diff, 4),
            threshold=round(eps_cut, 4),
            description=f"ADWIN mean difference {diff:.3f} > {eps_cut:.3f}" if is_drift else "Stable",
        )
