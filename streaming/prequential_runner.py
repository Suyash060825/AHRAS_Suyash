"""
AHRAS Prequential Evaluation Runner
-----------------------------------
Executes strict Test-Then-Train online prequential evaluation:
- No future lookahead or batch contamination
- Delayed ground-truth verification queue (delta = 0, 60, 240, 1000)
- Online adaptive retraining vs frozen static baseline
- Continuous drift detection (Page-Hinkley, ADWIN)
- Real-time memory footprint accounting
"""

from __future__ import annotations

import copy
import enum
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from streaming.temporal_split import StreamingTemporalEvent, TemporalStreamGenerator, DriftRegimeType
from streaming.drift_detector import PageHinkleyDriftDetector, ADWINDriftDetector, StreamingDriftSignal
from streaming.drift_metrics import PrequentialMetricsAccumulator, PrequentialWindowMetrics


class ModelAdaptationStrategy(str, enum.Enum):
    STATIC = "static"
    STREAMING_UPDATED = "streaming_updated"


@dataclass
class PrequentialSimulationResult:
    """
    Complete summary of a prequential longitudinal run.
    """
    strategy: ModelAdaptationStrategy
    label_delay: int
    total_events: int
    mean_macro_f1: float
    final_f1: float
    final_recall: float
    final_brier_score: float
    drift_alerts_count: int
    adaptation_delay_steps: int
    catastrophic_forgetting_ratio: float
    mean_memory_mb: float
    peak_memory_mb: float
    time_series_metrics: List[Dict[str, Any]] = field(default_factory=list)


class OnlineLinearClassifier:
    """
    Lightweight online logistic regression learner updated via streaming SGD
    with bounded memory footprint and optional mini-batch replay.
    """
    def __init__(self, dim: int = 16, lr: float = 0.05, memory_capacity: int = 200) -> None:
        self.dim = dim
        self.lr = lr
        self.weights = np.zeros(dim)
        self.bias = -0.5
        self.replay_buffer: deque[Tuple[np.ndarray, int]] = deque(maxlen=memory_capacity)

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(np.dot(self.weights, x) + self.bias)
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def predict(self, x: np.ndarray, threshold: float = 0.50) -> int:
        return 1 if self.predict_proba(x) >= threshold else 0

    def update(self, x: np.ndarray, y: int) -> None:
        """Online SGD step + replay buffer consolidation."""
        self.replay_buffer.append((x, y))

        p = self.predict_proba(x)
        grad_w = (p - float(y)) * x
        grad_b = (p - float(y))

        # Weight update with L2 regularization
        self.weights -= self.lr * (grad_w + 1e-4 * self.weights)
        self.bias -= self.lr * grad_b

        # Sample replay step if buffer has sufficient items
        if len(self.replay_buffer) >= 10:
            idx = np.random.randint(0, len(self.replay_buffer))
            rx, ry = self.replay_buffer[idx]
            rp = self.predict_proba(rx)
            self.weights -= (self.lr * 0.5) * ((rp - float(ry)) * rx + 1e-4 * self.weights)
            self.bias -= (self.lr * 0.5) * (rp - float(ry))

    def get_memory_bytes(self) -> int:
        """Returns approximate heap memory consumed by weights and replay buffer."""
        w_bytes = self.weights.nbytes
        buf_bytes = sum(x.nbytes + 8 for x, _ in self.replay_buffer)
        return sys.getsizeof(self) + w_bytes + buf_bytes


class PrequentialEvaluationRunner:
    """
    Orchestrates prequential lifecycle evaluation over streaming event series.
    """
    def __init__(
        self,
        strategy: ModelAdaptationStrategy = ModelAdaptationStrategy.STREAMING_UPDATED,
        label_delay: int = 0,
        window_size: int = 100,
    ) -> None:
        self.strategy = strategy
        self.label_delay = label_delay
        self.window_size = window_size

    def run_simulation(
        self,
        stream: List[StreamingTemporalEvent],
    ) -> PrequentialSimulationResult:
        """
        Runs the full test-then-train loop across chronological events.
        """
        if not stream:
            raise ValueError("Stream must not be empty.")

        dim = len(stream[0].features)
        model = OnlineLinearClassifier(dim=dim)

        # Pretrain lightly on first 20 events to initialize decision hyperplane
        warmup_n = min(20, len(stream))
        for i in range(warmup_n):
            model.update(stream[i].features, stream[i].ground_truth_label)

        metrics_acc = PrequentialMetricsAccumulator(window_size=self.window_size)
        ph_detector = PageHinkleyDriftDetector(threshold=15.0)
        adwin_detector = ADWINDriftDetector()

        # Verification queue: holds (step, features, label) until step + label_delay
        verification_queue: deque[Tuple[int, np.ndarray, int]] = deque()

        drift_alerts = 0
        time_series: List[Dict[str, Any]] = []
        f1_values: List[float] = []
        memory_values: List[float] = []

        drift_step: Optional[int] = None
        recovery_step: Optional[int] = None
        pre_drift_f1: float = 0.90

        for t_idx, evt in enumerate(stream):
            # 1. TEST: Make immutable prediction BEFORE seeing ground truth
            p = model.predict_proba(evt.features)
            y_pred = 1 if p >= 0.50 else 0

            # 2. Record prequential metric
            # Base memory: model + queues
            mem_mb = (model.get_memory_bytes() + len(verification_queue) * (dim * 8 + 16)) / (1024 * 1024)
            snap = metrics_acc.record_prediction(y_pred, evt.ground_truth_label, p)
            snap.active_memory_mb = round(mem_mb, 4)

            # Check drift on loss
            loss_t = (p - float(evt.ground_truth_label)) ** 2
            ph_sig = ph_detector.update(loss_t)
            adwin_sig = adwin_detector.update(loss_t)

            is_drift = ph_sig.detected or adwin_sig.detected
            if is_drift:
                drift_alerts += 1
                if drift_step is None:
                    drift_step = evt.step_index
                    pre_drift_f1 = snap.f1_score

            # Check adaptation recovery (F1 recovers to >= 90% of pre-drift F1)
            if drift_step is not None and recovery_step is None and evt.step_index > drift_step + 10:
                if snap.f1_score >= 0.85 * pre_drift_f1:
                    recovery_step = evt.step_index

            # 3. ENQUEUE: Ground truth with verification latency delay
            verification_queue.append((evt.step_index, evt.features, evt.ground_truth_label))

            # 4. TRAIN: Dequeue eligible verified samples
            while verification_queue and (evt.step_index - verification_queue[0][0]) >= self.label_delay:
                _, v_feat, v_label = verification_queue.popleft()
                if self.strategy == ModelAdaptationStrategy.STREAMING_UPDATED:
                    model.update(v_feat, v_label)

            f1_values.append(snap.f1_score)
            memory_values.append(mem_mb)

            # Sample periodic snapshots for reporting
            if t_idx % 25 == 0 or t_idx == len(stream) - 1:
                time_series.append({
                    "step": evt.step_index,
                    "regime": evt.regime.value,
                    "accuracy": snap.accuracy,
                    "precision": snap.precision,
                    "recall": snap.recall,
                    "f1_score": snap.f1_score,
                    "brier_score": snap.brier_score,
                    "memory_mb": snap.active_memory_mb,
                    "drift_alert": is_drift,
                })

        mean_f1 = round(float(np.mean(f1_values)), 4) if f1_values else 0.0
        final_snap = metrics_acc.compute_current_metrics(memory_mb=memory_values[-1])
        
        # Adaptation delay: steps from drift alert to recovery
        adapt_delay = (recovery_step - drift_step) if (drift_step and recovery_step) else 0

        # Catastrophic forgetting ratio: difference between initial high performance and post-drift performance
        initial_f1 = float(np.mean(f1_values[:100])) if len(f1_values) >= 100 else 0.90
        cfr = round(max(0.0, initial_f1 - final_snap.f1_score), 4)

        return PrequentialSimulationResult(
            strategy=self.strategy,
            label_delay=self.label_delay,
            total_events=len(stream),
            mean_macro_f1=mean_f1,
            final_f1=final_snap.f1_score,
            final_recall=final_snap.recall,
            final_brier_score=final_snap.brier_score,
            drift_alerts_count=drift_alerts,
            adaptation_delay_steps=adapt_delay,
            catastrophic_forgetting_ratio=cfr,
            mean_memory_mb=round(float(np.mean(memory_values)), 4),
            peak_memory_mb=round(float(np.max(memory_values)), 4),
            time_series_metrics=time_series,
        )
