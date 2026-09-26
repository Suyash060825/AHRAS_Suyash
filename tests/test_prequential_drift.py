"""
Tests for Streaming Prequential Evaluation & Concept Drift Harness (Research Frontier D / EXP-25)
--------------------------------------------------------------------------------------------------
Validates:
- TemporalStreamGenerator chronological monotonicity and 4 drift regimes
- Page-Hinkley and ADWIN change-point drift detectors
- Prequential sliding window metrics calculation
- Strict test-then-train protocol and verification delay queue
- Streaming adaptation gain over static baseline under non-stationarity
- End-to-end artifact generation and JSON/LaTeX export
"""

import json
from pathlib import Path
import pytest
import numpy as np

from streaming.temporal_split import (
    StreamingTemporalEvent,
    TemporalStreamGenerator,
    DriftRegimeType,
)
from streaming.drift_detector import (
    PageHinkleyDriftDetector,
    ADWINDriftDetector,
    StreamingDriftSignal,
)
from streaming.drift_metrics import (
    PrequentialWindowMetrics,
    PrequentialMetricsAccumulator,
)
from streaming.prequential_runner import (
    PrequentialEvaluationRunner,
    PrequentialSimulationResult,
    ModelAdaptationStrategy,
    OnlineLinearClassifier,
)
from evaluation.run_prequential_drift import run_benchmark


class TestTemporalStreamGenerator:
    def test_chronological_monotonicity(self):
        gen = TemporalStreamGenerator(seed=42)
        stream = gen.generate_stream(n_steps=200)

        assert len(stream) == 200
        for i in range(1, len(stream)):
            # Monotonic time invariant
            assert stream[i].timestamp > stream[i - 1].timestamp
            assert stream[i].step_index == stream[i - 1].step_index + 1

    def test_regimes_representation(self):
        gen = TemporalStreamGenerator(seed=42)
        stream = gen.generate_stream(n_steps=400)

        regimes_found = {e.regime for e in stream}
        assert DriftRegimeType.BASELINE_STATIONARY in regimes_found
        assert DriftRegimeType.SUDDEN_DRIFT in regimes_found
        assert DriftRegimeType.GRADUAL_DRIFT in regimes_found
        assert DriftRegimeType.RECURRING_DRIFT in regimes_found


class TestDriftDetectors:
    def test_page_hinkley_drift_trigger(self):
        ph = PageHinkleyDriftDetector(threshold=10.0)
        
        # Stationary low-loss sequence
        for _ in range(50):
            sig = ph.update(0.05)
            assert sig.detected is False

        # Abrupt surge in loss
        drift_triggered = False
        for _ in range(50):
            sig = ph.update(1.8)
            if sig.detected:
                drift_triggered = True
                break
        assert drift_triggered is True

    def test_adwin_drift_trigger(self):
        adwin = ADWINDriftDetector(delta_p=0.01)

        # Baseline mean = 0.1
        for _ in range(60):
            adwin.update(0.1)

        # Sudden mean jump to 2.5
        drift_triggered = False
        for _ in range(60):
            sig = adwin.update(2.5)
            if sig.detected:
                drift_triggered = True
                break
        assert drift_triggered is True


class TestPrequentialMetrics:
    def test_window_metrics_calculation(self):
        acc = PrequentialMetricsAccumulator(window_size=20)
        
        # Feed 10 true positives and 10 true negatives
        for _ in range(10):
            acc.record_prediction(y_pred=1, y_true=1, p_pred=0.9)
        for _ in range(10):
            snap = acc.record_prediction(y_pred=0, y_true=0, p_pred=0.1)

        assert snap.accuracy == 1.0
        assert snap.precision == 1.0
        assert snap.recall == 1.0
        assert snap.f1_score == 1.0
        assert snap.brier_score < 0.05


class TestPrequentialRunner:
    def test_streaming_adapts_under_sudden_drift(self):
        gen = TemporalStreamGenerator(seed=42)
        stream = gen.generate_stream(
            n_steps=300,
            regimes=[(DriftRegimeType.BASELINE_STATIONARY, 100), (DriftRegimeType.SUDDEN_DRIFT, 200)],
        )

        static_runner = PrequentialEvaluationRunner(strategy=ModelAdaptationStrategy.STATIC)
        static_res = static_runner.run_simulation(stream)

        streaming_runner = PrequentialEvaluationRunner(strategy=ModelAdaptationStrategy.STREAMING_UPDATED, label_delay=0)
        streaming_res = streaming_runner.run_simulation(stream)

        # Streaming model should outperform static model after drift
        assert streaming_res.final_f1 > static_res.final_f1
        assert streaming_res.final_brier_score < static_res.final_brier_score

    def test_label_delay_queue_behavior(self):
        gen = TemporalStreamGenerator(seed=42)
        stream = gen.generate_stream(n_steps=150)

        runner_imm = PrequentialEvaluationRunner(ModelAdaptationStrategy.STREAMING_UPDATED, label_delay=0)
        runner_del = PrequentialEvaluationRunner(ModelAdaptationStrategy.STREAMING_UPDATED, label_delay=50)

        res_imm = runner_imm.run_simulation(stream)
        res_del = runner_del.run_simulation(stream)

        # Memory footprint increases with queue depth
        assert res_del.peak_memory_mb >= res_imm.peak_memory_mb


class TestPrequentialArtifacts:
    def test_end_to_end_benchmark_run(self):
        report = run_benchmark()
        assert report["experiment_id"] == "EXP-25"
        assert "comparisons" in report
        assert "regime_comparisons" in report

        json_path = Path("evaluation/results/PREQUENTIAL_DRIFT_REPORT.json")
        latex_path = Path("publication/tables/prequential_drift.tex")

        assert json_path.exists()
        assert latex_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiment_id"] == "EXP-25"

        with open(latex_path, "r", encoding="utf-8") as f:
            latex = f.read()
        assert r"\begin{table*}" in latex
        assert r"Streaming Prequential Evaluation" in latex
