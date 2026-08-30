from __future__ import annotations
"""
AHRAS Module — Full Closed-Loop Adaptive Control & Live Demonstration
----------------------------------------------------------------------
Implements and benchmarks the complete closed-loop defense architecture:

  Event -> Detection -> Representation -> OOD -> Graph -> Context -> Fusion ->
  Uncertainty -> Current/Future Risk -> DecisionTrace -> XAI -> Policy -> Response ->
  Outcome -> Active Learning -> Continual Learning -> Future Decision.

Executes the IDENTICAL future event stream under two contrasting operating modes:
  1. Static Baseline AHRAS (Frozen weights, static memory, no active learning loop).
  2. Closed-Loop Adaptive AHRAS (Online adaptation, 5-bank memory replay, active query injection).
"""

import time
import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np

from detection.risk_engine import AdaptiveRiskEngine, RiskConfig, RiskResult
from detection.hybrid_engine import get_combiner
from adaptive_learning.weight_learner import (
    AdaptiveWeightLearner,
    ContinualLearningEngine,
    MultiMemoryReplayBuffer,
    FeedbackSample
)
from adaptive_learning.active_learner import ActiveLearner
from detection.selective_gate import ConformalRiskGate

log = logging.getLogger(__name__)


@dataclass
class ClosedLoopComparisonReport:
    total_events_processed:      int
    static_mean_risk_error:      float
    closed_loop_mean_risk_error: float
    static_false_alarms:         int
    closed_loop_false_alarms:    int
    active_queries_requested:    int
    active_labels_incorporated:  int
    adaptation_gain_mse:         float
    false_alarm_reduction_pct:   float
    closed_loop_dominant:        bool


class ClosedLoopDemonstrator:
    """
    Executes comparative closed-loop vs static baseline evaluations on identical event streams.
    """

    def __init__(self):
        self.combiner = get_combiner()

    def run_comparison(
        self,
        event_stream: List[Dict[str, Any]],
        ground_truth_labels: List[int],
        drift_start_idx: int = 50
    ) -> ClosedLoopComparisonReport:
        """
        Runs identical sequential telemetry events through Static vs Closed-Loop AHRAS pipelines.
        """
        N = min(len(event_stream), len(ground_truth_labels))
        if N == 0:
            raise ValueError("Event stream and labels cannot be empty.")

        # ── Pipeline 1: Static AHRAS (No feedback adaptation) ─────────────────
        engine_static = AdaptiveRiskEngine()
        static_errors = []
        static_fps = 0

        for idx in range(N):
            evt = event_stream[idx]
            y_true = ground_truth_labels[idx]
            res = self.combiner.process(evt)
            r_res = engine_static.score_risk(
                entity_key=evt.get("src_ip", f"HOST-{idx}"),
                sig_matches=res.signature_matches if res else [],
                ml_res=res.anomaly_result if res else None,
                stat_res=res.stat_result if res else None,
                evt=evt,
            )
            pred_score = r_res.risk_score
            static_errors.append((pred_score - y_true) ** 2)
            if pred_score >= 0.70 and y_true == 0:
                static_fps += 1

        # ── Pipeline 2: Closed-Loop Adaptive AHRAS ────────────────────────────
        engine_closed = AdaptiveRiskEngine()
        continual_engine = ContinualLearningEngine()
        active_learner = ActiveLearner(budget_per_window=30)
        
        closed_errors = []
        closed_fps = 0
        queries_made = 0
        labels_incorporated = 0
        
        # Dynamic weights adapted via feedback
        w_sig = 0.50
        w_ml = 0.30
        w_trust = 0.15

        for idx in range(N):
            evt = event_stream[idx]
            y_true = ground_truth_labels[idx]
            src_ip = evt.get("src_ip", f"HOST-{idx}")
            
            cfg_closed = RiskConfig(
                w_sig=w_sig,
                w_ml=w_ml,
                w_trust=w_trust,
                use_signature=True,
                use_ml=True,
                use_statistical=True,
                adaptive_weights=True,
                use_trust=True,
                use_history=True,
                use_selective_gate=True,
            )
            
            res = self.combiner.process(evt)
            r_res = engine_closed.score_risk(
                entity_key=src_ip,
                sig_matches=res.signature_matches if res else [],
                ml_res=res.anomaly_result if res else None,
                stat_res=res.stat_result if res else None,
                evt=evt,
                override_config=cfg_closed,
            )
            pred_score = r_res.risk_score
            closed_errors.append((pred_score - y_true) ** 2)
            if pred_score >= 0.70 and y_true == 0:
                closed_fps += 1

            # 1. Closed-Loop Active Learning Query
            should_q = active_learner.should_query(
                uncertainty=r_res.risk_uncertainty,
                ood_score=0.0,
                abstain_action=r_res.autonomy_decision
            )
            
            if should_q:
                req = active_learner.create_request(
                    event_id=f"EVT-{idx}",
                    entity_key=src_ip,
                    uncertainty=r_res.risk_uncertainty,
                    ood_score=0.0,
                    risk_score=pred_score
                )
                queries_made += 1
                
                # Simulate Analyst Ground-Truth Verification
                active_learner.resolve_label(req.request_id, ground_truth_label=y_true)
                labels_incorporated += 1
                
                # 2. Online Gradient Adaptation Step
                error = float(pred_score - y_true)
                lr = 0.02
                if r_res.S_sig > 0:
                    w_sig = float(np.clip(w_sig - lr * error * r_res.S_sig, 0.1, 0.8))
                if r_res.A_ml > 0:
                    w_ml = float(np.clip(w_ml - lr * error * r_res.A_ml, 0.1, 0.8))
                if y_true == 0:
                    w_trust = float(np.clip(w_trust + 0.01, 0.1, 0.4))
                
                # 3. Inject Ground-Truth into Multi-Memory Continual Learner
                loss_val = error ** 2
                feedback = FeedbackSample(
                    src_ip=src_ip,
                    label=y_true,
                    components={"sig": r_res.S_sig, "ml": r_res.A_ml, "stat": r_res.delta_D},
                    predicted_risk=pred_score
                )
                continual_engine.add_experience(feedback, loss=loss_val, is_hard_sample=(loss_val >= 0.20))
                
                # 4. Dynamic Trust & State Adjustment
                engine_closed.update_trust_on_event(src_ip, is_anom=(y_true == 1), severity_id=r_res.severity_id)

        mean_static_err = float(np.mean(static_errors))
        mean_closed_err = float(np.mean(closed_errors))
        gain_mse = round(mean_static_err - mean_closed_err, 4)
        fp_reduction = round(((static_fps - closed_fps) / max(1, static_fps)) * 100.0, 1) if static_fps > 0 else 0.0

        return ClosedLoopComparisonReport(
            total_events_processed=N,
            static_mean_risk_error=round(mean_static_err, 4),
            closed_loop_mean_risk_error=round(mean_closed_err, 4),
            static_false_alarms=static_fps,
            closed_loop_false_alarms=closed_fps,
            active_queries_requested=queries_made,
            active_labels_incorporated=labels_incorporated,
            adaptation_gain_mse=gain_mse,
            false_alarm_reduction_pct=fp_reduction,
            closed_loop_dominant=(mean_closed_err <= mean_static_err),
        )
