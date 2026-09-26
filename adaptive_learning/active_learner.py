from __future__ import annotations
"""
AHRAS Module — Active Learning & Uncertainty-Driven Analyst Labeling Loop
--------------------------------------------------------------------------
Implements an information-theoretic active learning acquisition loop for SOC analysts:

  1. Acquisition Score Computation:
       Selects ambiguous, high-uncertainty events using composite acquisition:
         a(x) = Uncertainty(x) * InformationGainEstimate(x) * (1 + OOD_Score(x))

  2. Budgeted Query Management:
       Enforces SOC analyst inquiry budget per time window to prevent alert fatigue.

  3. Feedback Injection to Continual Learning:
       Incorporates confirmed analyst ground-truth labels directly into the
       multi-memory replay buffers for online retraining.
"""

import time
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ActiveLearningRequest:
    """Represents a sample queried for human analyst annotation."""
    request_id:         str
    event_id:           str
    entity_key:         str
    timestamp:          float
    uncertainty:        float
    ood_score:          float
    acquisition_score:  float
    suggested_label:    int        # Model's tentative prediction (0=benign, 1=attack)
    suggested_risk:     float
    features:           Dict[str, float]
    status:             str = "PENDING"  # "PENDING", "LABELED", "REJECTED"
    analyst_label:      Optional[int] = None
    resolved_time:      Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ActiveLearner:
    """
    Manages active query acquisition, budget constraints, and analyst label ingestion.
    Integrates with Confidence-Gated PseudoLabelEngine for semi-supervised workload reduction.
    """

    def __init__(
        self,
        budget_per_window: int = 50,
        window_sec: float = 3600.0,
        seed: int = 42,
        pseudo_label_engine: Optional[Any] = None,
    ):
        self.budget_per_window = budget_per_window
        self.window_sec = window_sec
        self._pending_requests: Dict[str, ActiveLearningRequest] = {}
        self._resolved_requests: List[ActiveLearningRequest] = []
        self._window_queries: deque = deque()  # stores timestamps of queries

        if pseudo_label_engine is not None:
            self.pseudo_label_engine = pseudo_label_engine
        else:
            try:
                from adaptive_learning.pseudo_labeler import get_pseudo_label_engine
                from config.settings import USE_PSEUDO_LABEL_LEARNER
                self.pseudo_label_engine = get_pseudo_label_engine() if USE_PSEUDO_LABEL_LEARNER else None
            except Exception as e:
                log.warning(f"Could not initialize PseudoLabelEngine in ActiveLearner: {e}")
                self.pseudo_label_engine = None

    def compute_acquisition_score(
        self,
        uncertainty: float,
        ood_score: float,
        entropy: float = 0.50
    ) -> float:
        """
        Information-theoretic acquisition function:
          score = uncertainty * entropy * (1.0 + ood_score)
        """
        u = max(0.0, min(1.0, float(uncertainty)))
        ood = max(0.0, min(1.0, float(ood_score)))
        h = max(0.0, min(1.0, float(entropy)))
        
        score = u * (0.50 + 0.50 * h) * (1.0 + ood)
        return round(score, 4)

    def should_query(
        self,
        uncertainty: float,
        ood_score: float,
        abstain_action: str
    ) -> bool:
        """
        Determines if sample qualifies for active learning query within budget.
        """
        now = time.time()
        # Clean old timestamps from rolling window
        while self._window_queries and (now - self._window_queries[0] > self.window_sec):
            self._window_queries.popleft()

        if len(self._window_queries) >= self.budget_per_window:
            # Budget exhausted for current window
            return False

        is_abstain_or_escalate = abstain_action in ("ABSTAIN", "ESCALATE_ANALYST")
        is_uncertain = (uncertainty >= 0.30 or ood_score >= 0.60)
        
        return is_abstain_or_escalate or is_uncertain

    def create_request(
        self,
        event_id: str,
        entity_key: str,
        uncertainty: float,
        ood_score: float,
        risk_score: float,
        features: Optional[Dict[str, float]] = None
    ) -> ActiveLearningRequest:
        """Creates and enqueues an active learning request."""
        now = time.time()
        acq_score = self.compute_acquisition_score(uncertainty, ood_score)
        suggested = 1 if risk_score >= 0.50 else 0
        req_id = f"AL-REQ-{len(self._pending_requests) + len(self._resolved_requests) + 1:04d}"
        
        req = ActiveLearningRequest(
            request_id=req_id,
            event_id=event_id,
            entity_key=entity_key,
            timestamp=now,
            uncertainty=round(uncertainty, 4),
            ood_score=round(ood_score, 4),
            acquisition_score=acq_score,
            suggested_label=suggested,
            suggested_risk=round(risk_score, 4),
            features=features or {},
            status="PENDING"
        )
        
        self._pending_requests[req_id] = req
        self._window_queries.append(now)
        return req

    def resolve_label(self, request_id: str, ground_truth_label: int) -> Optional[ActiveLearningRequest]:
        """
        Records human analyst ground truth and marks request resolved.
        Also registers confirmed ground truth with PseudoLabelEngine if available.
        """
        req = self._pending_requests.pop(request_id, None)
        if not req:
            return None
            
        req.analyst_label = int(ground_truth_label)
        req.status = "LABELED"
        req.resolved_time = time.time()
        self._resolved_requests.append(req)

        if self.pseudo_label_engine is not None:
            self.pseudo_label_engine.add_human_verified_sample(
                event_id=req.event_id,
                entity_key=req.entity_key,
                ground_truth_label=int(ground_truth_label),
                predicted_risk=req.suggested_risk,
                features=req.features,
            )

        return req

    def triage_stream_sample(
        self,
        event_id: str,
        entity_key: str,
        risk_score: float,
        confidence: float,
        uncertainty: float,
        ood_score: float,
        features: Optional[Dict[str, float]] = None,
        temporal_instability: float = 0.0,
        cross_modal_consistency: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Hierarchical streaming triage combining Confidence-Gated Pseudo-Labeling with Active Learning:
          1. Evaluates sample against strict epistemic pseudo-label gates.
          2. If ultra-confident and in-distribution: auto-generates validated pseudo-label (bypassing analyst).
          3. If ambiguous, OOD, or uncertain: routes to human active learning queue if budget permits.
        """
        pseudo_rec = None
        auto_accepted = False

        if self.pseudo_label_engine is not None:
            pseudo_rec = self.pseudo_label_engine.evaluate_sample(
                event_id=event_id,
                entity_key=entity_key,
                predicted_risk=risk_score,
                confidence=confidence,
                epistemic_uncertainty=uncertainty,
                ood_score=ood_score,
                temporal_instability=temporal_instability,
                cross_modal_consistency=cross_modal_consistency,
                features=features,
            )
            if pseudo_rec.assigned_label is not None:
                auto_accepted = True

        # If not auto-accepted as pseudo-label, check if it qualifies for human active learning
        human_queried = False
        active_req = None
        if not auto_accepted:
            if self.should_query(uncertainty=uncertainty, ood_score=ood_score, abstain_action="ESCALATE_ANALYST"):
                active_req = self.create_request(
                    event_id=event_id,
                    entity_key=entity_key,
                    uncertainty=uncertainty,
                    ood_score=ood_score,
                    risk_score=risk_score,
                    features=features,
                )
                human_queried = True

        return {
            "event_id": event_id,
            "auto_accepted_pseudo": auto_accepted,
            "pseudo_label": pseudo_rec.assigned_label if pseudo_rec else None,
            "pseudo_decision": pseudo_rec.decision if pseudo_rec else "NO_ENGINE",
            "provenance": pseudo_rec.provenance if pseudo_rec else "NONE",
            "human_queried": human_queried,
            "active_request_id": active_req.request_id if active_req else None,
        }

    def get_pending(self) -> List[ActiveLearningRequest]:
        return list(self._pending_requests.values())

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "pending_count": len(self._pending_requests),
            "resolved_count": len(self._resolved_requests),
            "queries_in_current_window": len(self._window_queries),
            "budget_per_window": self.budget_per_window,
        }
        if self.pseudo_label_engine is not None:
            stats["pseudo_label_stats"] = self.pseudo_label_engine.get_statistics()
        return stats

