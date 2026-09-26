from __future__ import annotations
"""
AHRAS Module 9 / Extension J — Confidence-Gated Pseudo-Label Validation Engine
--------------------------------------------------------------------------------
Implements safe, uncertainty-gated semi-supervised continual adaptation:
  - Validates pseudo-labels via strict multi-condition epistemic gating:
      1. Confidence Gate (p >= tau_high or p <= tau_low)
      2. Epistemic Uncertainty Gate (U <= tau_unc)
      3. In-Distribution (OOD) Gate (OOD <= tau_ood)
      4. Temporal Trajectory Stability Gate (no flip oscillations)
      5. Cross-Modal Consistency Gate
  - Assigns explicit provenance ("PSEUDO_VALIDATED" vs "HUMAN_VERIFIED") with discounted sample weights.
  - Supports atomic generation-based quarantine and rollback to prevent confirmation bias and drift.
"""

import copy
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    PSEUDO_BUFFER_CAPACITY,
    PSEUDO_CONF_HIGH,
    PSEUDO_CONF_LOW,
    PSEUDO_MAX_OOD,
    PSEUDO_MAX_UNCERTAINTY,
    PSEUDO_SAMPLE_WEIGHT,
    USE_PSEUDO_LABEL_LEARNER,
)

log = logging.getLogger(__name__)


class LabelProvenance(str, Enum):
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    PSEUDO_VALIDATED = "PSEUDO_VALIDATED"
    QUARANTINED = "QUARANTINED"
    PURGED = "PURGED"


class PseudoLabelDecision(str, Enum):
    PSEUDO_LABEL_ATTACK = "PSEUDO_LABEL_ATTACK"
    PSEUDO_LABEL_BENIGN = "PSEUDO_LABEL_BENIGN"
    ROUTE_HUMAN_ACTIVE_LEARNING = "ROUTE_HUMAN_ACTIVE_LEARNING"
    REJECT_OOD = "REJECT_OOD"
    REJECT_UNSTABLE = "REJECT_UNSTABLE"
    REJECT_INCONSISTENT = "REJECT_INCONSISTENT"


@dataclass
class PseudoLabelRecord:
    """Represents a sample evaluated for pseudo-labeling with full provenance metadata."""
    record_id: str
    event_id: str
    entity_key: str
    predicted_risk: float
    confidence: float
    epistemic_uncertainty: float
    ood_score: float
    temporal_instability: float
    cross_modal_consistency: float
    assigned_label: Optional[int]
    decision: str
    provenance: str
    sample_weight: float
    generation: int
    features: Dict[str, float] = field(default_factory=dict)
    source_models: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PseudoLabelEngine:
    """
    Validates, filters, and manages pseudo-labeled events for continual semi-supervised training.
    Enforces multi-tier epistemic and distribution gates to eliminate confirmation bias.
    """

    def __init__(
        self,
        conf_high: float = PSEUDO_CONF_HIGH,
        conf_low: float = PSEUDO_CONF_LOW,
        max_uncertainty: float = PSEUDO_MAX_UNCERTAINTY,
        max_ood: float = PSEUDO_MAX_OOD,
        sample_weight: float = PSEUDO_SAMPLE_WEIGHT,
        buffer_capacity: int = PSEUDO_BUFFER_CAPACITY,
    ):
        self.conf_high = float(conf_high)
        self.conf_low = float(conf_low)
        self.max_uncertainty = float(max_uncertainty)
        self.max_ood = float(max_ood)
        self.sample_weight = float(sample_weight)
        self.buffer_capacity = int(buffer_capacity)

        self._validated_buffer: deque[PseudoLabelRecord] = deque(maxlen=self.buffer_capacity)
        self._human_verified_buffer: deque[PseudoLabelRecord] = deque(maxlen=self.buffer_capacity)
        self._quarantined_records: List[PseudoLabelRecord] = []
        self._generation_counter: int = 1
        self._lock = threading.RLock()

    @property
    def current_generation(self) -> int:
        return self._generation_counter

    def advance_generation(self) -> int:
        """Increments generation index for continual learning iteration checkpointing."""
        with self._lock:
            self._generation_counter += 1
            log.info(f"[PSEUDO LABEL] Advanced to generation {self._generation_counter}")
            return self._generation_counter

    def evaluate_sample(
        self,
        event_id: str,
        entity_key: str,
        predicted_risk: float,
        confidence: float,
        epistemic_uncertainty: float,
        ood_score: float,
        temporal_instability: float = 0.0,
        cross_modal_consistency: float = 1.0,
        features: Optional[Dict[str, float]] = None,
        source_models: Optional[List[str]] = None,
    ) -> PseudoLabelRecord:
        """
        Evaluates an unlabeled security telemetry event against all validation gates:
          1. OOD Gate: Rejects out-of-distribution events (routes to human queue).
          2. Epistemic Uncertainty Gate: Rejects ambiguous/high-variance events.
          3. Temporal Stability Gate: Rejects oscillating or volatile trajectories.
          4. Cross-Modal Consistency Gate: Rejects conflicting multi-modal views.
          5. Confidence Extremity Gate: Assigns label only if prediction is ultra-confident.
        """
        with self._lock:
            rec_id = f"PL-{uuid.uuid4().hex[:8]}"
            feats = features or {}
            models = source_models or ["ensemble"]
            now = time.time()

            # Gate 1: Out-of-Distribution Gate
            # Never pseudo-label out-of-distribution data (unseen attack variants / distribution shift)
            if ood_score > self.max_ood:
                return PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=None,
                    decision=PseudoLabelDecision.REJECT_OOD.value,
                    provenance=LabelProvenance.QUARANTINED.value,
                    sample_weight=0.0,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason=f"OOD score {ood_score:.3f} > {self.max_ood:.3f}. Out-of-distribution samples must be human-reviewed.",
                    timestamp=now,
                )

            # Gate 2: Epistemic Uncertainty Gate
            # Ambiguous events with model uncertainty require human analyst inquiry
            if epistemic_uncertainty > self.max_uncertainty:
                return PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=None,
                    decision=PseudoLabelDecision.ROUTE_HUMAN_ACTIVE_LEARNING.value,
                    provenance=LabelProvenance.QUARANTINED.value,
                    sample_weight=0.0,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason=f"Epistemic uncertainty {epistemic_uncertainty:.3f} > {self.max_uncertainty:.3f}. Requires analyst review.",
                    timestamp=now,
                )

            # Gate 3: Temporal Trajectory Stability Gate
            # High volatility indicates model boundary instability; do not reinforce instability
            if temporal_instability > 0.25:
                return PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=None,
                    decision=PseudoLabelDecision.REJECT_UNSTABLE.value,
                    provenance=LabelProvenance.QUARANTINED.value,
                    sample_weight=0.0,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason=f"Temporal instability {temporal_instability:.3f} > 0.250. Trajectory shows classification volatility.",
                    timestamp=now,
                )

            # Gate 4: Cross-Modal Consistency Gate
            if cross_modal_consistency < 0.70:
                return PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=None,
                    decision=PseudoLabelDecision.REJECT_INCONSISTENT.value,
                    provenance=LabelProvenance.QUARANTINED.value,
                    sample_weight=0.0,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason=f"Cross-modal consistency {cross_modal_consistency:.2f} < 0.70. Sensors disagree.",
                    timestamp=now,
                )

            # Gate 5: Confidence Extremity Gate
            # Attack validation: High risk (>= conf_high) and High model confidence
            if predicted_risk >= self.conf_high and confidence >= 0.85:
                rec = PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=1,
                    decision=PseudoLabelDecision.PSEUDO_LABEL_ATTACK.value,
                    provenance=LabelProvenance.PSEUDO_VALIDATED.value,
                    sample_weight=self.sample_weight,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason="",
                    timestamp=now,
                )
                self._validated_buffer.append(rec)
                log.info(f"[PSEUDO LABEL] Accepted ATTACK pseudo-label for {event_id} (Risk={predicted_risk:.3f}, Conf={confidence:.3f})")
                return rec

            # Benign validation: Low risk (<= conf_low) and High model confidence in benign status
            elif predicted_risk <= self.conf_low and confidence >= 0.85:
                rec = PseudoLabelRecord(
                    record_id=rec_id,
                    event_id=event_id,
                    entity_key=entity_key,
                    predicted_risk=round(predicted_risk, 4),
                    confidence=round(confidence, 4),
                    epistemic_uncertainty=round(epistemic_uncertainty, 4),
                    ood_score=round(ood_score, 4),
                    temporal_instability=round(temporal_instability, 4),
                    cross_modal_consistency=round(cross_modal_consistency, 4),
                    assigned_label=0,
                    decision=PseudoLabelDecision.PSEUDO_LABEL_BENIGN.value,
                    provenance=LabelProvenance.PSEUDO_VALIDATED.value,
                    sample_weight=self.sample_weight,
                    generation=self._generation_counter,
                    features=feats,
                    source_models=models,
                    rejection_reason="",
                    timestamp=now,
                )
                self._validated_buffer.append(rec)
                log.info(f"[PSEUDO LABEL] Accepted BENIGN pseudo-label for {event_id} (Risk={predicted_risk:.3f}, Conf={confidence:.3f})")
                return rec

            # Otherwise: In the ambiguous decision margin -> Route to human active learning
            return PseudoLabelRecord(
                record_id=rec_id,
                event_id=event_id,
                entity_key=entity_key,
                predicted_risk=round(predicted_risk, 4),
                confidence=round(confidence, 4),
                epistemic_uncertainty=round(epistemic_uncertainty, 4),
                ood_score=round(ood_score, 4),
                temporal_instability=round(temporal_instability, 4),
                cross_modal_consistency=round(cross_modal_consistency, 4),
                assigned_label=None,
                decision=PseudoLabelDecision.ROUTE_HUMAN_ACTIVE_LEARNING.value,
                provenance=LabelProvenance.QUARANTINED.value,
                sample_weight=0.0,
                generation=self._generation_counter,
                features=feats,
                source_models=models,
                rejection_reason=f"Risk {predicted_risk:.3f} is in ambiguous margin [{self.conf_low:.2f}, {self.conf_high:.2f}].",
                timestamp=now,
            )

    def add_human_verified_sample(
        self,
        event_id: str,
        entity_key: str,
        ground_truth_label: int,
        predicted_risk: float = 0.50,
        features: Optional[Dict[str, float]] = None,
    ) -> PseudoLabelRecord:
        """Records a 100% human-verified sample with full sample weight (1.00)."""
        with self._lock:
            rec = PseudoLabelRecord(
                record_id=f"HV-{uuid.uuid4().hex[:8]}",
                event_id=event_id,
                entity_key=entity_key,
                predicted_risk=round(predicted_risk, 4),
                confidence=1.0,
                epistemic_uncertainty=0.0,
                ood_score=0.0,
                temporal_instability=0.0,
                cross_modal_consistency=1.0,
                assigned_label=int(ground_truth_label),
                decision="HUMAN_CONFIRMED",
                provenance=LabelProvenance.HUMAN_VERIFIED.value,
                sample_weight=1.00,  # Full weight for ground truth
                generation=self._generation_counter,
                features=features or {},
                source_models=["HUMAN_ANALYST"],
                rejection_reason="",
                timestamp=time.time(),
            )
            self._human_verified_buffer.append(rec)
            return rec

    def quarantine_generation(self, generation: int) -> int:
        """
        Quarantines all pseudo-labeled records from a specific generation (e.g. upon detecting drift or degradation).
        Preserves human-verified data intact.
        """
        with self._lock:
            quarantined_count = 0
            surviving = deque(maxlen=self.buffer_capacity)
            for rec in self._validated_buffer:
                if rec.generation == generation:
                    rec.provenance = LabelProvenance.QUARANTINED.value
                    self._quarantined_records.append(rec)
                    quarantined_count += 1
                else:
                    surviving.append(rec)
            self._validated_buffer = surviving
            log.warning(f"[PSEUDO LABEL] Quarantined {quarantined_count} pseudo-labels from generation {generation}")
            return quarantined_count

    def purge_quarantined(self) -> int:
        """Permanently purges quarantined pseudo-labels."""
        with self._lock:
            count = len(self._quarantined_records)
            self._quarantined_records.clear()
            log.info(f"[PSEUDO LABEL] Purged {count} quarantined records from memory")
            return count

    def get_training_dataset(self) -> List[Tuple[Dict[str, float], int, float, str]]:
        """
        Returns combined training samples with explicit sample weights and provenance:
          [(features, label, sample_weight, provenance)]
        """
        with self._lock:
            dataset = []
            # 1. Human verified samples (weight = 1.0)
            for h in self._human_verified_buffer:
                if h.assigned_label is not None:
                    dataset.append((h.features, h.assigned_label, h.sample_weight, h.provenance))
            # 2. Validated pseudo-labels (weight = 0.5)
            for p in self._validated_buffer:
                if p.assigned_label is not None and p.provenance == LabelProvenance.PSEUDO_VALIDATED.value:
                    dataset.append((p.features, p.assigned_label, p.sample_weight, p.provenance))
            return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics and acceptance/rejection breakdowns."""
        with self._lock:
            validated = list(self._validated_buffer)
            n_attack = sum(1 for r in validated if r.assigned_label == 1)
            n_benign = sum(1 for r in validated if r.assigned_label == 0)
            return {
                "current_generation": self._generation_counter,
                "validated_pseudo_labels_count": len(validated),
                "validated_attack_count": n_attack,
                "validated_benign_count": n_benign,
                "human_verified_count": len(self._human_verified_buffer),
                "quarantined_count": len(self._quarantined_records),
                "sample_weight_pseudo": self.sample_weight,
                "sample_weight_human": 1.00,
                "conf_high_threshold": self.conf_high,
                "conf_low_threshold": self.conf_low,
                "max_uncertainty_threshold": self.max_uncertainty,
                "max_ood_threshold": self.max_ood,
            }


# Singleton accessor
_pseudo_engine_instance: Optional[PseudoLabelEngine] = None
_pseudo_lock = threading.Lock()


def get_pseudo_label_engine() -> PseudoLabelEngine:
    global _pseudo_engine_instance
    with _pseudo_lock:
        if _pseudo_engine_instance is None:
            _pseudo_engine_instance = PseudoLabelEngine()
    return _pseudo_engine_instance
