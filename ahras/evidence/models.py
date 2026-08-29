from __future__ import annotations
"""
AHRAS Evidence Model Specification (Tamper-Evident Append-Only Hash Chain)
---------------------------------------------------------------------------
Formal data models representing discrete, auditable detection contributions.
Every score, alert, XAI explanation, and response decision derives directly
from an immutable, provenance-tracked EvidenceRecord chained into a cryptographic ledger.

Chain Equation:
    H_0 = SHA256(canonical_record_0 || "0"*64)
    H_n = SHA256(canonical_record_n || H_(n-1))
"""

import time
import uuid
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class EvidenceType(str, Enum):
    SIGNATURE           = "signature"
    ML_ANOMALY          = "ml_anomaly"
    STATISTICAL_DRIFT   = "statistical_drift"
    HISTORICAL          = "historical"
    GRAPH_CORRELATION   = "graph_correlation"
    THREAT_INTEL        = "threat_intel"
    ASSET_CRITICALITY   = "asset_criticality"
    IDENTITY_CONTEXT    = "identity_context"
    TEMPORAL_CONTEXT    = "temporal_context"
    FORECAST            = "forecast"
    DECEPTION           = "deception"


class EvidenceSource(str, Enum):
    SURICATA_ENGINE       = "suricata_signature_engine"
    ISOLATION_FOREST      = "isolation_forest"
    AUTOENCODER           = "autoencoder_mlp"
    ONE_CLASS_SVM         = "one_class_svm"
    WELFORD_STAT_ENGINE   = "welford_statistical_engine"
    ENTITY_GRAPH_ENGINE   = "temporal_entity_graph"
    HISTORICAL_RECIDIVISM = "historical_risk_engine"
    STIX_TAXII_FEED       = "stix_threat_intel"
    DYNAMIC_HONEYPOT      = "deception_honeypot"
    HOLT_FORECASTER       = "holt_risk_forecaster"
    ASSET_DIRECTORY       = "asset_criticality_directory"


@dataclass
class EvidenceRecord:
    """
    Standardized atomic security evidence record chained with cryptographic pointers.
    Forms the tamper-evident append-only ledger block.
    """
    evidence_id:        str          = field(default_factory=lambda: f"EVD-{uuid.uuid4().hex[:12]}")
    event_id:           str          = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:12]}")
    seq_num:            int          = 0            # Monotonic sequence index in chain
    prev_hash:          str          = "0" * 64     # SHA-256 hash of predecessor record
    entity_id:          str          = "unknown"
    source:             str          = EvidenceSource.SURICATA_ENGINE.value
    detector_type:      str          = EvidenceType.SIGNATURE.value
    detector_version:   str          = "1.0.0"
    raw_score:          float        = 0.0
    normalized_score:   float        = 0.0          # Clamped to [0.0, 1.0]
    confidence:         float        = 1.0          # [0.0, 1.0] reliability/calibration
    uncertainty:        float        = 0.0          # [0.0, 1.0] epistemic/disagreement uncertainty
    timestamp:          float        = field(default_factory=time.time)
    observation_window: float        = 0.0          # Duration in seconds
    provenance:         Dict[str, Any] = field(default_factory=dict)
    mitre_mapping:      List[str]    = field(default_factory=list)
    explanation:        str          = ""
    model_version:      str          = "v1.0"
    feature_version:    str          = "f1.0"
    policy_version:     str          = "p1.0"
    record_hash:        str          = ""

    def __post_init__(self):
        self.normalized_score = max(0.0, min(1.0, float(self.normalized_score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))
        if not self.record_hash:
            self.record_hash = self.compute_hash()

    def canonical_payload(self) -> Dict[str, Any]:
        """Returns deterministic canonical dictionary representation."""
        return {
            "evidence_id":        self.evidence_id,
            "event_id":           self.event_id,
            "seq_num":            self.seq_num,
            "prev_hash":          self.prev_hash,
            "entity_id":          self.entity_id,
            "source":             self.source,
            "detector_type":      self.detector_type,
            "detector_version":   self.detector_version,
            "raw_score":          round(float(self.raw_score), 6),
            "normalized_score":   round(float(self.normalized_score), 6),
            "confidence":         round(float(self.confidence), 6),
            "uncertainty":        round(float(self.uncertainty), 6),
            "timestamp":          round(float(self.timestamp), 3),
            "observation_window": round(float(self.observation_window), 3),
            "mitre_mapping":      sorted(self.mitre_mapping),
            "model_version":      self.model_version,
            "feature_version":    self.feature_version,
            "policy_version":     self.policy_version,
        }

    def compute_hash(self) -> str:
        """Computes H_n = SHA256(canonical_payload || prev_hash)."""
        payload = self.canonical_payload()
        serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        to_hash = serialized.encode('utf-8') + self.prev_hash.encode('utf-8')
        return hashlib.sha256(to_hash).hexdigest()

    def verify_integrity(self) -> bool:
        """Verifies internal SHA-256 consistency of this record."""
        return self.record_hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
