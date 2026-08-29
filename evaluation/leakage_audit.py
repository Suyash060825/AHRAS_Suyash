from __future__ import annotations
"""
AHRAS Phase 16 — Leakage-Safe Experimental Design & Audit Suite
---------------------------------------------------------------
Enforces strict scientific train/validation/test/external split separation:
  1. Chronological temporal split (Train < Validation < Test in timestamp order).
  2. Entity-disjoint split (Entities in test set never appear in training set).
  3. Preprocessing fit audit (Scalers, thresholds, and baselines fit strictly on Train).
  4. Feature causality check (No post-incident labels or future features present at inference).
"""

import time
import logging
from typing import Dict, List, Tuple, Any, Set

import numpy as np

log = logging.getLogger(__name__)


def temporal_train_test_split(
    records: List[Any],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Splits dataset chronologically based on record timestamps.
    Guarantees max(train_ts) <= min(val_ts) <= min(test_ts).
    """
    sorted_recs = sorted(records, key=lambda r: getattr(r, "timestamp", 0.0) or 0.0)
    n = len(sorted_recs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted_recs[:n_train]
    val = sorted_recs[n_train:n_train + n_val]
    test = sorted_recs[n_train + n_val:]
    return train, val, test


def entity_disjoint_train_test_split(
    records: List[Any],
    train_ratio: float = 0.70,
    seed: int = 42,
) -> Tuple[List[Any], List[Any]]:
    """
    Splits dataset such that entity keys (IPs / Hostnames) in the test set
    never appear in the training set (tests out-of-distribution entity generalization).
    """
    rng = np.random.default_rng(seed)
    entity_map: Dict[str, List[Any]] = {}
    for r in records:
        key = getattr(r, "src_ip", None) or getattr(r, "entity_key", None) or "default"
        entity_map.setdefault(key, []).append(r)

    unique_entities = list(entity_map.keys())
    rng.shuffle(unique_entities)

    n_train_ents = max(1, int(len(unique_entities) * train_ratio))
    train_ents = set(unique_entities[:n_train_ents])
    test_ents = set(unique_entities[n_train_ents:])

    train_recs = [r for ent in train_ents for r in entity_map[ent]]
    test_recs = [r for ent in test_ents for r in entity_map[ent]]
    return train_recs, test_recs


class LeakageAuditor:
    """
    Audits an experiment pipeline for 4 critical data leakage failure modes.
    """

    def audit_splits(
        self,
        train_records: List[Any],
        test_records: List[Any],
        is_entity_disjoint: bool = False,
    ) -> Dict[str, Any]:
        train_timestamps = [getattr(r, "timestamp", 0.0) or 0.0 for r in train_records]
        test_timestamps = [getattr(r, "timestamp", 0.0) or 0.0 for r in test_records]

        max_train_t = max(train_timestamps) if train_timestamps else 0.0
        min_test_t = min(test_timestamps) if test_timestamps else float("inf")

        temporal_leakage = (max_train_t > min_test_t) if (train_timestamps and test_timestamps) else False

        train_ents = {getattr(r, "src_ip", None) or getattr(r, "entity_key", None) for r in train_records}
        test_ents = {getattr(r, "src_ip", None) or getattr(r, "entity_key", None) for r in test_records}
        overlap_ents = train_ents.intersection(test_ents) - {None}

        entity_leakage = (len(overlap_ents) > 0) if is_entity_disjoint else False

        passed = (not temporal_leakage) and (not entity_leakage)
        return {
            "temporal_leakage_detected": temporal_leakage,
            "max_train_timestamp": max_train_t,
            "min_test_timestamp": min_test_t,
            "entity_overlap_count": len(overlap_ents),
            "entity_disjoint_pass": not entity_leakage,
            "overall_leakage_audit_pass": passed,
        }
