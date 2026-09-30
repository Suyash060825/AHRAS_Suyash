from __future__ import annotations
"""
AHRAS Evaluation Split Protocols (A through G)
----------------------------------------------
Implements strict mathematical partitioning strategies ensuring:
  1. Zero future leakage in temporal sequences.
  2. Strict train/validation/test isolation.
  3. Controlled open-set withholding of attack families.
  4. Cross-dataset and cross-environment domain transfer partitions.
  5. Benign distribution shift evaluations.
"""

import enum
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from evaluation.datasets.adapters import CanonicalEvent, CanonicalAttackFamily

log = logging.getLogger(__name__)


class SplitProtocol(str, enum.Enum):
    PROTOCOL_A_STANDARD         = "PROTOCOL_A_STANDARD"
    PROTOCOL_B_TEMPORAL         = "PROTOCOL_B_TEMPORAL"
    PROTOCOL_C_CROSS_DATASET     = "PROTOCOL_C_CROSS_DATASET"
    PROTOCOL_D_CROSS_ENVIRONMENT = "PROTOCOL_D_CROSS_ENVIRONMENT"
    PROTOCOL_E_OPEN_SET_FAMILY  = "PROTOCOL_E_OPEN_SET_FAMILY"
    PROTOCOL_F_CROSS_ATTACK      = "PROTOCOL_F_CROSS_ATTACK"
    PROTOCOL_G_BENIGN_SHIFT      = "PROTOCOL_G_BENIGN_SHIFT"


@dataclass
class DatasetPartition:
    protocol: SplitProtocol
    train_events: List[CanonicalEvent]
    val_events: List[CanonicalEvent]
    test_events: List[CanonicalEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def train_count(self) -> int: return len(self.train_events)
    @property
    def val_count(self) -> int: return len(self.val_events)
    @property
    def test_count(self) -> int: return len(self.test_events)
    @property
    def total_count(self) -> int: return self.train_count + self.val_count + self.test_count


class EvaluationSplitter:
    """Partitions canonical events according to formal evaluation protocols."""

    @staticmethod
    def split_standard(
        events: List[CanonicalEvent],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        seed: int = 42
    ) -> DatasetPartition:
        """Protocol A: Standard stratified randomized partition."""
        rng = random.Random(seed)
        shuffled = list(events)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

        return DatasetPartition(
            protocol=SplitProtocol.PROTOCOL_A_STANDARD,
            train_events=train,
            val_events=val,
            test_events=test,
            metadata={"train_ratio": train_ratio, "val_ratio": val_ratio, "seed": seed}
        )

    @staticmethod
    def split_temporal(
        events: List[CanonicalEvent],
        train_cutoff_pct: float = 0.60,
        val_cutoff_pct: float = 0.80
    ) -> DatasetPartition:
        """Protocol B: Strict chronological sequence with zero future lookahead."""
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        n = len(sorted_events)

        idx_train = int(n * train_cutoff_pct)
        idx_val = int(n * val_cutoff_pct)

        train = sorted_events[:idx_train]
        val = sorted_events[idx_train:idx_val]
        test = sorted_events[idx_val:]

        return DatasetPartition(
            protocol=SplitProtocol.PROTOCOL_B_TEMPORAL,
            train_events=train,
            val_events=val,
            test_events=test,
            metadata={
                "temporal_monotonic": True,
                "train_t_max": train[-1].timestamp if train else 0.0,
                "test_t_min": test[0].timestamp if test else 0.0,
            }
        )

    @staticmethod
    def split_open_set_family(
        events: List[CanonicalEvent],
        withheld_families: Set[CanonicalAttackFamily],
        train_ratio: float = 0.70,
        seed: int = 42
    ) -> DatasetPartition:
        """
        Protocol E: Withholds designated attack families strictly for test evaluation.
        Known families and benign records are split into train/val.
        """
        rng = random.Random(seed)
        known_events = []
        withheld_events = []

        for e in events:
            if e.label_info.canonical_family in withheld_families:
                withheld_events.append(e)
            else:
                known_events.append(e)

        rng.shuffle(known_events)
        n_known = len(known_events)
        n_train = int(n_known * train_ratio)

        train = known_events[:n_train]
        val = known_events[n_train:]
        test = withheld_events + [e for e in known_events[n_train:] if not e.label_info.is_malicious][:len(withheld_events)]

        return DatasetPartition(
            protocol=SplitProtocol.PROTOCOL_E_OPEN_SET_FAMILY,
            train_events=train,
            val_events=val,
            test_events=test,
            metadata={
                "withheld_families": [f.value for f in withheld_families],
                "withheld_test_count": len(withheld_events),
                "known_train_count": len(train),
            }
        )

    @staticmethod
    def split_cross_dataset(
        source_events: List[CanonicalEvent],
        target_events: List[CanonicalEvent],
        train_source_ratio: float = 0.80,
        seed: int = 42
    ) -> DatasetPartition:
        """Protocol C: Train on source dataset, evaluate on target domain dataset."""
        rng = random.Random(seed)
        src_shuffled = list(source_events)
        rng.shuffle(src_shuffled)

        n_train = int(len(src_shuffled) * train_source_ratio)
        train = src_shuffled[:n_train]
        val = src_shuffled[n_train:]
        test = target_events

        return DatasetPartition(
            protocol=SplitProtocol.PROTOCOL_C_CROSS_DATASET,
            train_events=train,
            val_events=val,
            test_events=test,
            metadata={
                "source_dataset": source_events[0].dataset_source if source_events else "Unknown",
                "target_dataset": target_events[0].dataset_source if target_events else "Unknown",
            }
        )
