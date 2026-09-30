from __future__ import annotations
"""
AHRAS Dataset Loader
--------------------
Loads, cleans, validates, and adapts real-world dataset files into CanonicalEvent streams.
"""

import os
import csv
import logging
from typing import Iterator, List, Dict, Any, Optional
from evaluation.datasets.manifest import DatasetManifest, DatasetEntry
from evaluation.datasets.adapters import DatasetAdapter, CanonicalEvent
from evaluation.datasets.verifier import DatasetVerifier

log = logging.getLogger(__name__)


class DatasetLoader:
    """Streamed and in-memory loader for registered AHRAS datasets."""

    def __init__(self, manifest: Optional[DatasetManifest] = None):
        self.manifest = manifest or DatasetManifest()
        self.verifier = DatasetVerifier()

    def stream_dataset(
        self,
        dataset_key: str,
        limit: Optional[int] = None
    ) -> Iterator[CanonicalEvent]:
        entry = self.manifest.get_dataset(dataset_key)
        if not entry:
            raise FileNotFoundError(f"Dataset '{dataset_key}' not defined in manifest.")
        if not entry.is_available_locally:
            raise FileNotFoundError(
                f"[BLOCKED] Real dataset '{dataset_key}' ({entry.canonical_name}) is missing from disk. "
                f"Expected at: {entry.local_path}. Manual download required from: {entry.official_download_reference}"
            )

        filepath = entry.absolute_local_path
        # Verify integrity
        ver_res = self.verifier.verify_file(dataset_key, filepath, entry.sha256)
        if not ver_res.is_valid:
            log.warning(f"[DATASET INTEGRITY WARNING] {ver_res.error_message}")

        count = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            # Clean headers (strip leading/trailing whitespace)
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames] if reader.fieldnames else []

            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                if dataset_key == "cicids2017":
                    event = DatasetAdapter.adapt_cicids2017(clean_row, count)
                elif dataset_key == "unsw_nb15":
                    event = DatasetAdapter.adapt_unsw_nb15(clean_row, count)
                else:
                    raise NotImplementedError(f"Adapter for dataset '{dataset_key}' not yet implemented.")

                yield event
                count += 1
                if limit and count >= limit:
                    break

    def load_events(
        self,
        dataset_key: str,
        limit: Optional[int] = None
    ) -> List[CanonicalEvent]:
        """Loads canonical events into memory."""
        return list(self.stream_dataset(dataset_key, limit=limit))
