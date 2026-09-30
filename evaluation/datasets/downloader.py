from __future__ import annotations
"""
AHRAS Official Dataset Downloader & Reference Manager
-----------------------------------------------------
Provides reproducible downloading routines from official academic sources
and enforces fail-closed BLOCKED status when mandatory real datasets are absent.
"""

import os
import logging
from typing import Dict, Any, Optional
from evaluation.datasets.manifest import DatasetManifest, DatasetEntry

log = logging.getLogger(__name__)


class DatasetDownloader:
    """Manages official dataset download references and local staging."""

    def __init__(self, manifest: Optional[DatasetManifest] = None):
        self.manifest = manifest or DatasetManifest()

    def get_download_instructions(self, dataset_key: str) -> Dict[str, Any]:
        entry = self.manifest.get_dataset(dataset_key)
        if not entry:
            return {"status": "NOT_FOUND", "message": f"Dataset '{dataset_key}' not defined in manifest."}

        return {
            "dataset_id": entry.dataset_id,
            "canonical_name": entry.canonical_name,
            "official_source": entry.official_source,
            "download_url": entry.official_download_reference,
            "license": entry.license,
            "expected_files": entry.expected_files,
            "staging_path": entry.local_path,
            "is_available": entry.is_available_locally,
            "status": "AVAILABLE" if entry.is_available_locally else "BLOCKED_MANUAL_DOWNLOAD_REQUIRED",
            "instructions": (
                f"To obtain {entry.canonical_name}, download from official reference: "
                f"{entry.official_download_reference} and place into {entry.local_path}."
            )
        }
