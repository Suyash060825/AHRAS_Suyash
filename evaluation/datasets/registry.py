from __future__ import annotations
"""
AHRAS Dataset Registry & Unified API
------------------------------------
Provides top-level access to verified datasets, manifests, adapters,
splitters, and loader engines.
"""

from typing import Dict, List, Optional, Any
from evaluation.datasets.manifest import DatasetManifest, DatasetEntry
from evaluation.datasets.verifier import DatasetVerifier, VerificationResult
from evaluation.datasets.downloader import DatasetDownloader
from evaluation.datasets.loader import DatasetLoader
from evaluation.datasets.splitters import EvaluationSplitter, SplitProtocol, DatasetPartition
from evaluation.datasets.adapters import CanonicalEvent, CanonicalAttackFamily, DatasetAdapter


class DatasetRegistry:
    """Master registry interface for all AHRAS evaluation datasets."""

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest = DatasetManifest(manifest_path)
        self.verifier = DatasetVerifier()
        self.downloader = DatasetDownloader(self.manifest)
        self.loader = DatasetLoader(self.manifest)
        self.splitter = EvaluationSplitter()

    def get_dataset(self, key: str) -> Optional[DatasetEntry]:
        return self.manifest.get_dataset(key)

    def list_datasets(self) -> List[DatasetEntry]:
        return self.manifest.list_datasets()

    def verify_all_datasets(self) -> Dict[str, VerificationResult]:
        results = {}
        for key, entry in self.manifest.datasets.items():
            if entry.local_path:
                results[key] = self.verifier.verify_file(
                    dataset_key=key,
                    filepath=entry.absolute_local_path or "",
                    expected_sha256=entry.sha256
                )
            else:
                results[key] = VerificationResult(
                    dataset_key=key,
                    is_valid=False,
                    status="BLOCKED",
                    computed_sha256=None,
                    expected_sha256=None,
                    file_size_bytes=0,
                    row_count_estimate=0,
                    error_message="Dataset not staged locally. Download required."
                )
        return results


# Global singleton
_global_registry: Optional[DatasetRegistry] = None

def get_dataset_registry() -> DatasetRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = DatasetRegistry()
    return _global_registry
