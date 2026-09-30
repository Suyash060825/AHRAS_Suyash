from __future__ import annotations
"""
AHRAS Dataset Manifest Schema & Parser
--------------------------------------
Parses data/manifests/evaluation_datasets.yaml and exposes typed representations
of dataset provenance, verification metadata, and research roles.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    dataset_id: str
    canonical_name: str
    official_source: str
    official_download_reference: str
    version: str
    release_date: Optional[str]
    license: str
    data_type: str
    telemetry_modalities: List[str]
    attack_families: List[str]
    benign_data: str
    temporal_information: str
    labels: List[str]
    raw_format: str
    derived_format: str
    expected_files: List[str]
    local_path: Optional[str]
    sha256: Optional[str]
    preprocessing_version: str
    source_notes: str
    ahras_usage: str
    evaluation_role: str
    primary_research_question: str

    @property
    def is_available_locally(self) -> bool:
        if not self.local_path:
            return False
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(root, self.local_path)
        return os.path.exists(full_path)

    @property
    def absolute_local_path(self) -> Optional[str]:
        if not self.local_path:
            return None
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(root, self.local_path)


class DatasetManifest:
    """Loads and validates dataset manifest definitions."""

    def __init__(self, manifest_path: Optional[str] = None):
        if manifest_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            manifest_path = os.path.join(root, "data", "manifests", "evaluation_datasets.yaml")
        self.manifest_path = manifest_path
        self.version = "unknown"
        self.datasets: Dict[str, DatasetEntry] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.manifest_path):
            log.warning(f"Manifest file not found at {self.manifest_path}")
            return
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.version = data.get("version", "1.0.0")
        for key, entry_data in data.get("datasets", {}).items():
            self.datasets[key] = DatasetEntry(**entry_data)

    def get_dataset(self, key: str) -> Optional[DatasetEntry]:
        return self.datasets.get(key)

    def list_datasets(self) -> List[DatasetEntry]:
        return list(self.datasets.values())

    def list_available_datasets(self) -> List[DatasetEntry]:
        return [d for d in self.datasets.values() if d.is_available_locally]
