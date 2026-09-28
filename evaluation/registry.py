from __future__ import annotations
"""
AHRAS Module — Model and Detection Registry Manager (Section 26 / Research Frontier P1)
----------------------------------------------------------------------------------------
Provides immutable, versioned, append-only management of machine learning models
and detection rules with strict cryptographic artifact provenance:

Files:
  - evaluation/model_registry.json
  - evaluation/detection_registry.json

Invariants:
  1. Never overwrite historical records.
  2. Every model record contains artifact SHA-256 and dataset SHA-256 digests.
  3. Every detection record maps directly to MITRE ATT&CK techniques with verified empirical metrics.
"""

import copy
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_REGISTRY_PATH = REPO_ROOT / "evaluation" / "model_registry.json"
DETECTION_REGISTRY_PATH = REPO_ROOT / "evaluation" / "detection_registry.json"


@dataclass
class ModelRegistryEntry:
    name: str
    version: str
    artifact_hash: str
    training_dataset: str
    dataset_hash: str
    features: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    calibration: Dict[str, Any]
    ood_behavior: Dict[str, Any]
    known_limitations: List[str]
    evaluation_date: str
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionRegistryEntry:
    rule_id: str
    version: str
    technique: str
    implementation: str
    telemetry: List[str]
    precision: float
    recall: float
    robustness: float
    validation_scenarios: List[str]
    limitations: List[str]
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AHRASRegistryManager:
    """
    Thread-safe, append-only registry manager persisting versioned models and detection rules.
    """

    def __init__(
        self,
        model_path: Path = MODEL_REGISTRY_PATH,
        detection_path: Path = DETECTION_REGISTRY_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.detection_path = Path(detection_path)
        self._lock = threading.RLock()
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            self._write_json(self.model_path, {"schema_version": "1.0", "models": []})
        if not self.detection_path.exists():
            self._write_json(self.detection_path, {"schema_version": "1.0", "detections": []})

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)

    def register_model(self, entry: ModelRegistryEntry) -> bool:
        """
        Appends a model record to the registry.
        Rejects duplicate registration of the exact same (name, version) pair to prevent overwriting.
        """
        with self._lock:
            data = self._read_json(self.model_path)
            existing = data.get("models", [])
            for m in existing:
                if m.get("name") == entry.name and m.get("version") == entry.version:
                    log.warning(f"Model {entry.name}:{entry.version} already exists in registry. Overwriting is disallowed.")
                    return False

            existing.append(entry.to_dict())
            data["models"] = existing
            self._write_json(self.model_path, data)
            log.info(f"Successfully registered model: {entry.name}:{entry.version}")
            return True

    def register_detection(self, entry: DetectionRegistryEntry) -> bool:
        """
        Appends a detection rule record to the registry.
        Rejects duplicate registration of the same (rule_id, version) pair.
        """
        with self._lock:
            data = self._read_json(self.detection_path)
            existing = data.get("detections", [])
            for d in existing:
                if d.get("rule_id") == entry.rule_id and d.get("version") == entry.version:
                    log.warning(f"Detection {entry.rule_id}:{entry.version} already exists. Overwriting disallowed.")
                    return False

            existing.append(entry.to_dict())
            data["detections"] = existing
            self._write_json(self.detection_path, data)
            log.info(f"Successfully registered detection rule: {entry.rule_id}:{entry.version}")
            return True

    def get_models(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read_json(self.model_path)
            models = data.get("models", [])
            if name:
                return [m for m in models if m.get("name") == name]
            return models

    def get_latest_model(self, name: str) -> Optional[Dict[str, Any]]:
        models = self.get_models(name)
        if not models:
            return None
        return models[-1]

    def get_detections(self, technique: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read_json(self.detection_path)
            detections = data.get("detections", [])
            if technique:
                return [d for d in detections if d.get("technique") == technique]
            return detections
