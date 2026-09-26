from __future__ import annotations
"""
AHRAS Research Manifest Generator
---------------------------------
Provides strict reproducibility metadata tracking for all empirical evaluations:
  - experiment_id
  - code_version (git commit SHA)
  - dataset_version and dataset_hash (SHA-256)
  - configuration_hash (SHA-256)
  - random_seed
  - execution environment (Python version, OS, core dependency versions)
  - result_path
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_NON_GIT_RELEASE"


def get_git_branch() -> str:
    """Retrieves current Git branch name."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 digest of a local file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 digest of configuration dictionary."""
    encoded = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ResearchManifest:
    experiment_id: str
    code_version: str
    git_branch: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    configuration_hash: str
    configuration: Dict[str, Any]
    random_seed: int
    environment: Dict[str, Any]
    result_path: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_manifest(
    experiment_id: str,
    dataset_name: str,
    dataset_path: str,
    configuration: Dict[str, Any],
    result_path: str,
    random_seed: int = 42,
    dataset_version: str = "1.0",
) -> ResearchManifest:
    """Builds a verified ResearchManifest object with live system metadata."""
    import sklearn
    import numpy as np

    env_meta = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "dependencies": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        }
    }

    manifest = ResearchManifest(
        experiment_id=experiment_id,
        code_version=get_git_commit_hash(),
        git_branch=get_git_branch(),
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_hash=compute_file_sha256(dataset_path),
        configuration_hash=compute_dict_hash(configuration),
        configuration=configuration,
        random_seed=random_seed,
        environment=env_meta,
        result_path=result_path,
    )
    return manifest
