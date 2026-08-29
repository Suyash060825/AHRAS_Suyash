from __future__ import annotations
"""
AHRAS Experiment Truth Manifest Generator
------------------------------------------
Generates machine-readable provenance metadata for every formal experiment stage E0–E12:
  - Experiment ID & Evaluation Type
  - Active & Disabled Components
  - Dataset name & SHA-256 Hash
  - Split Policy & Random Seed
"""

import os
import sys
import json
import time
import hashlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_truth_manifest() -> dict:
    raw_dataset = os.path.join(_ROOT, "evaluation", "data", "synthetic_eval_dataset.csv")
    ds_hash = "FILE_NOT_FOUND"
    if os.path.exists(raw_dataset):
        h = hashlib.sha256()
        with open(raw_dataset, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        ds_hash = h.hexdigest()

    experiments = [
        {
            "experiment_id": "E0_Baseline",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Random uniform baseline detector",
            "enabled_components": [],
            "disabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E1_Signature_Only",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "23 Suricata/MITRE signature rules only",
            "enabled_components": ["signature"],
            "disabled_components": ["ml", "statistical", "trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E2_ML_Only",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "ML Anomaly Ensemble (Autoencoder + IF + SVM)",
            "enabled_components": ["ml"],
            "disabled_components": ["signature", "statistical", "trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E3_Statistical_Only",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Welford Z-score drift & Circadian probability",
            "enabled_components": ["statistical"],
            "disabled_components": ["signature", "ml", "trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E4_Fixed_Hybrid",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Static uncalibrated multi-detector fusion",
            "enabled_components": ["signature", "ml", "statistical"],
            "disabled_components": ["trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E5_Adaptive_Hybrid",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Online adaptive weight learning from training split",
            "enabled_components": ["signature", "ml", "statistical", "adaptive_weights"],
            "disabled_components": ["trust", "graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E6_Hybrid_Trust",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Multi-detector fusion with dynamic entity trust decay/recovery",
            "enabled_components": ["signature", "ml", "statistical", "adaptive_weights", "trust"],
            "disabled_components": ["graph", "uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E7_Hybrid_Graph",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Fusion + Temporal attack episode correlation graph",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph"],
            "disabled_components": ["uncertainty", "forecast", "response"],
        },
        {
            "experiment_id": "E8_Hybrid_Uncertainty",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Fusion + Detector-disagreement uncertainty dampening",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty"],
            "disabled_components": ["forecast", "response"],
        },
        {
            "experiment_id": "E9_Full_AHRAS_Risk",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Full multi-signal risk model with Asset Criticality & Threat Intel",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty", "threat_intel", "asset_criticality"],
            "disabled_components": ["forecast", "response"],
        },
        {
            "experiment_id": "E10_Full_AHRAS_Forecast",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Risk model + Causal walk-forward early warning forecaster",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty", "threat_intel", "asset_criticality", "forecast"],
            "disabled_components": ["response"],
        },
        {
            "experiment_id": "E11_Full_AHRAS_Policy",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Complete detection + Action Utility safety-gated response orchestrator",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty", "threat_intel", "asset_criticality", "forecast", "response"],
            "disabled_components": [],
        },
        {
            "experiment_id": "E12_Full_Closed_Loop_AHRAS",
            "evaluation_type": "CONTROLLED_SYNTHETIC",
            "description": "Unified closed-loop evidence-to-action controller",
            "enabled_components": ["signature", "ml", "statistical", "trust", "graph", "uncertainty", "threat_intel", "asset_criticality", "forecast", "response", "ledger"],
            "disabled_components": [],
        }
    ]

    manifest = {
        "manifest_version": "2.0-TRUTH-AUDITED",
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset_name": "synthetic_eval_dataset.csv",
        "dataset_sha256": ds_hash,
        "split_policy": "Temporal (70/15/15) + Entity-Disjoint",
        "rng_seed": 42,
        "experiments": experiments,
    }

    out_path = os.path.join(_ROOT, "evaluation", "results", "experiment_truth_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    m = generate_truth_manifest()
    print("Truth Manifest written to evaluation/results/experiment_truth_manifest.json")
