from __future__ import annotations
"""
AHRAS Controlled Synthetic Mechanism Evaluation Runner
-------------------------------------------------------
Executes controlled synthetic mechanism experiments (E0–E12 matrix, 12 ablations,
and multi-stage incident response state-machine simulations).

Explicitly tagged: evaluation_type = "CONTROLLED_SYNTHETIC" / "SIMULATION".
Never conflates synthetic stress-testing with in-the-wild real traffic.
"""

import os
import sys
import json
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.research_experiments import run_all_research_evaluations
from evaluation.generate_research_tables import generate_all_tables
from evaluation.experiment_manifest import generate_experiment_manifest

if __name__ == "__main__":
    print("=======================================================================")
    print("   Running AHRAS Controlled Synthetic Mechanism Evaluation Suite")
    print("=======================================================================")
    rep = run_all_research_evaluations()
    generate_all_tables()
    manifest = generate_experiment_manifest()
    print("\n✓ Controlled Synthetic Mechanism Evaluation completed successfully.")
    print(f"✓ Manifest: {manifest['experiment_id']} (Seed: {manifest['rng_seed']})")
