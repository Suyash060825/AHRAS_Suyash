#!/usr/bin/env python3
"""
AHRAS Unified Quality Gate & CI/CD Pre-Commit Verification Tool.
Enforces strict scientific invariants, cryptographic provenance, security boundaries,
and test suite cleanliness.
"""

import sys
import os
import subprocess

def run_step(name: str, cmd: list) -> bool:
    print(f"[*] Running Quality Gate: {name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"    [+] PASSED: {name}")
        return True
    else:
        print(f"    [-] FAILED: {name}")
        print(res.stdout)
        print(res.stderr)
        return False

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("=======================================================")
    print("           AHRAS UNIFIED QUALITY GATE (CI/CD)          ")
    print("=======================================================")

    all_ok = True

    # 1. Dataset Integrity & Provenance Check
    all_ok &= run_step(
        "Dataset Integrity & Cryptographic Registry",
        [sys.executable, "-c", "from evaluation.datasets.registry import DatasetRegistry; r = DatasetRegistry(); res = r.verify_all_datasets(); assert len([k for k,v in res.items() if v.is_valid]) == 2; print('Datasets verified.')"]
    )

    # 2. XAI Replay & DecisionTrace Reconstruction
    all_ok &= run_step(
        "XAI DecisionTrace 10-Path Computational Fidelity",
        [sys.executable, "-m", "pytest", "tests/test_xai_fidelity.py", "-q"]
    )

    # 3. Response Safety & Blast Radius Barrier
    all_ok &= run_step(
        "Response Invariants & Critical Asset Safety",
        [sys.executable, "-m", "pytest", "tests/test_safe_response.py", "-q"]
    )

    # 4. Graph Ground Truth Evaluation
    all_ok &= run_step(
        "Graph Ground Truth Evaluation",
        [sys.executable, "evaluation/experiments/graph_ground_truth.py"]
    )

    # 5. Full Pytest Suite Cleanliness
    all_ok &= run_step(
        "Complete Pytest Regression Suite",
        [sys.executable, "-m", "pytest", "-q"]
    )

    print("=======================================================")
    if all_ok:
        print("    STATUS: ALL QUALITY GATES PASSED (100% CLEAN)")
        print("=======================================================")
        sys.exit(0)
    else:
        print("    STATUS: QUALITY GATE FAILURE DETECTED")
        print("=======================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
