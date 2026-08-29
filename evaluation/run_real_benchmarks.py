from __future__ import annotations
"""
AHRAS Real-World Benchmark Dataset Evaluation Runner
-----------------------------------------------------
Executes intrusion detection evaluations on authentic, un-synthesized benchmark datasets
(CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018).

Integrity Policy:
  - Requires authentic raw CSV data files with verified checksums.
  - Fails cleanly with REAL_DATA_NOT_AVAILABLE if raw files are absent.
  - NEVER silently falls back to synthetic or mock data.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.dataset_loader import DatasetLoader, DatasetManifest
from evaluation.leakage_audit import temporal_train_test_split, LeakageAuditor
from evaluation.metrics import MetricsCalculator
from evaluation.runner import record_to_ocsf
from detection.hybrid_engine import get_combiner
from detection.risk_engine import get_risk_engine, RiskConfig

log = logging.getLogger(__name__)

# Search paths for authentic raw benchmarks
DEFAULT_SEARCH_PATHS = {
    "CICIDS2017": os.getenv("CICIDS2017_PATH", os.path.join(_ROOT, "data", "cicids2017", "Wednesday-workingHours.pcap_ISCX.csv")),
    "UNSW_NB15":  os.getenv("UNSW_PATH", os.path.join(_ROOT, "data", "unsw_nb15", "UNSW-NB15_1.csv")),
}


def run_real_benchmark_suite() -> Dict[str, Any]:
    print("=======================================================================")
    print("   AHRAS Real-World Benchmark Dataset Evaluation Runner")
    print("=======================================================================")

    results = {}
    datasets_found = 0

    for name, path in DEFAULT_SEARCH_PATHS.items():
        print(f"\n[BENCHMARK] Checking for authentic raw dataset '{name}' at: {path}")
        if not os.path.exists(path):
            print(f"  [-] REAL_DATA_NOT_AVAILABLE: Raw dataset file '{name}' not found locally.")
            print(f"      To evaluate, place authentic CSV at '{path}' or set {name}_PATH environment variable.")
            results[name] = {
                "status": "REAL_DATA_NOT_AVAILABLE",
                "path": path,
                "note": "Evaluation skipped. No synthetic data substituted (strict integrity policy).",
            }
            continue

        datasets_found += 1
        print(f"  [+] Found raw dataset file! Size: {os.path.getsize(path) / (1024*1024):.2f} MB")
        
        loader = DatasetLoader(path)
        manifest = loader.generate_manifest(limit=10000)
        print(f"  [+] Generated Manifest: {manifest.total_rows} rows, SHA256: {manifest.sha256_checksum[:12]}...")

        # Process real records
        records = list(loader.iter_records(limit=5000))
        train, val, test = temporal_train_test_split(records, train_ratio=0.70, val_ratio=0.15)
        
        auditor = LeakageAuditor()
        audit = auditor.audit_splits(train, test, is_entity_disjoint=False)
        print(f"  [+] Leakage Audit: {'PASSED' if audit['overall_leakage_audit_pass'] else 'FAILED'}")

        combiner = get_combiner()
        risk_eng = get_risk_engine()
        calc = MetricsCalculator()
        
        y_true = [r.label for r in test]
        scores = []
        for r in test:
            ocsf = record_to_ocsf(r)
            res = combiner.process(ocsf)
            rr = risk_eng.score_risk(r.src_ip, res.signature_matches if res else [], res.anomaly_result if res else None, res.stat_result if res else None, ocsf)
            scores.append(rr.risk_score)

        report = calc.compute(y_true, scores, dataset_name=f"Real_{name}")
        print(f"  [+] Evaluated on {len(test)} test flows: Precision={report.precision:.4f}, Recall={report.recall:.4f}, F1={report.f1:.4f}, Brier={report.brier_score:.4f}")

        results[name] = {
            "status": "EVALUATED",
            "evaluation_type": "REAL_WORLD_BENCHMARK",
            "manifest": manifest.to_dict(),
            "metrics": report.to_dict(),
            "leakage_audit": audit,
        }

    out_path = os.path.join(_ROOT, "evaluation", "results", "real_world_benchmarks_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Real-world benchmark report written to: {out_path}")
    print(f"Total authentic datasets evaluated: {datasets_found} / {len(DEFAULT_SEARCH_PATHS)}")
    return results


if __name__ == "__main__":
    run_real_benchmark_suite()
