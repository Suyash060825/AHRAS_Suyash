import os
import re

file_path = "evaluation/run_real_benchmarks.py"
with open(file_path, "r") as f:
    content = f.read()

# Add imports
imports_to_add = """
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from detection.selective_gate import ConformalRiskGate
"""
content = content.replace("import numpy as np\n", "import numpy as np\n" + imports_to_add)


# Function to extract features
extract_func = """
def extract_feature_vector(rec: DatasetRecord) -> list:
    feats = rec.features
    return [
        feats.get("dst_port", feats.get("Destination Port", 80)),
        feats.get("packet_count", feats.get("Total Fwd Packets", 1) + feats.get("Total Backward Packets", 0)),
        feats.get("duration_sec", max(0.001, feats.get("Flow Duration", 1000.0) / 1_000_000.0)),
        feats.get("byte_count", feats.get("Total Length of Fwd Packets", 100)),
        float(feats.get("Flow Packets/s", 10.0)),
        feats.get("SYN Flag Count", feats.get("syn_count", 0)),
        feats.get("unique_dst_ports", 1)
    ]
"""
content = content.replace("def optimize_threshold_on_validation", extract_func + "\ndef optimize_threshold_on_validation")

# Inside run_benchmark_for_dataset
baseline_training = """
    # 4b. Train External Baselines on Train Partition
    print("  [+] Training External Baselines (RandomForest, GradientBoosting, IsolationForest)...")
    X_train = [extract_feature_vector(r) for r in train]
    y_train_list = [r.label for r in train]
    
    clf_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=seed)
    clf_gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=seed)
    clf_if = IsolationForest(n_estimators=50, contamination=0.1, random_state=seed)
    
    if len(set(y_train_list)) > 1:
        clf_rf.fit(X_train, y_train_list)
        clf_gb.fit(X_train, y_train_list)
        clf_if.fit(X_train)
    else:
        print("  [-] Warning: Only one class in training set, skipping baseline supervised training.")

"""

content = content.replace("train_latencies = []", baseline_training + "    train_latencies = []")


conformal_calibration = """
    tau_star, val_f1 = optimize_threshold_on_validation(y_val, s_val)
    calib_params = fit_probability_calibration(y_val, s_val)
    print(f"  [+] Validation Locked Parameters: tau*={tau_star:.3f} (Val F1={val_f1:.4f}) | Calib Slope={calib_params['calib_slope']}")

    # Conformal Calibration
    conformal_gate = ConformalRiskGate()
    conformal_tau = conformal_gate.calibrate(s_val, y_val)
    print(f"  [+] Conformal Gate Calibrated: tau*={conformal_tau:.4f}")
"""

content = content.replace("""    tau_star, val_f1 = optimize_threshold_on_validation(y_val, s_val)
    calib_params = fit_probability_calibration(y_val, s_val)
    print(f"  [+] Validation Locked Parameters: tau*={tau_star:.3f} (Val F1={val_f1:.4f}) | Calib Slope={calib_params['calib_slope']}")""", conformal_calibration)

baseline_eval = """
    # 6b. Evaluate External Baselines
    X_test = [extract_feature_vector(r) for r in test]
    y_test_arr = np.array(y_test)
    baseline_metrics = {}
    
    if len(set(y_train_list)) > 1:
        preds_rf = clf_rf.predict(X_test)
        preds_gb = clf_gb.predict(X_test)
        preds_if = clf_if.predict(X_test)
        preds_if = np.where(preds_if == -1, 1, 0)  # IF: -1 is anomaly (attack), 1 is normal
        
        def calc_f1(preds, y_true):
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            return (2 * p * r / (p + r)) if (p + r) > 0 else 0
            
        baseline_metrics["RandomForest_F1"] = round(calc_f1(preds_rf, y_test_arr), 4)
        baseline_metrics["GradientBoosting_F1"] = round(calc_f1(preds_gb, y_test_arr), 4)
        baseline_metrics["IsolationForest_F1"] = round(calc_f1(preds_if, y_test_arr), 4)
        print(f"  [+] External Baselines F1 -> RF: {baseline_metrics['RandomForest_F1']:.4f}, GB: {baseline_metrics['GradientBoosting_F1']:.4f}, IF: {baseline_metrics['IsolationForest_F1']:.4f}")

    # 7. Metrics Calculation & Confidence Intervals
"""

content = content.replace("    # 7. Metrics Calculation & Confidence Intervals", baseline_eval)

add_metrics = """
        "test_metrics": {
            "precision": test_report.precision,
            "recall": test_report.recall,
            "f1": test_report.f1,
            "f1_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "pr_auc": test_report.pr_auc,
            "roc_auc": test_report.auc,
            "fpr": test_report.false_positive_rate,
            "fnr": test_report.false_negative_rate,
            "balanced_accuracy": test_report.balanced_accuracy,
            "brier_score": test_report.brier_score,
            "ece": test_report.ece,
            "mean_latency_ms": test_report.mean_latency_ms,
            "p95_latency_ms": test_report.p95_latency_ms,
            "baseline_comparison": baseline_metrics,
            "conformal_tau": conformal_tau
        },
"""

content = re.sub(r'"test_metrics": \{.*?"p95_latency_ms": test_report.p95_latency_ms,?\n\s*\},', add_metrics, content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
