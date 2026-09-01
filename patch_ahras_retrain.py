import re

file_path = "evaluation/run_real_benchmarks.py"
with open(file_path, "r") as f:
    content = f.read()

imports = """
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from detection.selective_gate import ConformalRiskGate
from normalizer.feature_extractor import extract as ahras_extract
from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic
"""
content = re.sub(r'from sklearn\.ensemble.*?from detection\.selective_gate import ConformalRiskGate', imports.strip(), content, flags=re.DOTALL)

retrain_block = """
        print("  [-] Warning: Only one class in training set, skipping baseline supervised training.")

    # 4c. Retrain AHRAS Anomaly Engine on Authentic Train Partition
    print("  [+] Dynamically Retraining AHRAS Anomaly Models on Authentic Data...")
    normal_ocsf_vecs = []
    for tr in train:
        if tr.label == 0:
            ocsf_evt = record_to_ocsf(tr)
            vec = ahras_extract(ocsf_evt)
            if vec is not None:
                normal_ocsf_vecs.append(vec)
                
    if normal_ocsf_vecs:
        bootstrap_with_normal_traffic("network_activity", normal_ocsf_vecs)
    else:
        print("  [-] Warning: No normal traffic found to train AHRAS Anomaly Engine.")
"""
content = content.replace('        print("  [-] Warning: Only one class in training set, skipping baseline supervised training.")\n', retrain_block)

with open(file_path, "w") as f:
    f.write(content)
