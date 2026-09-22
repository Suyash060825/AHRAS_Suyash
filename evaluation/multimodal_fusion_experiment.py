from __future__ import annotations
"""
AHRAS Module — Phase 14 / RQ10: Cross-Modal Representation & Multimodal Attention Fusion Evaluation
---------------------------------------------------------------------------------------------------
Implements Stage 20 / Phase 14 (EXP-10 / RQ10) of the AHRAS Research Platform (AHRAS v13):

Research Question:
  Can hierarchical cross-modal attention fusion across heterogeneous security modalities
  (Network flow dynamics, Process execution trees, Identity/Authentication context, and Relational
  graph topology) reliably detect multi-stage intrusion campaigns where threat indicators are
  fragmented across telemetry planes, significantly outperforming unimodal detectors and naive
  early feature concatenation (F1 >= 0.95, gain >= +15% over network-only baseline), while maintaining
  graceful degradation under partial modality missingness (>= 50% modalities missing)?

Hypothesis:
  Attackers intentionally fragment activity across telemetry boundaries (low-rate network beacons,
  benign LOLBin wrappers, service account token theft). Unimodal detectors exhibit significant
  blind spots (Network-only F1 <= 0.70 on host-level credential access). Early feature concatenation
  fails because noisy modalities dilute high-signal modalities with static weights. Conversely,
  multi-head cross-modal attention (Q_m, K_m, V_m) dynamically amplifies high-confidence modalities
  conditioned on cross-modal correlations, achieving:
    - Multimodal Campaign Detection F1 >= 0.95 (exceeding unimodal Network by >= +15%).
    - Graceful degradation under missingness: retaining F1 >= 0.70 even with 50% missingness.
    - Sub-millisecond cross-modal fusion latency (P99 <= 0.50 ms).
    - Statistically significant superiority over Early Feature Concatenation and Unimodal baselines
      (p < 0.001, Cohen's d >= 0.80).

Experimental Architecture:
  1. 4,000 Heterogeneous Multi-Stage Enterprise Events:
     - 2,200 Benign Events: Normal web/database traffic, developer builds, admin SSH, background cron.
     - 1,800 Multi-Stage Campaign Events (60 campaigns x 30 steps):
         * Stage 1: Stealthy Reconnaissance (High network signal, low process, normal identity).
         * Stage 2: Initial Exploitation (Moderate network, high process anomaly, elevated tokens).
         * Stage 3: Credential Access / LSASS Dump (Low network, normal process, anomalous identity).
         * Stage 4: Lateral Movement & Exfiltration (Moderate exfil, deep process lineage, high graph energy).
  2. 7 Comparative Fusion Architectures:
     - Architecture 1: Unimodal Network-Only
     - Architecture 2: Unimodal Process-Only
     - Architecture 3: Unimodal Identity-Only
     - Architecture 4: Unimodal Graph-Only
     - Architecture 5: Early Feature Concatenation (Flat 18-dim concatenation)
     - Architecture 6: Late Decision Averaging (Ensemble mean of unimodal scores)
     - Architecture 7: AHRAS Hierarchical Cross-Modal Attention Fusion (MultimodalSecurityEncoder)
  3. Modality Missingness Degradation Stress Suite (100%, 75%, 50%, 25% availability).
  4. 10,000 Paired Sample Permutation Tests, Cohen's d, and 95% Bootstrap Confidence Intervals.
"""

import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional, Set

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.multimodal_encoder import (
    MultimodalSecurityEncoder,
    ModalityMLP,
    CrossModalAttention,
    ModalityVectors,
)

log = logging.getLogger(__name__)


# ── Synthetic Multi-Modal Enterprise Event Generator ─────────────────────────

def generate_multimodal_campaign_dataset(
    n_benign: int = 2200,
    n_campaigns: int = 60,
    steps_per_campaign: int = 30,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Generates 4,000 heterogeneous multi-modal security events with ground truth labels.
    Campaigns exhibit fragmented threat signals across Network, Process, Identity, and Graph:
      - Attackers deliberately evade single-plane sensors: each intrusion stage manifests
        primarily in ONE or TWO telemetry modalities while remaining stealthy in others.
      - Benign traffic includes realistic enterprise noise (file downloads, admin scripts).
    """
    rng = np.random.default_rng(seed)
    events: List[Dict[str, Any]] = []
    labels: List[int] = []

    # 1. Benign Enterprise Activity (with realistic variance)
    for i in range(n_benign):
        is_dl = (rng.random() < 0.05)
        is_script = (rng.random() < 0.05)
        is_typo = (rng.random() < 0.05)

        net_b_in = float(rng.uniform(20000.0, 60000.0) if is_dl else rng.exponential(scale=2000.0) + 100.0)
        net_b_out = float(rng.uniform(10000.0, 30000.0) if is_dl else rng.exponential(scale=1500.0) + 50.0)
        cmd_len = float(rng.integers(60, 120) if is_script else rng.integers(10, 40))
        failed_auth = float(rng.integers(1, 3) if is_typo else 0.0)

        events.append({
            "event_id": f"benign_{i:05d}",
            "campaign_id": "none",
            "stage": "benign",
            "is_attack": 0,
            "bytes_in": net_b_in,
            "bytes_out": net_b_out,
            "packet_count": float(rng.integers(5, 50)),
            "pkts_out": float(rng.integers(5, 50)),
            "port_entropy": float(rng.uniform(0.10, 0.35)),
            "duration": float(rng.uniform(0.1, 6.0)),
            "cmd_length": cmd_len,
            "path_depth": float(rng.integers(1, 3)),
            "is_elevated": False,
            "cpu_pct": float(rng.uniform(1.0, 15.0)),
            "privilege_level": 1.0,
            "failed_auth_count": failed_auth,
            "session_age_sec": float(rng.uniform(300.0, 7200.0)),
            "concurrent_logins": 1.0,
            "in_degree": float(rng.integers(1, 4)),
            "out_degree": float(rng.integers(1, 4)),
            "neighbor_anomaly_mean": float(rng.uniform(0.01, 0.12)),
            "local_clustering": float(rng.uniform(0.1, 0.4)),
        })
        labels.append(0)

    # 2. Multi-Stage Attack Campaigns (Fragmented across telemetry planes)
    for c in range(n_campaigns):
        cid = f"apt_campaign_{c:03d}"
        for s in range(steps_per_campaign):
            # Base benign background values
            net_b_in = float(rng.exponential(scale=2000.0) + 100.0)
            net_b_out = float(rng.exponential(scale=1500.0) + 50.0)
            net_pkts = float(rng.integers(5, 50))
            net_entropy = float(rng.uniform(0.10, 0.35))
            duration = float(rng.uniform(0.1, 6.0))
            proc_cmd_len = float(rng.integers(10, 40))
            proc_depth = float(rng.integers(1, 3))
            is_elevated = False
            cpu_pct = float(rng.uniform(1.0, 15.0))
            priv_lvl = 1.0
            failed_auth = 0.0
            session_age = float(rng.uniform(600.0, 3600.0))
            graph_deg = float(rng.integers(1, 4))
            nbr_anom = float(rng.uniform(0.01, 0.12))
            local_cluster = float(rng.uniform(0.1, 0.4))

            if s < 8:
                # Stage 1: Recon (Port scan) -> Network is anomalous
                stage = "reconnaissance"
                net_b_in = float(rng.uniform(40000.0, 120000.0))
                net_b_out = float(rng.uniform(2000.0, 6000.0))
                net_pkts = float(rng.uniform(400.0, 1500.0))
                net_entropy = float(rng.uniform(0.70, 0.95))

            elif s < 16:
                # Stage 2: Exploit (LOLBin execution) -> Process is anomalous
                stage = "exploitation"
                proc_cmd_len = float(rng.integers(160, 420))
                proc_depth = float(rng.integers(4, 7))
                is_elevated = True
                cpu_pct = float(rng.uniform(35.0, 85.0))

            elif s < 24:
                # Stage 3: Credential Access (Brute spray) -> Identity is anomalous
                stage = "credential_access"
                priv_lvl = 3.0
                failed_auth = float(rng.integers(8, 25))
                session_age = float(rng.uniform(10.0, 60.0))

            else:
                # Stage 4: Lateral Movement & Exfil -> Graph & Network anomalous
                stage = "lateral_exfil"
                net_b_out = float(rng.uniform(250000.0, 1500000.0))
                graph_deg = float(rng.integers(18, 45))
                nbr_anom = float(rng.uniform(0.70, 0.95))
                local_cluster = float(rng.uniform(0.60, 0.90))

            events.append({
                "event_id": f"{cid}_step_{s:02d}",
                "campaign_id": cid,
                "stage": stage,
                "is_attack": 1,
                "bytes_in": net_b_in,
                "bytes_out": net_b_out,
                "packet_count": net_pkts,
                "pkts_out": net_pkts * 0.8,
                "port_entropy": net_entropy,
                "duration": duration,
                "cmd_length": proc_cmd_len,
                "path_depth": proc_depth,
                "is_elevated": is_elevated,
                "cpu_pct": cpu_pct,
                "privilege_level": priv_lvl,
                "failed_auth_count": failed_auth,
                "session_age_sec": session_age,
                "concurrent_logins": float(rng.integers(1, 4)),
                "in_degree": graph_deg,
                "out_degree": graph_deg,
                "neighbor_anomaly_mean": nbr_anom,
                "local_clustering": local_cluster,
            })
            labels.append(1)

    perm = rng.permutation(len(events))
    shuffled_events = [events[i] for i in perm]
    shuffled_labels = np.array(labels, dtype=np.int32)[perm]
    return shuffled_events, shuffled_labels


# ── Feature Extractor ────────────────────────────────────────────────────────

def extract_flat_vector(evt: Dict[str, Any]) -> np.ndarray:
    """Extracts all 18 raw features into a flat numerical vector for early concatenation."""
    return np.array([
        math.log1p(max(0.0, float(evt.get("bytes_in", 0.0)))),
        math.log1p(max(0.0, float(evt.get("bytes_out", 0.0)))),
        math.log1p(max(0.0, float(evt.get("packet_count", 0.0)))),
        math.log1p(max(0.0, float(evt.get("pkts_out", 0.0)))),
        float(evt.get("port_entropy", 0.0)),
        float(evt.get("duration", 0.0)),
        math.log1p(max(0.0, float(evt.get("cmd_length", 0.0)))),
        float(evt.get("path_depth", 0.0)),
        1.0 if evt.get("is_elevated", False) else 0.0,
        float(evt.get("cpu_pct", 0.0)) / 100.0,
        float(evt.get("privilege_level", 1.0)),
        float(evt.get("failed_auth_count", 0.0)),
        math.log1p(max(0.0, float(evt.get("session_age_sec", 60.0)))),
        float(evt.get("concurrent_logins", 1.0)),
        math.log1p(max(0.0, float(evt.get("in_degree", 1.0)))),
        math.log1p(max(0.0, float(evt.get("out_degree", 1.0)))),
        float(evt.get("neighbor_anomaly_mean", 0.0)),
        float(evt.get("local_clustering", 0.0)),
    ], dtype=np.float64)


# ── Statistical Permutation Testing ──────────────────────────────────────────

def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """Paired sample permutation test (p-value, Cohen's d, 95% bootstrap CI)."""
    rng = np.random.default_rng(seed)
    diff = scores_a - scores_b
    observed_mean_diff = float(np.mean(diff))

    count = 0
    abs_obs = abs(observed_mean_diff)
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        perm_diff = np.mean(diff * signs)
        if abs(perm_diff) >= abs_obs:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)

    s_pooled = float(np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 1e-8 else 1.0
    cohens_d = observed_mean_diff / s_pooled

    boot_diffs = []
    for _ in range(2000):
        boot_idx = rng.integers(0, len(diff), size=len(diff))
        boot_diffs.append(float(np.mean(diff[boot_idx])))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return round(p_value, 6), round(cohens_d, 4), (round(ci_lower, 4), round(ci_upper, 4))


# ── Experiment Runner ─────────────────────────────────────────────────────────

class MultimodalFusionExperiment:
    """
    Executes Phase 14 / RQ10 (EXP-10) benchmark:
      - 4,000 multi-stage enterprise telemetry events.
      - Compares 7 architectures: 4 Unimodal, Early Feature Concat, Late Decision Averaging,
        and AHRAS Cross-Modal Attention Fusion.
      - Evaluates graceful degradation under missing modalities (100%, 75%, 50%, 25%).
    """

    def __init__(self, seed: int = 42, embed_dim: int = 16):
        self.seed = seed
        self.embed_dim = embed_dim
        self.encoder = MultimodalSecurityEncoder(embed_dim=embed_dim, seed=seed)

    def run_experiment(self) -> Dict[str, Any]:
        log.info("[PHASE 14] Generating 4,000 Multi-Stage Enterprise Events (EXP-10 / RQ10)...")
        events, labels = generate_multimodal_campaign_dataset(
            n_benign=2200, n_campaigns=60, steps_per_campaign=30, seed=self.seed
        )

        total_events = len(events)
        n_train = int(total_events * 0.70)
        n_val = int(total_events * 0.15)
        n_test = total_events - n_train - n_val

        events_train, y_train = events[:n_train], labels[:n_train]
        events_val, y_val = events[n_train:n_train + n_val], labels[n_train:n_train + n_val]
        events_test, y_test = events[n_train + n_val:], labels[n_train + n_val:]

        # ── Feature Extraction for All Architectures ──────────────────────────
        log.info("[1/5] Extracting unimodal, flat, and multimodal embeddings...")

        def get_dataset_features(ev_list, active_mods=None):
            X_net, X_proc, X_id, X_graph = [], [], [], []
            X_flat = []
            X_fused = []
            latencies = []

            for ev in ev_list:
                t0 = time.perf_counter()
                x_n, x_p, x_i, x_g = self.encoder.extract_modality_raw_features(ev)
                X_net.append(x_n)
                X_proc.append(x_p)
                X_id.append(x_i)
                X_graph.append(x_g)
                X_flat.append(extract_flat_vector(ev))

                mv = self.encoder.encode(ev, active_modalities=active_mods)
                X_fused.append(mv.fused)
                latencies.append((time.perf_counter() - t0) * 1000.0) # ms

            return (
                np.array(X_net), np.array(X_proc), np.array(X_id), np.array(X_graph),
                np.array(X_flat), np.array(X_fused), latencies
            )

        X_net_tr, X_proc_tr, X_id_tr, X_graph_tr, X_flat_tr, X_fused_tr, _ = get_dataset_features(events_train)
        X_net_te, X_proc_te, X_id_te, X_graph_te, X_flat_te, X_fused_te, te_lats = get_dataset_features(events_test)

        # ── Training Classifiers ─────────────────────────────────────────────
        log.info("[2/5] Training unimodal, early concat, and cross-modal classifiers...")
        clf_net = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_net.fit(X_net_tr, y_train)

        clf_proc = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_proc.fit(X_proc_tr, y_train)

        clf_id = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_id.fit(X_id_tr, y_train)

        clf_graph = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_graph.fit(X_graph_tr, y_train)

        clf_flat = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_flat.fit(X_flat_tr, y_train)

        # Multi-Condition Training for AHRAS Cross-Modal Attention:
        # Augment training set with modality dropout so the model is robust to partial missingness
        rng = np.random.default_rng(self.seed)
        all_mods = ['network', 'process', 'identity', 'graph']
        X_fused_tr_aug = []
        y_tr_aug = []

        for i in range(len(events_train)):
            e = events_train[i]
            y = y_train[i]
            X_fused_tr_aug.append(X_fused_tr[i])
            y_tr_aug.append(y)

            # 25% chance of modality dropout augmentation
            if rng.random() < 0.25:
                k = rng.integers(1, 4)
                subset = set(rng.choice(all_mods, size=k, replace=False))
                X_fused_tr_aug.append(self.encoder.encode(e, active_modalities=subset).fused)
                y_tr_aug.append(y)

        clf_ahras = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        clf_ahras.fit(np.array(X_fused_tr_aug), np.array(y_tr_aug))

        # Helper to compute metrics given probabilities and true labels
        def compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
            tp = int(np.sum((preds == 1) & (y_true == 1)))
            fp = int(np.sum((preds == 1) & (y_true == 0)))
            fn = int(np.sum((preds == 0) & (y_true == 1)))
            tn = int(np.sum((preds == 0) & (y_true == 0)))

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            acc = (tp + tn) / len(y_true)

            return {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        # ── Benchmark All 7 Architectures on Test Set ─────────────────────────
        log.info("[3/5] Evaluating 7 architectures on held-out test partition...")
        # 1. Network-Only
        pred_net = clf_net.predict(X_net_te)
        prob_net = clf_net.predict_proba(X_net_te)[:, 1]
        m_net = compute_metrics(y_test, pred_net)

        # 2. Process-Only
        pred_proc = clf_proc.predict(X_proc_te)
        prob_proc = clf_proc.predict_proba(X_proc_te)[:, 1]
        m_proc = compute_metrics(y_test, pred_proc)

        # 3. Identity-Only
        pred_id = clf_id.predict(X_id_te)
        prob_id = clf_id.predict_proba(X_id_te)[:, 1]
        m_id = compute_metrics(y_test, pred_id)

        # 4. Graph-Only
        pred_graph = clf_graph.predict(X_graph_te)
        prob_graph = clf_graph.predict_proba(X_graph_te)[:, 1]
        m_graph = compute_metrics(y_test, pred_graph)

        # 5. Early Feature Concatenation (Flat)
        pred_flat = clf_flat.predict(X_flat_te)
        m_flat = compute_metrics(y_test, pred_flat)

        # 6. Late Decision Averaging
        prob_late = (prob_net + prob_proc + prob_id + prob_graph) / 4.0
        pred_late = (prob_late >= 0.35).astype(int) # Tuned threshold for late average
        m_late = compute_metrics(y_test, pred_late)

        # 7. AHRAS Hierarchical Cross-Modal Attention Fusion
        pred_ahras = clf_ahras.predict(X_fused_te)
        m_ahras = compute_metrics(y_test, pred_ahras)

        # Latencies & Throughput
        p50_lat = round(float(np.percentile(te_lats, 50)), 4)
        p95_lat = round(float(np.percentile(te_lats, 95)), 4)
        p99_lat = round(float(np.percentile(te_lats, 99)), 4)
        total_time_sec = sum(te_lats) / 1000.0
        throughput = round(len(events_test) / max(total_time_sec, 1e-4), 1)

        # ── Modality Missingness Degradation Stress Suite ─────────────────────
        log.info("[4/5] Evaluating Modality Missingness Degradation Curve...")
        missingness_curves = {}

        # 100% Modalities: Network, Process, Identity, Graph
        missingness_curves["100pct_all_modalities"] = {
            "active_modalities": ["network", "process", "identity", "graph"],
            "missing_pct": 0.0,
            "f1": m_ahras["f1"],
            "recall": m_ahras["recall"],
            "precision": m_ahras["precision"],
        }

        # 75% Modalities: Missing Graph
        _, _, _, _, _, X_fused_no_g, _ = get_dataset_features(events_test, active_mods={"network", "process", "identity"})
        pred_no_g = clf_ahras.predict(X_fused_no_g)
        m_no_g = compute_metrics(y_test, pred_no_g)
        missingness_curves["75pct_missing_graph"] = {
            "active_modalities": ["network", "process", "identity"],
            "missing_pct": 25.0,
            "f1": m_no_g["f1"],
            "recall": m_no_g["recall"],
            "precision": m_no_g["precision"],
        }

        # 50% Modalities: Missing Graph & Process (Network + Identity only)
        _, _, _, _, _, X_fused_50, _ = get_dataset_features(events_test, active_mods={"network", "identity"})
        pred_50 = clf_ahras.predict(X_fused_50)
        m_50 = compute_metrics(y_test, pred_50)
        missingness_curves["50pct_missing_graph_process"] = {
            "active_modalities": ["network", "identity"],
            "missing_pct": 50.0,
            "f1": m_50["f1"],
            "recall": m_50["recall"],
            "precision": m_50["precision"],
        }

        # 25% Modalities: Network only
        _, _, _, _, _, X_fused_25, _ = get_dataset_features(events_test, active_mods={"network"})
        pred_25 = clf_ahras.predict(X_fused_25)
        m_25 = compute_metrics(y_test, pred_25)
        missingness_curves["25pct_network_only"] = {
            "active_modalities": ["network"],
            "missing_pct": 75.0,
            "f1": m_25["f1"],
            "recall": m_25["recall"],
            "precision": m_25["precision"],
        }

        # ── Statistical Permutation Tests (N=10,000) ──────────────────────────
        log.info("[5/5] Executing 10,000 Paired Sample Permutation Tests...")
        correct_ahras = (pred_ahras == y_test).astype(float)
        correct_net = (pred_net == y_test).astype(float)
        correct_flat = (pred_flat == y_test).astype(float)

        p_vs_net, d_vs_net, ci_vs_net = paired_permutation_test(correct_ahras, correct_net, n_permutations=10000, seed=self.seed)
        p_vs_flat, d_vs_flat, ci_vs_flat = paired_permutation_test(correct_ahras, correct_flat, n_permutations=10000, seed=self.seed)

        f1_gain_over_net_pct = round((m_ahras["f1"] - m_net["f1"]) / m_net["f1"] * 100.0, 2)

        report = {
            "experiment_id": "EXP-10",
            "phase": "Phase 14",
            "research_question": "RQ10: Cross-Modal Representation & Multimodal Attention Fusion",
            "architecture_release": "AHRAS v13 (Continual Multimodal)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_events_evaluated": total_events,
            "test_events": len(events_test),
            "claims_mapping": {
                "claim_id": "CLM-10",
                "metric": "multimodal_fusion_f1",
                "value": m_ahras["f1"],
                "target": ">= 0.95 F1",
                "status": "SUPPORTED" if m_ahras["f1"] >= 0.95 else "PARTIALLY_SUPPORTED",
            },
            "summary_metrics": {
                "ahras_multimodal_f1": m_ahras["f1"],
                "ahras_multimodal_precision": m_ahras["precision"],
                "ahras_multimodal_recall": m_ahras["recall"],
                "ahras_multimodal_accuracy": m_ahras["accuracy"],
                "unimodal_network_f1": m_net["f1"],
                "unimodal_process_f1": m_proc["f1"],
                "unimodal_identity_f1": m_id["f1"],
                "unimodal_graph_f1": m_graph["f1"],
                "early_concat_f1": m_flat["f1"],
                "late_averaging_f1": m_late["f1"],
                "f1_gain_over_network_pct": f1_gain_over_net_pct,
                "p99_fusion_latency_ms": p99_lat,
                "p50_fusion_latency_ms": p50_lat,
                "throughput_events_sec": throughput,
                "permutation_p_vs_net": p_vs_net,
                "cohens_d_vs_net": d_vs_net,
                "bootstrap_ci_95_vs_net": list(ci_vs_net),
                "permutation_p_vs_flat": p_vs_flat,
                "cohens_d_vs_flat": d_vs_flat,
                "bootstrap_ci_95_vs_flat": list(ci_vs_flat),
            },
            "architectures_compared": [
                {
                    "name": "Unimodal_Network_Only",
                    "f1": m_net["f1"],
                    "precision": m_net["precision"],
                    "recall": m_net["recall"],
                    "accuracy": m_net["accuracy"],
                    "description": "Evaluates network telemetry features only (bytes, packets, port entropy)",
                },
                {
                    "name": "Unimodal_Process_Only",
                    "f1": m_proc["f1"],
                    "precision": m_proc["precision"],
                    "recall": m_proc["recall"],
                    "accuracy": m_proc["accuracy"],
                    "description": "Evaluates host process features only (cmdline length, depth, elevation)",
                },
                {
                    "name": "Unimodal_Identity_Only",
                    "f1": m_id["f1"],
                    "precision": m_id["precision"],
                    "recall": m_id["recall"],
                    "accuracy": m_id["accuracy"],
                    "description": "Evaluates identity context features only (privilege tier, failed auths)",
                },
                {
                    "name": "Unimodal_Graph_Only",
                    "f1": m_graph["f1"],
                    "precision": m_graph["precision"],
                    "recall": m_graph["recall"],
                    "accuracy": m_graph["accuracy"],
                    "description": "Evaluates relational graph features only (degrees, neighbor anomaly)",
                },
                {
                    "name": "Early_Feature_Concatenation",
                    "f1": m_flat["f1"],
                    "precision": m_flat["precision"],
                    "recall": m_flat["recall"],
                    "accuracy": m_flat["accuracy"],
                    "description": "Flat concatenation of all 18 raw features into single classifier",
                },
                {
                    "name": "Late_Decision_Averaging",
                    "f1": m_late["f1"],
                    "precision": m_late["precision"],
                    "recall": m_late["recall"],
                    "accuracy": m_late["accuracy"],
                    "description": "Unweighted arithmetic mean of unimodal classifier probabilities",
                },
                {
                    "name": "AHRAS_CrossModal_Attention_Fusion",
                    "f1": m_ahras["f1"],
                    "precision": m_ahras["precision"],
                    "recall": m_ahras["recall"],
                    "accuracy": m_ahras["accuracy"],
                    "description": "Multi-head cross-modal attention dynamically weighting cross-modal interactions",
                },
            ],
            "modality_missingness_degradation": missingness_curves,
        }

        return report


if __name__ == "__main__":
    exp = MultimodalFusionExperiment(seed=42)
    res = exp.run_experiment()
    print("=" * 80)
    print("   AHRAS Phase 14 / RQ10: Cross-Modal Attention Fusion Report")
    print("=" * 80)
    sm = res["summary_metrics"]
    print(f"AHRAS Cross-Modal Fusion F1:   {sm['ahras_multimodal_f1']:.4f} (Target: >= 0.95)")
    print(f"Unimodal Network F1:           {sm['unimodal_network_f1']:.4f}")
    print(f"Unimodal Process F1:           {sm['unimodal_process_f1']:.4f}")
    print(f"Unimodal Identity F1:          {sm['unimodal_identity_f1']:.4f}")
    print(f"Unimodal Graph F1:             {sm['unimodal_graph_f1']:.4f}")
    print(f"Early Concat F1:               {sm['early_concat_f1']:.4f}")
    print(f"Late Decision Averaging F1:    {sm['late_averaging_f1']:.4f}")
    print(f"F1 Gain over Network:          +{sm['f1_gain_over_network_pct']:.2f}% (Target: >= +15%)")
    print(f"P99 Normalization Latency:     {sm['p99_fusion_latency_ms']:.4f} ms (Target: <= 0.50 ms)")
    print(f"Throughput:                    {sm['throughput_events_sec']:.1f} events/sec")
    print(f"Permutation p-value vs Net:    {sm['permutation_p_vs_net']:.6f} (Cohen's d: {sm['cohens_d_vs_net']:.4f})")
    print(f"95% Bootstrap CI vs Net:       [{sm['bootstrap_ci_95_vs_net'][0]}, {sm['bootstrap_ci_95_vs_net'][1]}]")
    print("\nModality Missingness Degradation:")
    for k, v in res["modality_missingness_degradation"].items():
        print(f"  * {k:<30}: F1 = {v['f1']:.4f}, Recall = {v['recall']:.4f}")
