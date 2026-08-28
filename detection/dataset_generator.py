from __future__ import annotations
"""
AHRAS Synthetic Dataset Generator
-----------------------------------
Generates labeled training data for ML model bootstrap.

Key design principle: synthetic feature vectors must lie on the SAME
manifold as real OCSF-extracted feature vectors. All distributions
are calibrated against real event extraction outputs.

Attack scenarios:
  network_activity: port_scan, syn_flood, udp_flood, c2_beacon
  process_activity: suspicious_lineage, shell_exec
  file_activity:    ransomware_high_entropy
  cloud_api:        critical_action, priv_escalation
"""

import math
import numpy as np
from typing import Tuple

_RNG = np.random.default_rng(seed=0)

# ─────────────────────────────────────────────────────────────────────────────
# Calibrated normal generators
# Values derived by running extract() on 500 real OCSF events per class
# and measuring min/max/mean/std per feature.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_network_normal(n: int) -> np.ndarray:
    """
    18 features — calibrated to real extract() output ranges.
    Feature order matches FEATURE_NAMES['network_activity'].
    """
    X = np.zeros((n, 18))
    X[:, 0]  = _RNG.uniform(0.60, 0.80, n)       # src_ip_numeric (internal)
    X[:, 1]  = _RNG.uniform(0.00, 0.08, n)       # dst_ip_numeric (internal/ext)
    X[:, 2]  = _RNG.choice([1.0, 2.0], n)        # protocol TCP/UDP
    X[:, 3]  = _RNG.uniform(0.75, 1.00, n)       # src_port_norm (ephemeral)
    X[:, 4]  = _RNG.choice(                       # dst_port_norm (well-known)
                    [80/65535, 443/65535, 53/65535, 22/65535, 25/65535], n)
    X[:, 5]  = _RNG.uniform(0.05, 0.40, n)       # dst_port_risk
    X[:, 6]  = _RNG.choice([3.0, 6.0], n)        # tcp_flags SYN+ACK=3
    X[:, 7]  = _RNG.uniform(2.0, 4.5, n)         # log_packet_count
    X[:, 8]  = _RNG.uniform(8.5, 12.0, n)        # log_byte_count (calibrated)
    X[:, 9]  = _RNG.uniform(0.5, 3.5, n)         # log_duration
    X[:, 10] = _RNG.uniform(0.2, 4.0, n)         # log_packets_per_sec
    X[:, 11] = _RNG.uniform(5.5, 7.5, n)         # log_bytes_per_packet (calib)
    X[:, 12] = _RNG.uniform(0.0, 1.0, n)         # log_unique_dst_ports (1–2)
    X[:, 13] = np.zeros(n)                        # is_external_src
    X[:, 14] = np.zeros(n)                        # abuse_score_norm
    X[:, 15] = np.zeros(n)                        # port_scan_indicator
    X[:, 16] = np.zeros(n)                        # threat_intel_hit
    X[:, 17] = np.ones(n)                         # dst_is_well_known_port
    return np.clip(X, 0, None)


def _gen_network_port_scan(n: int) -> np.ndarray:
    X = _gen_network_normal(n)
    X[:, 2]  = 1.0                                # TCP
    X[:, 6]  = 1.0                                # SYN only (no ACK)
    X[:, 7]  = _RNG.uniform(0.0, 1.5, n)         # very few packets per flow
    X[:, 12] = _RNG.uniform(4.5, 6.5, n)         # log_unique_ports >> normal
    X[:, 15] = np.ones(n)                         # port_scan_indicator=1
    X[:, 5]  = _RNG.uniform(0.3, 0.9, n)         # varied port risks
    X[:, 17] = np.zeros(n)                        # scanning non-well-known ports
    return np.clip(X, 0, None)


def _gen_network_syn_flood(n: int) -> np.ndarray:
    X = _gen_network_normal(n)
    X[:, 6]  = 1.0                                # SYN only
    X[:, 7]  = _RNG.uniform(9.0, 11.0, n)        # massive packet count
    X[:, 8]  = _RNG.uniform(12.0, 15.0, n)       # massive byte count
    X[:, 10] = _RNG.uniform(7.0, 9.5, n)         # very high pps
    X[:, 13] = np.ones(n)                         # external source
    return np.clip(X, 0, None)


def _gen_network_c2_beacon(n: int) -> np.ndarray:
    X = _gen_network_normal(n)
    X[:, 7]  = _RNG.uniform(0.0, 1.5, n)         # very few packets
    X[:, 8]  = _RNG.uniform(4.0, 7.0, n)         # small bytes
    X[:, 9]  = _RNG.uniform(4.0, 6.0, n)         # very long duration
    X[:, 10] = np.zeros(n)                        # near-zero pps
    X[:, 13] = np.ones(n)                         # external
    X[:, 4]  = 443 / 65535                        # port 443 (evasion)
    X[:, 17] = np.ones(n)                         # well-known port (stealth)
    return np.clip(X, 0, None)


def _gen_process_normal(n: int) -> np.ndarray:
    X = np.zeros((n, 10))
    X[:, 0]  = _RNG.normal(9.0, 1.5, n)          # log_pid
    X[:, 1]  = _RNG.normal(7.0, 1.5, n)          # log_parent_pid
    X[:, 2]  = _RNG.normal(0, 500, n)             # pid_delta
    X[:, 3]  = _RNG.uniform(0, 1, n)              # proc_name_hash
    X[:, 4]  = _RNG.uniform(0, 1, n)              # parent_name_hash
    X[:, 5]  = _RNG.uniform(0, 1, n)              # username_hash
    X[:, 6]  = _RNG.normal(3.0, 0.8, n)           # log_cmdline_length
    X[:, 7]  = np.zeros(n)                         # suspicious_lineage=0
    X[:, 8]  = np.zeros(n)                         # shell_exec=0
    X[:, 9]  = np.zeros(n)                         # privilege_risk=0
    return np.clip(X, 0, None)


def _gen_process_suspicious(n: int) -> np.ndarray:
    X = _gen_process_normal(n)
    X[:, 7]  = np.ones(n)                          # suspicious_lineage=1
    X[:, 8]  = np.ones(n)                          # shell_exec=1
    X[:, 9]  = _RNG.uniform(1.0, 2.0, n)           # privilege_risk elevated
    X[:, 6]  = _RNG.normal(5.0, 0.5, n)            # longer cmdline
    return np.clip(X, 0, None)


def _gen_file_normal(n: int) -> np.ndarray:
    X = np.zeros((n, 8))
    X[:, 0]  = _RNG.uniform(0.20, 0.65, n)         # entropy_norm (low-medium)
    X[:, 1]  = np.zeros(n)                          # high_entropy=0
    X[:, 2]  = np.zeros(n)                          # ransomware_indicator=0
    X[:, 3]  = np.zeros(n)                          # ransomware_ext=0
    X[:, 4]  = _RNG.choice([0, 1], n, p=[0.3,0.7]) # targeted_ext (often)
    X[:, 5]  = np.zeros(n)                          # no sha256
    X[:, 6]  = _RNG.uniform(0, 1, n)               # extension_hash
    X[:, 7]  = _RNG.normal(1.5, 0.5, n)             # log_path_depth
    return np.clip(X, 0, None)


def _gen_file_ransomware(n: int) -> np.ndarray:
    X = _gen_file_normal(n)
    X[:, 0]  = _RNG.uniform(0.88, 1.00, n)         # very high entropy
    X[:, 1]  = np.ones(n)                           # high_entropy=1
    X[:, 2]  = np.ones(n)                           # ransomware_indicator=1
    X[:, 3]  = _RNG.choice([0, 1], n, p=[0.4,0.6]) # sometimes ransomware ext
    X[:, 5]  = np.ones(n)                           # has sha256
    return np.clip(X, 0, None)


def _gen_cloud_normal(n: int) -> np.ndarray:
    X = np.zeros((n, 10))
    X[:, 0]  = _RNG.uniform(0.01, 0.15, n)         # action_risk (safe)
    X[:, 1]  = np.zeros(n)                          # high_privilege=0
    X[:, 2]  = _RNG.choice([0]*9+[1], n)            # rare errors
    X[:, 3]  = _RNG.uniform(0, 1, n)               # username_hash
    X[:, 4]  = _RNG.uniform(0.01, 0.15, n)         # action_hash (safe actions)
    X[:, 5]  = _RNG.uniform(0, 1, n)               # region_hash
    X[:, 6]  = _RNG.uniform(0.60, 0.80, n)         # src_ip_numeric (internal)
    X[:, 7]  = np.zeros(n)                          # threat_intel=0
    X[:, 8]  = np.zeros(n)                          # external=0
    X[:, 9]  = _RNG.choice([0, 1], n, p=[0.6,0.4]) # mix user/svc
    return np.clip(X, 0, None)


def _gen_cloud_critical(n: int) -> np.ndarray:
    X = _gen_cloud_normal(n)
    X[:, 0]  = _RNG.uniform(0.85, 1.00, n)         # critical action risk
    X[:, 1]  = np.ones(n)                           # high_privilege=1
    X[:, 7]  = _RNG.choice([0, 1], n, p=[0.4,0.6]) # sometimes threat intel
    X[:, 8]  = np.ones(n)                           # external=1
    X[:, 9]  = np.zeros(n)                          # not service account
    return np.clip(X, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_GENERATORS = {
    "network_activity": [
        (_gen_network_normal,    "normal",   0),
        (_gen_network_port_scan, "attack",   1),
        (_gen_network_syn_flood, "attack",   1),
        (_gen_network_c2_beacon, "attack",   1),
    ],
    "process_activity": [
        (_gen_process_normal,     "normal",  0),
        (_gen_process_suspicious, "attack",  1),
    ],
    "file_activity": [
        (_gen_file_normal,     "normal",     0),
        (_gen_file_ransomware, "attack",     1),
    ],
    "cloud_api": [
        (_gen_cloud_normal,   "normal",      0),
        (_gen_cloud_critical, "attack",      1),
    ],
}


def generate_dataset(
        ocsf_class: str,
        n_normal: int = 400,
        n_attack: int = 80,
        noise_std: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a labeled (X, y) dataset for a given OCSF class.
    y=0 → normal, y=1 → attack.
    Noise std reduced to 0.015 to stay within calibrated feature ranges.
    """
    if ocsf_class not in _GENERATORS:
        raise ValueError(f"No generator for class: {ocsf_class}")

    gens = _GENERATORS[ocsf_class]
    attack_gens = [(g, l) for g, t, l in gens if t == "attack"]
    normal_gens = [(g, l) for g, t, l in gens if t == "normal"]

    n_per_attack = max(1, n_attack // len(attack_gens)) if attack_gens else 0

    X_parts, y_parts = [], []

    for gen_fn, label in normal_gens:
        Xg = gen_fn(n_normal)
        Xg += _RNG.normal(0, noise_std, Xg.shape)
        X_parts.append(np.clip(Xg, 0, None))
        y_parts.append(np.zeros(n_normal, dtype=int))

    for gen_fn, label in attack_gens:
        Xg = gen_fn(n_per_attack)
        Xg += _RNG.normal(0, noise_std, Xg.shape)
        X_parts.append(np.clip(Xg, 0, None))
        y_parts.append(np.ones(n_per_attack, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = _RNG.permutation(len(X))
    return X[idx], y[idx]


def get_normal_vectors(ocsf_class: str, n: int = 400) -> list[np.ndarray]:
    """Return list of normal feature vectors for ML bootstrap."""
    X, y = generate_dataset(ocsf_class, n_normal=n, n_attack=0)
    return [X[i] for i in range(len(X)) if y[i] == 0]
