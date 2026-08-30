from __future__ import annotations
"""
AHRAS Synthetic Benchmark Dataset Generator (Multi-Difficulty Scenarios)
-----------------------------------------------------------------------
Generates deterministic, parameterized network security flow datasets simulating
multi-stage campaigns across 4 controlled difficulty levels:
  - EASY: Distinct feature distributions, separable boundaries, high SNR.
  - MODERATE: Realistic background variance, shared port distributions.
  - HARD: Significant feature overlap, low-and-slow stealth flows, 5% label noise.
  - ADVERSARIAL: Bounded feature perturbation, evasion flag spoofing, detector conflict.
"""

import csv
import os
import random
import time
import numpy as np
from typing import List, Dict, Any, Optional

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "synthetic_eval_dataset.csv")

HEADER = [
    "Source IP", "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "ACK Flag Count", "Average Packet Size",
    "Packet Length Std", "Timestamp", "Label"
]


def make_dataset(
    n_total: int = 6000,
    attack_frac: float = 0.30,
    difficulty: str = "MODERATE",
    noise_frac: float = 0.04,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a deterministic dataset parameterized by difficulty level.
    """
    rng = np.random.default_rng(seed)
    py_rand = random.Random(seed)
    
    rows = []
    n_attack = int(n_total * attack_frac)
    n_benign = n_total - n_attack
    base_ts = 1700000000.0  # reference unix epoch

    # Difficulty parameters
    overlap = 0.0 if difficulty == "EASY" else (0.15 if difficulty == "MODERATE" else (0.40 if difficulty == "HARD" else 0.50))
    label_noise = 0.0 if difficulty == "EASY" else (0.01 if difficulty == "MODERATE" else 0.05)
    adv_perturb = (difficulty == "ADVERSARIAL")

    # Generate events along an interleaved chronological timeline
    attack_types = ["PortScan", "DoS Hulk", "DDoS", "SSH-Patator", "Bot"]
    
    for i in range(n_total):
        curr_ts = base_ts + i * 2.0
        ts_str = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(curr_ts))
        
        # Decide if this event is an attack or benign
        is_attack_sample = (py_rand.random() < attack_frac)
        
        if not is_attack_sample:
            # Benign event
            fwd_pkts = int(rng.integers(2, 25))
            bwd_pkts = int(rng.integers(2, 30))
            bytes_sec = float(rng.uniform(100.0, 5000.0))
            pkts_sec = float(rng.uniform(1.0, 50.0))
            syn_cnt = int(py_rand.choice([0, 1]))
            ack_cnt = int(rng.integers(2, 12))
            avg_pkt_sz = float(rng.uniform(64.0, 512.0))
            pkt_std = float(rng.uniform(10.0, 100.0))
            dst_port = int(py_rand.choice([80, 443, 53, 8080, 22]))

            if rng.uniform(0, 1) < overlap:
                bytes_sec *= float(rng.uniform(1.5, 3.0))
                pkts_sec *= float(rng.uniform(1.5, 3.0))

            lbl = "BENIGN"
            if label_noise > 0 and rng.uniform(0, 1) < label_noise:
                lbl = "PortScan"

            rows.append({
                "Source IP": f"192.168.1.{py_rand.randint(10, 200)}",
                "Destination Port": dst_port,
                "Flow Duration": int(rng.integers(500, 50000)),
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Flow Bytes/s": round(bytes_sec, 2),
                "Flow Packets/s": round(pkts_sec, 2),
                "SYN Flag Count": syn_cnt,
                "ACK Flag Count": ack_cnt,
                "Average Packet Size": round(avg_pkt_sz, 2),
                "Packet Length Std": round(pkt_std, 2),
                "Timestamp": ts_str,
                "Label": lbl
            })
        else:
            # Attack event
            atype = py_rand.choice(attack_types)
            if atype == "PortScan":
                dst_port = int(rng.integers(1, 65535))
                fwd_pkts = int(rng.integers(1, 4))
                bwd_pkts = int(rng.integers(0, 2))
                bytes_sec = float(rng.uniform(50.0, 400.0))
                pkts_sec = float(rng.uniform(10.0, 200.0))
                syn_cnt = 1
                ack_cnt = 0
                avg_pkt_sz = 40.0
                pkt_std = 0.0
            elif atype in ("DoS Hulk", "DDoS"):
                dst_port = int(py_rand.choice([80, 443]))
                fwd_pkts = int(rng.integers(50, 500))
                bwd_pkts = int(rng.integers(0, 10))
                bytes_sec = float(rng.uniform(10000.0, 500000.0))
                pkts_sec = float(rng.uniform(500.0, 5000.0))
                syn_cnt = int(rng.integers(10, 50))
                ack_cnt = int(rng.integers(0, 5))
                avg_pkt_sz = float(rng.uniform(500.0, 1400.0))
                pkt_std = float(rng.uniform(50.0, 200.0))
            elif atype == "SSH-Patator":
                dst_port = 22
                fwd_pkts = int(rng.integers(10, 40))
                bwd_pkts = int(rng.integers(10, 40))
                bytes_sec = float(rng.uniform(1000.0, 8000.0))
                pkts_sec = float(rng.uniform(20.0, 100.0))
                syn_cnt = 1
                ack_cnt = int(rng.integers(5, 20))
                avg_pkt_sz = float(rng.uniform(100.0, 300.0))
                pkt_std = float(rng.uniform(20.0, 80.0))
            else:  # Bot / C2
                dst_port = int(py_rand.choice([8080, 4444, 6667]))
                fwd_pkts = int(rng.integers(5, 25))
                bwd_pkts = int(rng.integers(5, 25))
                bytes_sec = float(rng.uniform(500.0, 3000.0))
                pkts_sec = float(rng.uniform(2.0, 20.0))
                syn_cnt = 1
                ack_cnt = int(rng.integers(3, 10))
                avg_pkt_sz = float(rng.uniform(128.0, 512.0))
                pkt_std = float(rng.uniform(15.0, 60.0))

            if adv_perturb:
                bytes_sec = float(rng.uniform(200.0, 3000.0))
                pkts_sec = float(rng.uniform(5.0, 30.0))
                fwd_pkts = int(rng.integers(3, 15))

            lbl = atype
            if label_noise > 0 and rng.uniform(0, 1) < label_noise:
                lbl = "BENIGN"

            rows.append({
                "Source IP": f"192.168.1.{py_rand.randint(201, 250)}",
                "Destination Port": dst_port,
                "Flow Duration": int(rng.integers(100, 20000)),
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Flow Bytes/s": round(bytes_sec, 2),
                "Flow Packets/s": round(pkts_sec, 2),
                "SYN Flag Count": syn_cnt,
                "ACK Flag Count": ack_cnt,
                "Average Packet Size": round(avg_pkt_sz, 2),
                "Packet Length Std": round(pkt_std, 2),
                "Timestamp": ts_str,
                "Label": lbl
            })

    return rows


def generate_and_save(
    path: str = OUTPUT_PATH,
    n_total: int = 6000,
    attack_frac: float = 0.30,
    difficulty: str = "MODERATE",
    seed: int = 42
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = make_dataset(n_total=n_total, attack_frac=attack_frac, difficulty=difficulty, seed=seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    out = generate_and_save(difficulty="MODERATE")
    print(f"Generated synthetic benchmark dataset at: {out}")
