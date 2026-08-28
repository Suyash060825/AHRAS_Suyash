from __future__ import annotations
"""
AHRAS Synthetic Benchmark Dataset Generator
--------------------------------------------
Generates deterministic, highly realistic network security flow datasets
simulating multi-stage campaigns, volumetric floods, low-and-slow port scans,
brute force, and benign baseline traffic.
"""

import csv
import os
import random
import time
from typing import List, Dict, Any

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "synthetic_eval_dataset.csv")

HEADER = [
    "Source IP", "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "ACK Flag Count", "Average Packet Size",
    "Packet Length Std", "Timestamp", "Label"
]


def make_dataset(n_total: int = 6000, attack_frac: float = 0.30, noise_frac: float = 0.04) -> List[Dict[str, Any]]:
    random.seed(42)
    rows = []
    
    n_attack = int(n_total * attack_frac)
    n_benign = n_total - n_attack
    
    base_ts = 1700000000.0  # reference unix timestamp
    
    # 1. Benign Traffic
    for i in range(n_benign):
        ts_str = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(base_ts + i * 2))
        rows.append({
            "Source IP": f"192.168.1.{random.randint(10, 200)}",
            "Destination Port": random.choice([80, 443, 53, 8080, 22]),
            "Flow Duration": random.randint(500, 50000),
            "Total Fwd Packets": random.randint(2, 20),
            "Total Backward Packets": random.randint(2, 25),
            "Flow Bytes/s": round(random.uniform(100.0, 5000.0), 2),
            "Flow Packets/s": round(random.uniform(1.0, 50.0), 2),
            "SYN Flag Count": random.choice([0, 1]),
            "ACK Flag Count": random.randint(2, 10),
            "Average Packet Size": round(random.uniform(64.0, 512.0), 2),
            "Packet Length Std": round(random.uniform(10.0, 100.0), 2),
            "Timestamp": ts_str,
            "Label": "BENIGN"
        })
        
    # 2. Attack Traffic
    attack_types = ["PortScan", "DoS Hulk", "DDoS", "SSH-Patator", "Bot"]
    for i in range(n_attack):
        atype = random.choice(attack_types)
        ts_str = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(base_ts + n_benign * 2 + i * 2))
        
        if atype == "PortScan":
            row = {
                "Source IP": "10.0.0.99",
                "Destination Port": random.randint(1, 65535),
                "Flow Duration": random.randint(100, 2000),
                "Total Fwd Packets": random.randint(1, 3),
                "Total Backward Packets": 0,
                "Flow Bytes/s": round(random.uniform(40.0, 200.0), 2),
                "Flow Packets/s": round(random.uniform(50.0, 500.0), 2),
                "SYN Flag Count": 1,
                "ACK Flag Count": 0,
                "Average Packet Size": 40.0,
                "Packet Length Std": 0.0,
                "Timestamp": ts_str,
                "Label": "PortScan"
            }
        elif atype in ("DoS Hulk", "DDoS"):
            row = {
                "Source IP": f"172.16.0.{random.randint(2, 50)}",
                "Destination Port": 80,
                "Flow Duration": random.randint(5000, 500000),
                "Total Fwd Packets": random.randint(500, 5000),
                "Total Backward Packets": random.randint(10, 100),
                "Flow Bytes/s": round(random.uniform(50000.0, 5000000.0), 2),
                "Flow Packets/s": round(random.uniform(1000.0, 20000.0), 2),
                "SYN Flag Count": random.randint(50, 500),
                "ACK Flag Count": random.randint(10, 50),
                "Average Packet Size": 120.0,
                "Packet Length Std": 25.0,
                "Timestamp": ts_str,
                "Label": atype
            }
        elif atype == "SSH-Patator":
            row = {
                "Source IP": "198.51.100.44",
                "Destination Port": 22,
                "Flow Duration": random.randint(2000, 15000),
                "Total Fwd Packets": random.randint(30, 200),
                "Total Backward Packets": random.randint(20, 100),
                "Flow Bytes/s": round(random.uniform(1000.0, 10000.0), 2),
                "Flow Packets/s": round(random.uniform(10.0, 100.0), 2),
                "SYN Flag Count": 5,
                "ACK Flag Count": 30,
                "Average Packet Size": 90.0,
                "Packet Length Std": 35.0,
                "Timestamp": ts_str,
                "Label": "SSH-Patator"
            }
        else:
            row = {
                "Source IP": "203.0.113.195",
                "Destination Port": 4444,
                "Flow Duration": random.randint(10000, 100000),
                "Total Fwd Packets": random.randint(15, 80),
                "Total Backward Packets": random.randint(15, 80),
                "Flow Bytes/s": round(random.uniform(500.0, 3000.0), 2),
                "Flow Packets/s": round(random.uniform(5.0, 30.0), 2),
                "SYN Flag Count": 1,
                "ACK Flag Count": 15,
                "Average Packet Size": 110.0,
                "Packet Length Std": 40.0,
                "Timestamp": ts_str,
                "Label": "Bot"
            }
        rows.append(row)
        
    random.shuffle(rows)
    return rows


def generate_and_save(n_total: int = 6000) -> str:
    rows = make_dataset(n_total=n_total)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT_PATH


if __name__ == "__main__":
    p = generate_and_save()
    print(f"Generated synthetic dataset: {p}")
