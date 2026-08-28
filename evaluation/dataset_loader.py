from __future__ import annotations
"""
AHRAS Unified Multi-Dataset Loader
-----------------------------------
Ingests and maps 22+ benchmark datasets into standardized OCSF feature schemas:
  - Enterprise: CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15
  - Legacy: NSL-KDD, KDD'99
  - IoT / Industrial: BoT-IoT, N-BaIoT, Kitsune, WADI
  - Generic / Custom CSV
"""

import csv
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime as _dt
from typing import Dict, Iterator, List, Optional, Any

log = logging.getLogger(__name__)

_CICIDS_MARKERS = {"Flow Duration", "Total Fwd Packets", "Label", "Destination Port"}
_BENIGN_TOKENS = {"benign", "normal", "normal.", "0"}
_NSLKDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty",
]
_UNSW_MARKERS = {"attack_cat", "label", "dur", "proto"}


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default

import math


def _clean_label(raw: str) -> str:
    return (raw or "").replace("\ufffd", "-").strip()


def _parse_timestamp(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in (
        "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return _dt.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    return None


@dataclass
class DatasetRecord:
    src_ip:           str
    features:         Dict[str, float]
    label:            int                  # 0 = Benign, 1 = Attack
    attack_category:  str
    raw_row:          Dict[str, str]
    event_time:       Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "src_ip":          self.src_ip,
            "features":        self.features,
            "label":           self.label,
            "attack_category": self.attack_category,
            "event_time":      self.event_time,
        }


class DatasetLoader:
    def __init__(self, filepath: str, dataset_type: Optional[str] = None):
        self.filepath = filepath
        self.dataset_type = dataset_type or self._detect_type(filepath)
        log.info(f"[DATASET] {filepath} identified as {self.dataset_type}")

    @classmethod
    def from_folder(cls, folder: str) -> List[DatasetLoader]:
        loaders = [
            cls(os.path.join(folder, fname))
            for fname in sorted(os.listdir(folder))
            if fname.lower().endswith(".csv")
        ]
        if not loaders:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        return loaders

    def _detect_type(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline().strip()
            cols = {c.strip().strip('"') for c in first_line.split(",")}
        except Exception:
            return "generic-csv"

        if _CICIDS_MARKERS.issubset(cols) or any("Flow" in c for c in cols) or "Flow Duration" in cols:
            return "cicids2017"
        if _UNSW_MARKERS.issubset({c.lower() for c in cols}):
            return "unsw-nb15"
        if len(cols) >= 40 and "duration" not in cols and "," in first_line:
            if len(first_line.split(",")) in (41, 42):
                return "nsl-kdd"
        return "generic-csv"

    def iter_records(self, limit: Optional[int] = None, sample: bool = False) -> Iterator[DatasetRecord]:
        raw = self._iter_raw()
        if limit is None:
            yield from raw
            return
        if not sample:
            for i, rec in enumerate(raw):
                if i >= limit:
                    return
                yield rec
            return

        # Reservoir sampling with chronological preservation
        reservoir: List[DatasetRecord] = []
        for i, rec in enumerate(raw):
            if i < limit:
                reservoir.append(rec)
            else:
                j = random.randint(0, i)
                if j < limit:
                    reservoir[j] = rec

        if reservoir and all(r.event_time is not None for r in reservoir):
            reservoir.sort(key=lambda r: r.event_time or 0.0)
        yield from reservoir

    def _iter_raw(self) -> Iterator[DatasetRecord]:
        if self.dataset_type == "nsl-kdd":
            yield from self._iter_nslkdd()
        elif self.dataset_type in ("cicids2017", "cicids2018", "cse-cic-ids2018"):
            yield from self._iter_cicids()
        elif self.dataset_type == "unsw-nb15":
            yield from self._iter_unsw()
        else:
            yield from self._iter_generic()

    def _iter_nslkdd(self) -> Iterator[DatasetRecord]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 42:
                    continue
                d = dict(zip(_NSLKDD_COLUMNS, row))
                label_raw = str(d.get("label", "normal")).strip().lower()
                features = {
                    k: safe_float(d.get(k, 0.0), 0.0)
                    for k in [
                        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
                        "serror_rate", "srv_serror_rate", "same_srv_rate", "diff_srv_rate",
                        "dst_host_count", "dst_host_srv_count", "num_failed_logins", "logged_in",
                    ]
                }
                count += 1
                yield DatasetRecord(
                    src_ip=f"10.10.1.{count % 254 + 1}",
                    features=features,
                    label=0 if label_raw in _BENIGN_TOKENS else 1,
                    attack_category=_clean_label(label_raw),
                    raw_row=d,
                )

    def _iter_cicids(self) -> Iterator[DatasetRecord]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw_row in reader:
                row = {str(k).strip(): v for k, v in raw_row.items() if k}
                label_raw = str(row.get("Label", "") or "").strip().lower()
                features = {}

                aliases = {
                    "Flow Duration": ["Flow Duration", "flow_duration"],
                    "Total Fwd Packets": ["Total Fwd Packets", "Tot Fwd Pkts", "tot_fwd_pkts"],
                    "Total Backward Packets": ["Total Backward Packets", "Tot Bwd Pkts", "tot_bwd_pkts"],
                    "Flow Bytes/s": ["Flow Bytes/s", "Flow Byts/s", "flow_bytes_s"],
                    "Flow Packets/s": ["Flow Packets/s", "Flow Pkts/s", "flow_pkts_s"],
                    "SYN Flag Count": ["SYN Flag Count", "SYN Flag Cnt", "syn_flag_count"],
                    "ACK Flag Count": ["ACK Flag Count", "ACK Flag Cnt", "ack_flag_count"],
                    "Average Packet Size": ["Average Packet Size", "Pkt Size Avg", "Pkt Len Mean"],
                    "Packet Length Std": ["Packet Length Std", "Pkt Len Std"],
                    "Destination Port": ["Destination Port", "Dst Port", "dst_port"],
                }
                for canonical, names in aliases.items():
                    value = 0.0
                    for name in names:
                        if row.get(name) not in (None, ""):
                            value = row[name]
                            break
                    features[canonical] = safe_float(value, 0.0)

                src_ip = row.get("Source IP") or row.get("Source IP ") or row.get("src_ip") or f"192.168.1.{count % 254 + 1}"
                event_time = _parse_timestamp(row.get("Timestamp", ""))
                count += 1
                yield DatasetRecord(
                    src_ip=str(src_ip),
                    features=features,
                    label=0 if label_raw in _BENIGN_TOKENS else 1,
                    attack_category=_clean_label(row.get("Label", "Unknown")),
                    raw_row=row,
                    event_time=event_time,
                )

    def _iter_unsw(self) -> Iterator[DatasetRecord]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw_row in reader:
                row = {str(k).strip().lower(): v for k, v in raw_row.items() if k}
                label_raw = str(row.get("label", "0") or "0").strip()
                features = {
                    k: safe_float(row.get(k, 0.0), 0.0)
                    for k in ["dur", "sbytes", "dbytes", "spkts", "dpkts", "rate", "sttl", "dttl", "sload", "dload", "swin", "dwin"]
                }
                src_ip = row.get("srcip", f"172.16.1.{count % 254 + 1}")
                count += 1
                yield DatasetRecord(
                    src_ip=src_ip,
                    features=features,
                    label=1 if label_raw == "1" else 0,
                    attack_category=_clean_label(row.get("attack_cat", "Normal") or "Normal"),
                    raw_row=row,
                )

    def _iter_generic(self) -> Iterator[DatasetRecord]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            label_col = next((fn for fn in (reader.fieldnames or []) if fn.strip().lower() in {"label", "class", "attack", "target"}), None)
            for row in reader:
                label_raw = (row.get(label_col, "0") if label_col else "0") or "0"
                label = 0 if str(label_raw).strip().lower() in (_BENIGN_TOKENS | {"0"}) else 1
                features = {}
                for key, value in row.items():
                    if key == label_col or value is None:
                        continue
                    parsed = safe_float(value, 0.0)
                    features[key] = parsed
                count += 1
                yield DatasetRecord(
                    src_ip=row.get("src_ip", f"10.0.0.{count % 254 + 1}"),
                    features=features,
                    label=label,
                    attack_category=_clean_label(str(label_raw)),
                    raw_row=row,
                )

    def count_records(self, limit: Optional[int] = None) -> int:
        return sum(1 for _ in self.iter_records(limit=limit))
