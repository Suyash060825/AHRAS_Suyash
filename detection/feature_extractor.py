"""
AHRAS Feature Extractor
------------------------
Converts OCSF-normalized events into flat numeric feature vectors
for ML models and statistical engines.

Per-class feature counts:
  network_activity  → 18 features
  process_activity  → 10 features
  file_activity     →  8 features
  cloud_api         → 10 features
  network_conn      →  8 features

All features are float64, non-negative, NaN/Inf-free.
Categorical values use stable hash encoding (reproducible across runs).

Publication note:
  Feature names are exported for XAI explanation labelling.
  All extractors guard against None/missing fields.
"""

from __future__ import annotations

import math
import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Stable categorical encodings ─────────────────────────────────────────────

_PROTOCOLS: Dict[str, int] = {"TCP": 1, "UDP": 2, "ICMP": 3, "OTHER": 0}

_FLAG_BITS: Dict[str, int] = {
    "SYN": 1, "ACK": 2, "FIN": 4, "RST": 8, "PSH": 16, "URG": 32,
}

_CLOUD_ACTIONS: Dict[str, int] = {
    "s3:GetObject": 1, "s3:PutObject": 2, "s3:ListBucket": 3,
    "ec2:DescribeInstances": 4, "cloudwatch:PutMetricData": 5,
    "lambda:InvokeFunction": 6, "rds:DescribeDBInstances": 7,
    "iam:ListUsers": 10, "iam:GetUser": 11,
    "ec2:DescribeSecurityGroups": 12, "sts:GetCallerIdentity": 13,
    "sts:AssumeRole": 14,
    "iam:CreateUser": 20, "iam:CreateAccessKey": 21,
    "iam:AttachUserPolicy": 22, "iam:PutUserPolicy": 23,
    "ec2:AuthorizeSecurityGroupIngress": 24, "s3:PutBucketAcl": 25,
    "s3:DeleteBucket": 26,
    "cloudtrail:StopLogging": 30, "cloudtrail:DeleteTrail": 31,
    "iam:DeleteAccountPasswordPolicy": 32, "iam:DeactivateMFADevice": 33,
}

_WELL_KNOWN_PORTS = frozenset([
    21, 22, 23, 25, 53, 80, 110, 143, 443,
    445, 3306, 3389, 5432, 6379, 8080, 8443, 27017,
])

_SUSPICIOUS_CHILDREN = frozenset([
    "bash", "sh", "zsh", "dash", "nc", "ncat", "nmap",
    "wget", "curl", "perl", "python3", "python",
])

_RANSOMWARE_EXTENSIONS = frozenset([
    ".enc", ".locked", ".crypt", ".crypto", ".crypted",
    ".encrypted", ".pay2me", ".cry", ".wnry", ".wncry",
])

_TARGET_EXTENSIONS = frozenset([
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pdf", ".jpg", ".jpeg", ".png", ".mp4", ".zip",
    ".tar", ".gz", ".sql", ".db", ".bak", ".txt",
])

_SHELL_INDICATORS = [
    " -c ", "exec", "eval", "base64", "powershell",
    "wget", "curl", "chmod +x",
]


# ─────────────────────────────────────────────────────────────────────────────
# Safe field accessors
# ─────────────────────────────────────────────────────────────────────────────

def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur if cur is not None else default


def _flt(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _str(val) -> str:
    return str(val) if val is not None else ""


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _ip_to_float(ip: str) -> float:
    """IPv4 → normalised float [0,1]. Returns 0.0 on error."""
    try:
        parts = [int(p) for p in str(ip).split(".")]
        if len(parts) == 4 and all(0 <= p <= 255 for p in parts):
            raw = sum(p * (256 ** (3 - i)) for i, p in enumerate(parts))
            return raw / (256 ** 4)
    except (ValueError, AttributeError):
        pass
    return 0.0


_PORT_RISK: Dict[int, float] = {
    0: 0.5, 21: 0.8, 22: 0.4, 23: 0.9, 25: 0.3,
    80: 0.1, 443: 0.1, 445: 0.8, 3389: 0.8,
    4444: 0.9, 1337: 0.9, 31337: 0.9,
}


def _port_risk(port: int) -> float:
    return _PORT_RISK.get(port, 0.3)


def _flag_vector(flags) -> int:
    bits = 0
    for f in (flags or []):
        bits |= _FLAG_BITS.get(_str(f), 0)
    return bits


def _str_hash_norm(s: str) -> float:
    """Stable hash → [0,1]. Reproducible across runs via MD5."""
    if not s:
        return 0.0
    h = int(hashlib.md5(s.encode(), usedforsecurity=False).hexdigest(), 16)
    return (h % 100_000) / 100_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature names (parallel to vectors — used by XAI)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: Dict[str, List[str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-class extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_network(evt: dict) -> np.ndarray:
    src      = evt.get("src_endpoint") or {}
    dst      = evt.get("dst_endpoint") or {}
    tr       = evt.get("traffic") or {}
    enr      = evt.get("enrichment") or {}

    src_ip   = _str(src.get("ip"))
    dst_ip   = _str(dst.get("ip"))
    src_port = _int(src.get("port"))
    dst_port = _int(dst.get("port"))
    proto    = _str(evt.get("protocol", "OTHER"))
    flags    = evt.get("tcp_flags") or []
    packets  = _flt(tr.get("packets"))
    byt      = _flt(tr.get("bytes"))
    dur      = max(_flt(tr.get("duration_sec")), 1e-6)
    u_ports  = _flt(evt.get("unique_dst_ports", 1))

    pps      = packets / dur
    bpp      = byt / max(packets, 1.0)
    geo      = enr.get("geo") or src.get("geo") or {}
    is_ext   = 0.0 if enr.get("is_private", True) else 1.0
    abuse    = _flt(enr.get("abuse_score", 0)) / 100.0
    port_scan = 1.0 if enr.get("port_scan_indicator") else 0.0
    threat   = 1.0 if enr.get("is_threat_intel_hit") else 0.0

    return np.array([
        _ip_to_float(src_ip),              # 0
        _ip_to_float(dst_ip),              # 1
        float(_PROTOCOLS.get(proto, 0)),   # 2
        float(src_port) / 65535.0,         # 3
        float(dst_port) / 65535.0,         # 4
        _port_risk(dst_port),              # 5
        float(_flag_vector(flags)),        # 6
        math.log1p(packets),               # 7
        math.log1p(byt),                   # 8
        math.log1p(dur),                   # 9
        math.log1p(pps),                   # 10
        math.log1p(bpp),                   # 11
        math.log1p(u_ports),               # 12
        is_ext,                            # 13
        abuse,                             # 14
        port_scan,                         # 15
        threat,                            # 16
        1.0 if dst_port in _WELL_KNOWN_PORTS else 0.0,  # 17
    ], dtype=np.float64)


FEATURE_NAMES["network_activity"] = [
    "src_ip_numeric", "dst_ip_numeric", "protocol",
    "src_port_norm", "dst_port_norm", "dst_port_risk",
    "tcp_flags_bitmask", "log_packet_count", "log_byte_count",
    "log_duration", "log_packets_per_sec", "log_bytes_per_packet",
    "log_unique_dst_ports", "is_external_src",
    "abuse_score_norm", "port_scan_indicator",
    "threat_intel_hit", "dst_is_well_known_port",
]


def _extract_process(evt: dict) -> np.ndarray:
    actor   = evt.get("actor") or {}
    proc    = actor.get("process") or {}
    parent  = evt.get("process") or {}
    enr     = evt.get("enrichment") or {}

    pid         = _flt(proc.get("pid"))
    parent_pid  = _flt(parent.get("parent_pid"))
    name        = _str(proc.get("name"))
    parent_name = _str(parent.get("parent_name"))
    cmdline     = _str(proc.get("cmd_line"))
    username    = _str(_get(proc, "user", "name"))

    suspicious  = 1.0 if enr.get("suspicious_lineage") else 0.0
    has_shell   = 1.0 if any(s in cmdline.lower() for s in _SHELL_INDICATORS) else 0.0
    is_root     = 1.0 if username in ("root", "SYSTEM", "Administrator") else 0.0
    child_risk  = 1.0 if name in _SUSPICIOUS_CHILDREN else 0.0

    return np.array([
        math.log1p(pid),
        math.log1p(parent_pid),
        float(abs(pid - parent_pid)),
        _str_hash_norm(name),
        _str_hash_norm(parent_name),
        _str_hash_norm(username),
        math.log1p(len(cmdline)),
        suspicious,
        has_shell,
        is_root + child_risk,
    ], dtype=np.float64)


FEATURE_NAMES["process_activity"] = [
    "log_pid", "log_parent_pid", "pid_delta",
    "proc_name_hash", "parent_name_hash", "username_hash",
    "log_cmdline_length", "suspicious_lineage",
    "shell_execution_indicator", "privilege_risk_score",
]


def _extract_file(evt: dict) -> np.ndarray:
    file_info  = evt.get("file") or {}
    enr        = evt.get("enrichment") or {}

    filepath   = _str(file_info.get("path"))
    sha256     = _str(file_info.get("sha256"))
    entropy    = _flt(enr.get("entropy"))
    high_ent   = 1.0 if enr.get("high_entropy") else 0.0
    ransomw    = 1.0 if enr.get("ransomware_indicator") else 0.0

    ext = ""
    if "." in filepath:
        ext = "." + filepath.rsplit(".", 1)[-1].lower()

    ransomw_ext = 1.0 if ext in _RANSOMWARE_EXTENSIONS else 0.0
    target_ext  = 1.0 if ext in _TARGET_EXTENSIONS else 0.0
    has_hash    = 1.0 if sha256 else 0.0
    path_depth  = float(filepath.count("/"))

    return np.array([
        entropy / 8.0,
        high_ent,
        ransomw,
        ransomw_ext,
        target_ext,
        has_hash,
        _str_hash_norm(ext),
        math.log1p(path_depth),
    ], dtype=np.float64)


FEATURE_NAMES["file_activity"] = [
    "entropy_norm", "high_entropy_flag", "ransomware_indicator",
    "ransomware_extension", "targeted_extension",
    "has_sha256", "extension_hash", "log_path_depth",
]


def _extract_cloud(evt: dict) -> np.ndarray:
    api    = evt.get("api") or {}
    actor  = evt.get("actor") or {}
    cloud  = evt.get("cloud") or {}
    enr    = evt.get("enrichment") or {}
    src    = evt.get("src_endpoint") or {}

    action   = _str(api.get("operation"))
    username = _str(_get(actor, "user", "name"))
    has_err  = 1.0 if api.get("error") else 0.0
    region   = _str(cloud.get("region"))
    src_ip   = _str(src.get("ip"))

    action_risk = _CLOUD_ACTIONS.get(action, 5) / 33.0
    high_priv   = 1.0 if enr.get("high_privilege_action") else 0.0
    threat      = 1.0 if enr.get("is_threat_intel_hit") else 0.0
    is_ext      = 0.0 if enr.get("is_private", True) else 1.0
    abuse       = _flt(enr.get("abuse_score", 0)) / 100.0
    is_svc      = 1.0 if "svc-" in username else 0.0
    unusual_ua  = 1.0 if "requests" in _str(evt.get("user_agent")).lower() else 0.0

    return np.array([
        action_risk,
        high_priv,
        has_err,
        _str_hash_norm(username),
        _str_hash_norm(action),
        _str_hash_norm(region),
        _ip_to_float(src_ip),
        threat,
        is_ext,
        is_svc,
    ], dtype=np.float64)


FEATURE_NAMES["cloud_api"] = [
    "action_risk_score", "high_privilege_action", "has_api_error",
    "username_hash", "action_hash", "region_hash",
    "src_ip_numeric", "threat_intel_hit",
    "external_source", "is_service_account",
]


def _extract_network_conn(evt: dict) -> np.ndarray:
    actor     = evt.get("actor") or {}
    proc      = actor.get("process") or {}
    dst       = evt.get("dst_endpoint") or {}
    enr       = evt.get("enrichment") or {}

    proc_name   = _str(proc.get("name"))
    remote_ip   = _str(dst.get("ip"))
    remote_port = _int(dst.get("port"))
    threat      = 1.0 if enr.get("is_threat_intel_hit") else 0.0
    abuse       = _flt(enr.get("abuse_score", 0)) / 100.0
    is_ext      = 0.0 if enr.get("is_private", True) else 1.0

    return np.array([
        _str_hash_norm(proc_name),
        _ip_to_float(remote_ip),
        float(remote_port) / 65535.0,
        _port_risk(remote_port),
        threat,
        abuse,
        is_ext,
        1.0 if remote_port in _WELL_KNOWN_PORTS else 0.0,
    ], dtype=np.float64)


FEATURE_NAMES["network_conn"] = [
    "process_name_hash", "remote_ip_numeric", "remote_port_norm",
    "remote_port_risk", "threat_intel_hit",
    "abuse_score_norm", "is_external", "remote_is_well_known",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    "network_activity": _extract_network,
    "process_activity": _extract_process,
    "file_activity":    _extract_file,
    "cloud_api":        _extract_cloud,
    "network_conn":     _extract_network_conn,
}


def extract(evt: dict) -> Optional[np.ndarray]:
    """
    Extract a feature vector from an OCSF-normalised event.
    Returns None for unsupported OCSF classes.
    Guarantees: output is float64, no NaN, no Inf, all values >= 0.
    """
    if not isinstance(evt, dict):
        return None
    cls = evt.get("ocsf_class", "")
    fn  = _EXTRACTORS.get(cls)
    if fn is None:
        log.debug(f"[FEAT] No extractor for class: {cls}")
        return None
    try:
        vec = fn(evt)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = np.clip(vec, 0.0, None)
        return vec
    except Exception as e:
        log.error(f"[FEAT] Extraction error for {cls}: {e}")
        return None


def feature_names(ocsf_class: str) -> List[str]:
    """Return human-readable feature names for a given OCSF class."""
    return FEATURE_NAMES.get(ocsf_class, [])


def feature_dim(ocsf_class: str) -> int:
    """Return feature vector dimension for a given OCSF class."""
    names = FEATURE_NAMES.get(ocsf_class)
    return len(names) if names else 0
