"""
AHRAS Signature Engine
-----------------------
Rule-based detection for known attack patterns across all OCSF classes.

Each rule is a pure function:
    rule(evt: dict) -> SignatureMatch | None

Rules are grouped by OCSF class for O(1) dispatch.
Adding a new rule: define the function, register it in RULE_REGISTRY.

Severity scale:  1=INFO  2=LOW  3=MEDIUM  4=HIGH  5=CRITICAL

Publication note:
  23 rules covering MITRE ATT&CK tactics: Discovery, Execution,
  Credential Access, Lateral Movement, Impact, Defense Evasion,
  Command and Control, Initial Access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignatureMatch:
    rule_id:         str
    rule_name:       str
    attack_type:     str
    severity:        int            # 1–5
    confidence:      float          # 0–1
    description:     str
    mitre_tactic:    str = ""
    mitre_technique: str = ""
    evidence:        dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

_T: dict = {
    "port_scan_unique_ports":  20,
    "port_scan_packets_max":   5,
    "flood_pps":               1000,
    "udp_flood_pkts":          3000,
    "dns_amp_bytes":           100_000,
    "ssh_brute_pkts":          100,
    "high_entropy_threshold":  7.2,
    "cloud_critical_actions": {
        "cloudtrail:StopLogging", "cloudtrail:DeleteTrail",
        "iam:DeleteAccountPasswordPolicy", "iam:DeactivateMFADevice",
    },
    "cloud_high_actions": {
        "iam:CreateUser", "iam:CreateAccessKey", "iam:AttachUserPolicy",
        "iam:PutUserPolicy", "ec2:AuthorizeSecurityGroupIngress",
        "s3:DeleteBucket", "s3:PutBucketAcl",
    },
    "suspicious_children": {
        "bash", "sh", "zsh", "dash", "nc", "ncat", "nmap",
        "wget", "curl", "perl", "python3", "python",
    },
    "suspicious_parents": {
        "libreoffice", "soffice", "winword", "excel",
        "python3", "python", "node", "java", "php", "ruby",
    },
    "ransomware_extensions": {
        ".enc", ".locked", ".crypt", ".crypto", ".crypted",
        ".encrypted", ".pay2me", ".cry", ".wnry", ".wncry",
    },
    "shell_patterns": [
        "base64 -d", "base64 --decode", "/dev/tcp",
        "exec 5<>", "wget http", "curl http", " | bash",
        "python -c", "perl -e", "ruby -e", "php -r",
    ],
    "cred_patterns": [
        "mimikatz", "lsass", "sekurlsa", "hashdump",
        "procdump", "/ma lsass", "ntdsutil", "vssadmin",
    ],
    "sensitive_paths": [
        "/etc/passwd", "/etc/shadow", "id_rsa", ".ssh/",
        "authorized_keys", "credentials", ".aws/credentials",
        "ntds.dit", "sam ", "/proc/",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Safe field accessors
# ─────────────────────────────────────────────────────────────────────────────

def _str(val) -> str:
    """Return val as str, empty string if None."""
    return str(val) if val is not None else ""


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get(d, *keys, default=None):
    """Safe nested dict access: _get(d,'a','b') == d.get('a',{}).get('b')"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# Network rules
# ─────────────────────────────────────────────────────────────────────────────

def _rule_port_scan(evt: dict) -> Optional[SignatureMatch]:
    u     = _int(_get(evt, "unique_dst_ports"))
    flags = _get(evt, "tcp_flags") or []
    pkts  = _int(_get(evt, "traffic", "packets"))
    if (u >= _T["port_scan_unique_ports"]
            and "SYN" in flags
            and "ACK" not in flags
            and pkts <= _T["port_scan_packets_max"] * u):
        return SignatureMatch(
            rule_id="NET-001", rule_name="Port Scan Detected",
            attack_type="port_scan", severity=3, confidence=0.90,
            description=f"SYN scan across {u} unique ports from "
                        f"{_get(evt,'src_endpoint','ip',default='')}",
            mitre_tactic="Discovery",
            mitre_technique="T1046 - Network Service Scanning",
            evidence={"unique_ports": u, "flags": flags, "packets": pkts},
        )


def _rule_syn_flood(evt: dict) -> Optional[SignatureMatch]:
    dur   = _float(_get(evt, "traffic", "duration_sec")) or 1.0
    pkts  = _float(_get(evt, "traffic", "packets"))
    flags = _get(evt, "tcp_flags") or []
    pps   = pkts / dur
    if pps >= _T["flood_pps"] and "SYN" in flags and "ACK" not in flags:
        return SignatureMatch(
            rule_id="NET-002", rule_name="SYN Flood",
            attack_type="dos_syn_flood", severity=4, confidence=0.88,
            description=f"SYN flood: {pps:.0f} pkt/s from "
                        f"{_get(evt,'src_endpoint','ip',default='')}",
            mitre_tactic="Impact",
            mitre_technique="T1499 - Endpoint Denial of Service",
            evidence={"pps": round(pps, 1), "flags": flags},
        )


def _rule_udp_flood(evt: dict) -> Optional[SignatureMatch]:
    pkts  = _float(_get(evt, "traffic", "packets"))
    proto = _str(_get(evt, "protocol"))
    if proto == "UDP" and pkts >= _T["udp_flood_pkts"]:
        return SignatureMatch(
            rule_id="NET-003", rule_name="UDP Flood",
            attack_type="dos_udp_flood", severity=4, confidence=0.85,
            description=f"UDP flood: {pkts:.0f} packets",
            mitre_tactic="Impact",
            mitre_technique="T1499 - Endpoint Denial of Service",
            evidence={"packets": pkts},
        )


def _rule_dns_amplification(evt: dict) -> Optional[SignatureMatch]:
    dst_port = _int(_get(evt, "dst_endpoint", "port"))
    byt      = _float(_get(evt, "traffic", "bytes"))
    proto    = _str(_get(evt, "protocol"))
    if dst_port == 53 and proto == "UDP" and byt >= _T["dns_amp_bytes"]:
        return SignatureMatch(
            rule_id="NET-004", rule_name="DNS Amplification Attack",
            attack_type="dns_amplification", severity=4, confidence=0.87,
            description=f"DNS amplification: {byt/1024:.1f} KB via UDP/53",
            mitre_tactic="Impact",
            mitre_technique="T1498 - Network Denial of Service",
            evidence={"bytes": byt, "dst_port": dst_port},
        )


def _rule_ssh_brute_force(evt: dict) -> Optional[SignatureMatch]:
    dst_port = _int(_get(evt, "dst_endpoint", "port"))
    pkts     = _float(_get(evt, "traffic", "packets"))
    if dst_port == 22 and pkts >= _T["ssh_brute_pkts"]:
        return SignatureMatch(
            rule_id="NET-005", rule_name="SSH Brute Force",
            attack_type="brute_force_ssh", severity=4, confidence=0.85,
            description=f"SSH brute force: {pkts:.0f} packets to port 22",
            mitre_tactic="Credential Access",
            mitre_technique="T1110 - Brute Force",
            evidence={"packets": pkts, "dst_port": 22},
        )


def _rule_smb_lateral(evt: dict) -> Optional[SignatureMatch]:
    dst_port = _int(_get(evt, "dst_endpoint", "port"))
    src_ip   = _str(_get(evt, "src_endpoint", "ip"))
    is_priv  = _get(evt, "enrichment", "is_private", default=True)
    pkts     = _float(_get(evt, "traffic", "packets"))
    if dst_port == 445 and is_priv and pkts > 10:
        return SignatureMatch(
            rule_id="NET-006", rule_name="SMB Lateral Movement",
            attack_type="lateral_movement_smb", severity=4, confidence=0.80,
            description=f"SMB lateral movement: {src_ip} → port 445",
            mitre_tactic="Lateral Movement",
            mitre_technique="T1021.002 - SMB/Windows Admin Shares",
            evidence={"dst_port": 445, "src_ip": src_ip},
        )


def _rule_rdp_access(evt: dict) -> Optional[SignatureMatch]:
    dst_port = _int(_get(evt, "dst_endpoint", "port"))
    is_priv  = _get(evt, "enrichment", "is_private", default=True)
    if dst_port == 3389 and not is_priv:
        return SignatureMatch(
            rule_id="NET-007", rule_name="External RDP Access",
            attack_type="rdp_external", severity=4, confidence=0.82,
            description=f"External RDP from {_get(evt,'src_endpoint','ip',default='')}",
            mitre_tactic="Lateral Movement",
            mitre_technique="T1021.001 - Remote Desktop Protocol",
            evidence={"dst_port": 3389, "external": True},
        )


def _rule_telnet(evt: dict) -> Optional[SignatureMatch]:
    if _int(_get(evt, "dst_endpoint", "port")) == 23:
        return SignatureMatch(
            rule_id="NET-008", rule_name="Telnet Usage",
            attack_type="telnet_cleartext", severity=3, confidence=0.75,
            description="Cleartext Telnet connection detected",
            mitre_tactic="Lateral Movement",
            mitre_technique="T1021 - Remote Services",
            evidence={"dst_port": 23},
        )


def _rule_threat_intel_hit(evt: dict) -> Optional[SignatureMatch]:
    enr   = evt.get("enrichment") or {}
    abuse = _int(enr.get("abuse_score", 0))
    src   = _str(_get(evt, "src_endpoint", "ip"))
    if enr.get("is_threat_intel_hit") and abuse > 50:
        sev = 5 if abuse > 80 else 4
        return SignatureMatch(
            rule_id="NET-009", rule_name="Threat Intel Hit",
            attack_type="threat_intel_match", severity=sev,
            confidence=min(0.95, abuse / 100.0),
            description=f"Source IP {src} AbuseIPDB score={abuse}",
            mitre_tactic="Initial Access",
            mitre_technique="T1190 - Exploit Public-Facing Application",
            evidence={"src_ip": src, "abuse_score": abuse},
        )


def _rule_c2_beacon(evt: dict) -> Optional[SignatureMatch]:
    pkts  = _float(_get(evt, "traffic", "packets"))
    dur   = _float(_get(evt, "traffic", "duration_sec"))
    is_priv = _get(evt, "enrichment", "is_private", default=True)
    dst_p = _int(_get(evt, "dst_endpoint", "port"))
    if (pkts <= 5 and dur > 60
            and not is_priv
            and dst_p in (80, 443, 8080, 8443)):
        return SignatureMatch(
            rule_id="NET-010", rule_name="C2 Beacon Pattern",
            attack_type="c2_beaconing", severity=4, confidence=0.72,
            description=f"Low-volume beacon: {pkts:.0f} pkts over {dur:.0f}s → port {dst_p}",
            mitre_tactic="Command and Control",
            mitre_technique="T1071 - Application Layer Protocol",
            evidence={"packets": pkts, "duration": dur, "dst_port": dst_p},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Process rules
# ─────────────────────────────────────────────────────────────────────────────

def _rule_suspicious_lineage(evt: dict) -> Optional[SignatureMatch]:
    enr = evt.get("enrichment") or {}
    if enr.get("suspicious_lineage"):
        actor  = _get(evt, "actor", "process") or {}
        parent = evt.get("process") or {}
        return SignatureMatch(
            rule_id="PROC-001", rule_name="Suspicious Process Lineage",
            attack_type="suspicious_lineage", severity=4, confidence=0.88,
            description=(f"Suspicious parent→child: "
                         f"{_str(parent.get('parent_name'))} → "
                         f"{_str(actor.get('name'))}"),
            mitre_tactic="Execution",
            mitre_technique="T1059 - Command and Scripting Interpreter",
            evidence={
                "parent":  _str(parent.get("parent_name")),
                "child":   _str(actor.get("name")),
                "cmdline": _str(actor.get("cmd_line")),
            },
        )


def _rule_shell_exec_in_cmdline(evt: dict) -> Optional[SignatureMatch]:
    actor   = _get(evt, "actor", "process") or {}
    raw_cmd = actor.get("cmd_line")
    cmdline = _str(raw_cmd).lower()          # safe: _str handles None
    hits    = [b for b in _T["shell_patterns"] if b in cmdline]
    if hits:
        return SignatureMatch(
            rule_id="PROC-002", rule_name="Shell Code Execution",
            attack_type="code_execution", severity=5, confidence=0.92,
            description=f"Dangerous command pattern: {hits[0]}",
            mitre_tactic="Execution",
            mitre_technique="T1059.004 - Unix Shell",
            evidence={"pattern": hits[0], "cmdline_snippet": cmdline[:120]},
        )


def _rule_root_child_spawn(evt: dict) -> Optional[SignatureMatch]:
    actor    = _get(evt, "actor", "process") or {}
    username = _str(_get(actor, "user", "name"))
    name     = _str(actor.get("name"))
    if username in ("root", "SYSTEM", "Administrator") and name in _T["suspicious_children"]:
        return SignatureMatch(
            rule_id="PROC-003", rule_name="Root Spawned Shell",
            attack_type="privilege_abuse", severity=5, confidence=0.90,
            description=f"Root/SYSTEM spawned {name}",
            mitre_tactic="Privilege Escalation",
            mitre_technique="T1548 - Abuse Elevation Control Mechanism",
            evidence={"username": username, "process": name},
        )


def _rule_credential_dump(evt: dict) -> Optional[SignatureMatch]:
    actor   = _get(evt, "actor", "process") or {}
    raw_cmd = actor.get("cmd_line")
    cmdline = _str(raw_cmd).lower()          # safe: _str handles None
    hits    = [c for c in _T["cred_patterns"] if c in cmdline]
    if hits:
        return SignatureMatch(
            rule_id="PROC-004", rule_name="Credential Dumping",
            attack_type="credential_dumping", severity=5, confidence=0.95,
            description=f"Credential dump pattern: {hits[0]}",
            mitre_tactic="Credential Access",
            mitre_technique="T1003 - OS Credential Dumping",
            evidence={"pattern": hits[0]},
        )


# ─────────────────────────────────────────────────────────────────────────────
# File rules
# ─────────────────────────────────────────────────────────────────────────────

def _rule_ransomware_entropy(evt: dict) -> Optional[SignatureMatch]:
    enr     = evt.get("enrichment") or {}
    entropy = _float(enr.get("entropy"))
    if enr.get("ransomware_indicator") and entropy >= _T["high_entropy_threshold"]:
        path = _str(_get(evt, "file", "path"))
        sev  = 5 if enr.get("ransomware_extension") else 4
        return SignatureMatch(
            rule_id="FILE-001", rule_name="Ransomware Encryption Pattern",
            attack_type="ransomware", severity=sev, confidence=0.91,
            description=f"High-entropy write H={entropy:.2f} → {path}",
            mitre_tactic="Impact",
            mitre_technique="T1486 - Data Encrypted for Impact",
            evidence={"entropy": entropy, "path": path,
                      "ransomware_ext": bool(enr.get("ransomware_extension"))},
        )


def _rule_ransomware_extension(evt: dict) -> Optional[SignatureMatch]:
    enr = evt.get("enrichment") or {}
    if enr.get("ransomware_extension") and not enr.get("ransomware_indicator"):
        path = _str(_get(evt, "file", "path"))
        return SignatureMatch(
            rule_id="FILE-002", rule_name="Ransomware Extension",
            attack_type="ransomware_ext", severity=4, confidence=0.85,
            description=f"Known ransomware extension: {path}",
            mitre_tactic="Impact",
            mitre_technique="T1486 - Data Encrypted for Impact",
            evidence={"path": path},
        )


def _rule_shadow_copy_delete(evt: dict) -> Optional[SignatureMatch]:
    path = _str(_get(evt, "file", "path")).lower()
    if any(s in path for s in ("shadow", "vssadmin", "wbadmin")):
        return SignatureMatch(
            rule_id="FILE-003", rule_name="Shadow Copy Deletion",
            attack_type="shadow_copy_deletion", severity=5, confidence=0.93,
            description=f"Backup/shadow copy deletion: {path}",
            mitre_tactic="Impact",
            mitre_technique="T1490 - Inhibit System Recovery",
            evidence={"path": path},
        )


def _rule_sensitive_file_access(evt: dict) -> Optional[SignatureMatch]:
    path = _str(_get(evt, "file", "path")).lower()
    hits = [s for s in _T["sensitive_paths"] if s in path]
    if hits:
        return SignatureMatch(
            rule_id="FILE-004", rule_name="Sensitive File Access",
            attack_type="sensitive_file_access", severity=4, confidence=0.82,
            description=f"Sensitive file accessed: {hits[0]}",
            mitre_tactic="Credential Access",
            mitre_technique="T1552 - Unsecured Credentials",
            evidence={"path": path, "pattern": hits[0]},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cloud rules
# ─────────────────────────────────────────────────────────────────────────────

def _rule_cloud_critical_action(evt: dict) -> Optional[SignatureMatch]:
    action = _str(_get(evt, "api", "operation"))
    if action in _T["cloud_critical_actions"]:
        user = _str(_get(evt, "actor", "user", "name"))
        return SignatureMatch(
            rule_id="CLOUD-001", rule_name="Critical Cloud Action",
            attack_type="cloud_defense_evasion", severity=5, confidence=0.95,
            description=f"Critical cloud action {action} by {user}",
            mitre_tactic="Defense Evasion",
            mitre_technique="T1562 - Impair Defenses",
            evidence={"action": action, "user": user},
        )


def _rule_cloud_high_privilege(evt: dict) -> Optional[SignatureMatch]:
    action  = _str(_get(evt, "api", "operation"))
    enr     = evt.get("enrichment") or {}
    if action in _T["cloud_high_actions"] and enr.get("high_privilege_action"):
        user = _str(_get(evt, "actor", "user", "name"))
        return SignatureMatch(
            rule_id="CLOUD-002", rule_name="High Privilege Cloud Action",
            attack_type="cloud_privilege_escalation", severity=4, confidence=0.85,
            description=f"High-privilege action {action} by {user}",
            mitre_tactic="Privilege Escalation",
            mitre_technique="T1078 - Valid Accounts",
            evidence={"action": action, "user": user},
        )


def _rule_cloud_access_denied(evt: dict) -> Optional[SignatureMatch]:
    err = _str(_get(evt, "api", "error"))
    if err == "AccessDenied":
        return SignatureMatch(
            rule_id="CLOUD-003", rule_name="Cloud Access Denied",
            attack_type="cloud_enumeration", severity=2, confidence=0.60,
            description="AccessDenied — possible enumeration",
            mitre_tactic="Discovery",
            mitre_technique="T1526 - Cloud Service Discovery",
            evidence={"error": err},
        )


def _rule_cloud_external_unknown_user(evt: dict) -> Optional[SignatureMatch]:
    enr   = evt.get("enrichment") or {}
    user  = _str(_get(evt, "actor", "user", "name"))
    is_ext = not enr.get("is_private", True)
    if is_ext and "unknown" in user.lower():
        return SignatureMatch(
            rule_id="CLOUD-004", rule_name="Unknown External Cloud Actor",
            attack_type="cloud_unauthorized_access", severity=4, confidence=0.80,
            description=f"Unknown external user '{user}' accessing cloud API",
            mitre_tactic="Initial Access",
            mitre_technique="T1078.004 - Cloud Accounts",
            evidence={"user": user,
                      "src_ip": _str(_get(evt, "src_endpoint", "ip"))},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rule registry — maps OCSF class → list of rule functions
# ─────────────────────────────────────────────────────────────────────────────

RULE_REGISTRY: dict[str, List[Callable]] = {
    "network_activity": [
        _rule_port_scan, _rule_syn_flood, _rule_udp_flood,
        _rule_dns_amplification, _rule_ssh_brute_force,
        _rule_smb_lateral, _rule_rdp_access, _rule_telnet,
        _rule_threat_intel_hit, _rule_c2_beacon,
    ],
    "process_activity": [
        _rule_suspicious_lineage, _rule_shell_exec_in_cmdline,
        _rule_root_child_spawn, _rule_credential_dump,
    ],
    "file_activity": [
        _rule_ransomware_entropy, _rule_ransomware_extension,
        _rule_shadow_copy_delete, _rule_sensitive_file_access,
    ],
    "cloud_api": [
        _rule_cloud_critical_action, _rule_cloud_high_privilege,
        _rule_cloud_access_denied, _rule_cloud_external_unknown_user,
    ],
    "network_conn": [
        _rule_threat_intel_hit,
    ],
}


def run_signature_engine(evt: dict) -> List[SignatureMatch]:
    """
    Run all applicable rules against a normalized OCSF event.
    Returns list of SignatureMatch (empty = no hits).
    Never raises — exceptions are logged and swallowed per rule.
    """
    if not isinstance(evt, dict):
        return []
    cls   = evt.get("ocsf_class", "")
    rules = RULE_REGISTRY.get(cls, [])
    hits: List[SignatureMatch] = []
    for rule_fn in rules:
        try:
            result = rule_fn(evt)
            if result is not None:
                hits.append(result)
        except Exception as e:
            log.error(f"[SIG] Rule {rule_fn.__name__} error: {e}")
    return hits
