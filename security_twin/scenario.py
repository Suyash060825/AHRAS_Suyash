from __future__ import annotations
"""
AHRAS Module / Digital Twin — Attack Scenario & Event Generation Engine
-----------------------------------------------------------------------
Builds ordered multi-stage attack scenarios across the cyber kill chain:
  RECON -> INITIAL_ACCESS -> EXECUTION -> PERSISTENCE -> CREDENTIAL_ACCESS ->
  LATERAL_MOVEMENT -> COLLECTION -> EXFILTRATION / IMPACT.

Emits OCSF-compliant telemetry events and enforces precondition/postcondition
dependencies against the digital twin topology.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from security_twin.models import (
    AttackScenario, AttackStep, AttackStage, Host, User, Process, File, Network
)
from security_twin.state import SecurityTwin


def check_step_preconditions(step: AttackStep, twin: SecurityTwin) -> Tuple[bool, str]:
    """
    Evaluates whether an attack step's preconditions are satisfied in the current twin state.
    Returns (True, "OK") or (False, reason).
    """
    pre = step.preconditions

    # 1. Source IP check
    src_ip = pre.get("src_ip")
    if src_ip and twin.is_ip_blocked(src_ip):
        return False, f"Source IP {src_ip} is blocked by perimeter control"

    # 2. Host isolation check
    src_host = pre.get("src_host")
    if src_host and twin.is_host_isolated(src_host):
        return False, f"Source host {src_host} is isolated from network"

    dst_host = pre.get("dst_host")
    if dst_host and twin.is_host_isolated(dst_host):
        return False, f"Destination host {dst_host} is isolated from network"

    # 3. Process execution check
    pid = pre.get("pid")
    if pid is not None and twin.is_process_terminated(pid):
        return False, f"Required process PID {pid} has been terminated"

    # 4. Identity / Token revocation check
    token_id = pre.get("token_id")
    if token_id and twin.is_token_revoked(token_id):
        return False, f"Authentication token {token_id} has been revoked"

    # 5. Required compromised host check
    req_compromise = pre.get("min_compromise_level")
    if req_compromise and src_host:
        h = twin.get_host(src_host)
        if h and h.compromise_level < req_compromise:
            return False, f"Host {src_host} compromise level {h.compromise_level} < required {req_compromise}"

    return True, "Preconditions satisfied"


def apply_step_postconditions(step: AttackStep, twin: SecurityTwin) -> None:
    """Applies the state transitions of a successfully executed attack step to the digital twin."""
    post = step.postconditions

    # 1. Host compromise level update
    target_host_id = post.get("compromise_host")
    if target_host_id:
        h = twin.get_host(target_host_id)
        if h:
            h.compromise_level = min(1.0, h.compromise_level + post.get("compromise_delta", 0.3))

    # 2. Process creation
    spawn_proc = post.get("spawn_process")
    if isinstance(spawn_proc, dict):
        new_pid = spawn_proc.get("pid", 9000 + len(twin.processes))
        p = Process(
            pid=new_pid,
            process_name=spawn_proc.get("name", "malware.elf"),
            host_id=spawn_proc.get("host_id", target_host_id or "host-01"),
            user_id=spawn_proc.get("user_id", "u-01"),
            cmdline=spawn_proc.get("cmdline", ""),
            parent_pid=spawn_proc.get("parent_pid"),
            is_terminated=False,
        )
        twin.add_process(p)

    # 3. File encryption / entropy modification
    enc_file_path = post.get("encrypt_file")
    if enc_file_path and enc_file_path in twin.files:
        f = twin.files[enc_file_path]
        f.is_encrypted = True
        f.entropy = post.get("entropy", 7.95)

    # 4. Update risk state in twin
    entity_key = post.get("entity_key") or step.destination
    if entity_key:
        twin.risk_state[entity_key] = min(1.0, twin.risk_state.get(entity_key, 0.2) + post.get("risk_delta", 0.25))


def emit_ocsf_event(step: AttackStep, twin: SecurityTwin) -> Dict[str, Any]:
    """
    Translates an attack step into an OCSF-compliant dictionary event,
    compatible with the AHRAS normalizer and evidence ledger.
    """
    ev = step.expected_evidence
    cls_id = ev.get("ocsf_class_id", 1001)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(step.timestamp))

    if cls_id == 1001:  # network_activity
        return {
            "ocsf_class_id": 1001,
            "ocsf_class": "network_activity",
            "event_id": f"ev-{uuid.uuid4().hex[:10]}",
            "time": now_iso,
            "severity_id": ev.get("severity_id", 3),
            "src_endpoint": {
                "ip": step.source,
                "port": ev.get("src_port", 44120),
                "geo": {"country": "Unknown", "city": "Remote"},
            },
            "dst_endpoint": {
                "ip": step.destination,
                "port": ev.get("dst_port", 80),
                "geo": {"country": "US", "city": "Internal"},
            },
            "protocol": ev.get("protocol", "TCP"),
            "traffic": {
                "packets": ev.get("packets", 150),
                "bytes": ev.get("bytes", 8400),
                "duration_sec": ev.get("duration_sec", 1.5),
            },
            "tcp_flags": ev.get("tcp_flags", ["SYN"]),
            "enrichment": {
                "mitre_technique": step.technique,
                "mitre_name": step.technique_name,
                "stage": step.stage.value,
                "is_threat_intel_hit": ev.get("is_threat_intel_hit", False),
            },
            "raw_source": "security_twin_sim",
        }

    elif cls_id == 1002:  # process_activity
        proc_name = ev.get("process_name", "bash")
        return {
            "ocsf_class_id": 1002,
            "ocsf_class": "process_activity",
            "event_id": f"ev-{uuid.uuid4().hex[:10]}",
            "time": now_iso,
            "severity_id": ev.get("severity_id", 3),
            "actor": {
                "process": {
                    "pid": ev.get("pid", 2048),
                    "name": proc_name,
                    "exe": f"/usr/bin/{proc_name}",
                    "cmd_line": ev.get("cmdline", f"{proc_name} -c 'malicious payload'"),
                    "user": {"name": ev.get("username", "root")},
                }
            },
            "process": {
                "parent_pid": ev.get("parent_pid", 1024),
                "parent_name": ev.get("parent_name", "systemd"),
            },
            "device": {"hostname": step.destination},
            "enrichment": {
                "suspicious_lineage": ev.get("suspicious_lineage", True),
                "mitre_technique": step.technique,
                "mitre_name": step.technique_name,
            },
            "raw_source": "security_twin_sim",
        }

    elif cls_id == 1003:  # file_activity
        return {
            "ocsf_class_id": 1003,
            "ocsf_class": "file_activity",
            "event_id": f"ev-{uuid.uuid4().hex[:10]}",
            "time": now_iso,
            "severity_id": ev.get("severity_id", 4),
            "file": {
                "path": ev.get("file_path", "/var/data/confidential.db.locked"),
                "entropy": ev.get("entropy", 7.92),
                "size": ev.get("size", 10485760),
            },
            "device": {"hostname": step.destination},
            "enrichment": {
                "ransomware_burst": ev.get("ransomware_burst", True),
                "mitre_technique": step.technique,
            },
            "raw_source": "security_twin_sim",
        }

    else:  # 4001 cloud_api or generic
        return {
            "ocsf_class_id": 4001,
            "ocsf_class": "cloud_api",
            "event_id": f"ev-{uuid.uuid4().hex[:10]}",
            "time": now_iso,
            "severity_id": ev.get("severity_id", 3),
            "actor": {"user": {"name": ev.get("username", "admin_service")}},
            "api": {
                "operation": ev.get("api_operation", "AssumeRoleWithWebIdentity"),
                "service": ev.get("api_service", "iam.amazonaws.com"),
            },
            "enrichment": {
                "mitre_technique": step.technique,
                "token_id": ev.get("token_id", ""),
            },
            "raw_source": "security_twin_sim",
        }


# ── Scenario Builders ─────────────────────────────────────────────────────────

def build_ransomware_burst_scenario(
    attacker_ip: str = "203.0.113.55",
    web_host_id: str = "host-web-01",
    file_host_id: str = "host-file-01",
    base_time: float = 1700000000.0,
) -> AttackScenario:
    """Multi-stage Ransomware Burst scenario from reconnaissance to data encryption."""
    steps = [
        AttackStep(
            step_id="step-1-recon",
            stage=AttackStage.RECON,
            timestamp=base_time + 10.0,
            source=attacker_ip,
            destination=web_host_id,
            technique="T1046",
            technique_name="Network Service Discovery",
            preconditions={"src_ip": attacker_ip},
            postconditions={"risk_delta": 0.15},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 80, "tcp_flags": ["SYN"], "severity_id": 1},
        ),
        AttackStep(
            step_id="step-2-exploit",
            stage=AttackStage.INITIAL_ACCESS,
            timestamp=base_time + 35.0,
            source=attacker_ip,
            destination=web_host_id,
            technique="T1190",
            technique_name="Exploit Public-Facing Application",
            preconditions={"src_ip": attacker_ip, "dst_host": web_host_id},
            postconditions={"compromise_host": web_host_id, "compromise_delta": 0.35, "risk_delta": 0.30},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 80, "is_threat_intel_hit": True, "severity_id": 4},
        ),
        AttackStep(
            step_id="step-3-spawn",
            stage=AttackStage.EXECUTION,
            timestamp=base_time + 45.0,
            source=web_host_id,
            destination=web_host_id,
            technique="T1059.004",
            technique_name="Unix Shell Execution",
            preconditions={"src_host": web_host_id, "min_compromise_level": 0.2},
            postconditions={
                "spawn_process": {"pid": 7720, "name": "sh", "host_id": web_host_id, "parent_pid": 1024},
                "risk_delta": 0.25,
            },
            expected_evidence={"ocsf_class_id": 1002, "pid": 7720, "process_name": "sh", "parent_name": "nginx", "severity_id": 3},
        ),
        AttackStep(
            step_id="step-4-lateral",
            stage=AttackStage.LATERAL_MOVEMENT,
            timestamp=base_time + 75.0,
            source=web_host_id,
            destination=file_host_id,
            technique="T1021.004",
            technique_name="Remote Services: SSH",
            preconditions={"src_host": web_host_id, "dst_host": file_host_id, "pid": 7720},
            postconditions={"compromise_host": file_host_id, "compromise_delta": 0.40, "risk_delta": 0.30},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 22, "severity_id": 3},
        ),
        AttackStep(
            step_id="step-5-encrypt",
            stage=AttackStage.EXFILTRATION,
            timestamp=base_time + 110.0,
            source=file_host_id,
            destination=file_host_id,
            technique="T1486",
            technique_name="Data Encrypted for Impact",
            preconditions={"src_host": file_host_id, "min_compromise_level": 0.3},
            postconditions={"encrypt_file": "/var/data/finance.db", "entropy": 7.96, "risk_delta": 0.40},
            expected_evidence={"ocsf_class_id": 1003, "file_path": "/var/data/finance.db", "entropy": 7.96, "severity_id": 4},
        ),
    ]

    return AttackScenario(
        scenario_id="scenario-ransomware-01",
        name="Targeted Enterprise Ransomware Burst",
        description="Reconnaissance, initial web exploit, shell execution, lateral movement, and high-entropy encryption.",
        steps=steps,
    )


def build_lateral_movement_scenario(
    attacker_ip: str = "198.51.100.12",
    bastion_host_id: str = "host-bastion-01",
    db_host_id: str = "host-db-prod-01",
    base_time: float = 1700000000.0,
) -> AttackScenario:
    """Multi-stage SSH Bruteforce and Lateral Movement traversal scenario."""
    steps = [
        AttackStep(
            step_id="step-1-bruteforce",
            stage=AttackStage.INITIAL_ACCESS,
            timestamp=base_time + 15.0,
            source=attacker_ip,
            destination=bastion_host_id,
            technique="T1110.001",
            technique_name="SSH Password Guessing",
            preconditions={"src_ip": attacker_ip, "dst_host": bastion_host_id},
            postconditions={"compromise_host": bastion_host_id, "compromise_delta": 0.30, "risk_delta": 0.35},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 22, "severity_id": 3},
        ),
        AttackStep(
            step_id="step-2-cred-dump",
            stage=AttackStage.CREDENTIAL_ACCESS,
            timestamp=base_time + 40.0,
            source=bastion_host_id,
            destination=bastion_host_id,
            technique="T1003.008",
            technique_name="/etc/passwd and /etc/shadow Theft",
            preconditions={"src_host": bastion_host_id, "min_compromise_level": 0.2},
            postconditions={"risk_delta": 0.20},
            expected_evidence={"ocsf_class_id": 1002, "pid": 8110, "process_name": "cat", "severity_id": 2},
        ),
        AttackStep(
            step_id="step-3-lateral-traversal",
            stage=AttackStage.LATERAL_MOVEMENT,
            timestamp=base_time + 70.0,
            source=bastion_host_id,
            destination=db_host_id,
            technique="T1021.004",
            technique_name="SSH Lateral Traversal to DB",
            preconditions={"src_host": bastion_host_id, "dst_host": db_host_id},
            postconditions={"compromise_host": db_host_id, "compromise_delta": 0.45, "risk_delta": 0.35},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 22, "severity_id": 4},
        ),
        AttackStep(
            step_id="step-4-exfil",
            stage=AttackStage.EXFILTRATION,
            timestamp=base_time + 105.0,
            source=db_host_id,
            destination=attacker_ip,
            technique="T1041",
            technique_name="Exfiltration Over C2 Channel",
            preconditions={"src_host": db_host_id, "src_ip": attacker_ip},
            postconditions={"risk_delta": 0.40},
            expected_evidence={"ocsf_class_id": 1001, "dst_port": 443, "severity_id": 4},
        ),
    ]

    return AttackScenario(
        scenario_id="scenario-lateral-01",
        name="Bastion Compromise & Database Lateral Movement",
        description="SSH bruteforce on perimeter bastion host, credential harvesting, and database lateral access.",
        steps=steps,
    )


def build_credential_abuse_scenario(
    user_id: str = "usr-ops-99",
    token_id: str = "tok-jwt-ops-99",
    cloud_host_id: str = "host-cloud-mgmt",
    base_time: float = 1700000000.0,
) -> AttackScenario:
    """Multi-stage Identity Token theft and Cloud Privilege Escalation scenario."""
    steps = [
        AttackStep(
            step_id="step-1-token-theft",
            stage=AttackStage.CREDENTIAL_ACCESS,
            timestamp=base_time + 20.0,
            source=user_id,
            destination=cloud_host_id,
            technique="T1528",
            technique_name="Steal Application Access Token",
            preconditions={"token_id": token_id},
            postconditions={"risk_delta": 0.25},
            expected_evidence={"ocsf_class_id": 4001, "api_operation": "GetSessionToken", "token_id": token_id, "severity_id": 2},
        ),
        AttackStep(
            step_id="step-2-unauthorized-api",
            stage=AttackStage.EXECUTION,
            timestamp=base_time + 50.0,
            source=user_id,
            destination=cloud_host_id,
            technique="T1078.004",
            technique_name="Cloud Accounts Privilege Abuse",
            preconditions={"token_id": token_id},
            postconditions={"risk_delta": 0.35},
            expected_evidence={"ocsf_class_id": 4001, "api_operation": "AttachUserPolicy", "token_id": token_id, "severity_id": 4},
        ),
        AttackStep(
            step_id="step-3-cloud-impact",
            stage=AttackStage.COLLECTION,
            timestamp=base_time + 85.0,
            source=user_id,
            destination=cloud_host_id,
            technique="T1530",
            technique_name="Data from Cloud Storage Object",
            preconditions={"token_id": token_id},
            postconditions={"risk_delta": 0.40},
            expected_evidence={"ocsf_class_id": 4001, "api_operation": "GetObject", "token_id": token_id, "severity_id": 4},
        ),
    ]

    return AttackScenario(
        scenario_id="scenario-cred-abuse-01",
        name="Privileged Cloud Token Theft & Data Exfiltration",
        description="Compromised operations JWT token used for unauthorized policy escalation and S3 bucket access.",
        steps=steps,
    )
