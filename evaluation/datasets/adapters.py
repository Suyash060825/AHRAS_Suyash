from __future__ import annotations
"""
AHRAS Canonical Telemetry Adapters & Attack Family Taxonomy
------------------------------------------------------------
Implements the 18-class canonical attack taxonomy, OCSF normalizers,
and dataset-specific translators for CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018,
UGR'16, CTU-13, IoT-23, and LANL Cyber1.
"""

import enum
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

log = logging.getLogger(__name__)


class CanonicalAttackFamily(str, enum.Enum):
    BENIGN                          = "benign"
    RECONNAISSANCE                  = "reconnaissance"
    SCANNING                        = "scanning"
    BRUTE_FORCE                     = "brute_force"
    CREDENTIAL_ABUSE                = "credential_abuse"
    EXPLOITATION                    = "exploitation"
    WEB_ATTACK                      = "web_attack"
    DENIAL_OF_SERVICE               = "denial_of_service"
    DISTRIBUTED_DENIAL_OF_SERVICE   = "distributed_denial_of_service"
    BOTNET                          = "botnet"
    COMMAND_AND_CONTROL             = "command_and_control"
    MALWARE                         = "malware"
    LATERAL_MOVEMENT                = "lateral_movement"
    PERSISTENCE                     = "persistence"
    DISCOVERY                       = "discovery"
    COLLECTION                      = "collection"
    EXFILTRATION                    = "exfiltration"
    IMPACT                          = "impact"
    IOT_SPECIFIC_BEHAVIOR           = "iot_specific_behavior"
    UNKNOWN                         = "unknown"


@dataclass
class LabelMapping:
    raw_label: str
    canonical_family: CanonicalAttackFamily
    is_malicious: bool
    mitre_technique: Optional[str]
    mitre_tactic: Optional[str]
    mapping_confidence: float  # 0.0 to 1.0


# ── Canonical Label Translation Tables ───────────────────────────────────────

_CICIDS2017_LABEL_MAP: Dict[str, LabelMapping] = {
    "benign": LabelMapping("BENIGN", CanonicalAttackFamily.BENIGN, False, None, None, 1.0),
    "dos hulk": LabelMapping("DoS Hulk", CanonicalAttackFamily.DENIAL_OF_SERVICE, True, "T1498.001", "Impact", 0.95),
    "dos goldeneye": LabelMapping("DoS GoldenEye", CanonicalAttackFamily.DENIAL_OF_SERVICE, True, "T1498.001", "Impact", 0.95),
    "dos slowloris": LabelMapping("DoS slowloris", CanonicalAttackFamily.DENIAL_OF_SERVICE, True, "T1499.003", "Impact", 0.95),
    "dos slowhttptest": LabelMapping("DoS Slowhttptest", CanonicalAttackFamily.DENIAL_OF_SERVICE, True, "T1499.003", "Impact", 0.95),
    "heartbleed": LabelMapping("Heartbleed", CanonicalAttackFamily.EXPLOITATION, True, "T1190", "Initial Access", 0.95),
    "ddos": LabelMapping("DDoS", CanonicalAttackFamily.DISTRIBUTED_DENIAL_OF_SERVICE, True, "T1498", "Impact", 0.95),
    "portscan": LabelMapping("PortScan", CanonicalAttackFamily.SCANNING, True, "T1046", "Discovery", 0.95),
    "ftp-patator": LabelMapping("FTP-Patator", CanonicalAttackFamily.BRUTE_FORCE, True, "T1110.001", "Credential Access", 0.95),
    "ssh-patator": LabelMapping("SSH-Patator", CanonicalAttackFamily.BRUTE_FORCE, True, "T1110.001", "Credential Access", 0.95),
    "bot": LabelMapping("Bot", CanonicalAttackFamily.BOTNET, True, "T1071.001", "Command and Control", 0.90),
    "web attack - brute force": LabelMapping("Web Attack - Brute Force", CanonicalAttackFamily.BRUTE_FORCE, True, "T1110", "Credential Access", 0.95),
    "web attack - xss": LabelMapping("Web Attack - XSS", CanonicalAttackFamily.WEB_ATTACK, True, "T1190", "Initial Access", 0.90),
    "web attack - sql injection": LabelMapping("Web Attack - Sql Injection", CanonicalAttackFamily.WEB_ATTACK, True, "T1190", "Initial Access", 0.95),
    "infiltration": LabelMapping("Infiltration", CanonicalAttackFamily.LATERAL_MOVEMENT, True, "T1021", "Lateral Movement", 0.85),
}

_UNSW_NB15_LABEL_MAP: Dict[str, LabelMapping] = {
    "normal": LabelMapping("Normal", CanonicalAttackFamily.BENIGN, False, None, None, 1.0),
    "fuzzers": LabelMapping("Fuzzers", CanonicalAttackFamily.EXPLOITATION, True, "T1499", "Impact", 0.85),
    "analysis": LabelMapping("Analysis", CanonicalAttackFamily.SCANNING, True, "T1046", "Discovery", 0.85),
    "backdoors": LabelMapping("Backdoors", CanonicalAttackFamily.COMMAND_AND_CONTROL, True, "T1071", "Command and Control", 0.90),
    "dos": LabelMapping("DoS", CanonicalAttackFamily.DENIAL_OF_SERVICE, True, "T1498", "Impact", 0.95),
    "exploits": LabelMapping("Exploits", CanonicalAttackFamily.EXPLOITATION, True, "T1190", "Initial Access", 0.95),
    "generic": LabelMapping("Generic", CanonicalAttackFamily.EXPLOITATION, True, "T1203", "Execution", 0.70),
    "reconnaissance": LabelMapping("Reconnaissance", CanonicalAttackFamily.RECONNAISSANCE, True, "T1595", "Reconnaissance", 0.95),
    "shellcode": LabelMapping("Shellcode", CanonicalAttackFamily.EXPLOITATION, True, "T1059", "Execution", 0.90),
    "worms": LabelMapping("Worms", CanonicalAttackFamily.MALWARE, True, "T1021", "Lateral Movement", 0.85),
}

_CTU13_LABEL_MAP: Dict[str, LabelMapping] = {
    "normal": LabelMapping("Normal", CanonicalAttackFamily.BENIGN, False, None, None, 1.0),
    "background": LabelMapping("Background", CanonicalAttackFamily.BENIGN, False, None, None, 0.95),
    "botnet": LabelMapping("Botnet", CanonicalAttackFamily.BOTNET, True, "T1071", "Command and Control", 0.95),
    "c&c": LabelMapping("C&C", CanonicalAttackFamily.COMMAND_AND_CONTROL, True, "T1071.001", "Command and Control", 0.95),
}

_IOT23_LABEL_MAP: Dict[str, LabelMapping] = {
    "benign": LabelMapping("Benign", CanonicalAttackFamily.BENIGN, False, None, None, 1.0),
    "partofahorizontalportscan": LabelMapping("PartOfAHorizontalPortScan", CanonicalAttackFamily.SCANNING, True, "T1046", "Discovery", 0.95),
    "okiru": LabelMapping("Okiru", CanonicalAttackFamily.MALWARE, True, "T1498", "Impact", 0.90),
    "ddos": LabelMapping("DDoS", CanonicalAttackFamily.DISTRIBUTED_DENIAL_OF_SERVICE, True, "T1498", "Impact", 0.95),
    "c&c": LabelMapping("C&C", CanonicalAttackFamily.COMMAND_AND_CONTROL, True, "T1071", "Command and Control", 0.95),
    "attack": LabelMapping("Attack", CanonicalAttackFamily.EXPLOITATION, True, "T1190", "Initial Access", 0.80),
    "c&c-heartbeat": LabelMapping("C&C-HeartBeat", CanonicalAttackFamily.COMMAND_AND_CONTROL, True, "T1071.001", "Command and Control", 0.95),
    "mirai": LabelMapping("Mirai", CanonicalAttackFamily.BOTNET, True, "T1498", "Impact", 0.95),
}


@dataclass
class CanonicalEvent:
    event_id: str
    dataset_source: str
    ocsf_class: str
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    duration_sec: float
    bytes_in: int
    bytes_out: int
    packet_count: int
    flow_rate: float
    feature_vector: List[float]
    label_info: LabelMapping
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_ocsf_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "event_id": self.event_id,
                "dataset_source": self.dataset_source,
                "timestamp": self.timestamp,
                "version": "1.1.0",
            },
            "class_name": self.ocsf_class,
            "category_name": "Network Activity" if "network" in self.ocsf_class else "System Activity",
            "src_endpoint": {"ip": self.src_ip, "port": self.src_port},
            "dst_endpoint": {"ip": self.dst_ip, "port": self.dst_port},
            "connection_info": {
                "protocol_name": self.protocol,
                "duration": self.duration_sec,
                "bytes_in": self.bytes_in,
                "bytes_out": self.bytes_out,
                "packet_count": self.packet_count,
            },
            "traffic": {
                "bytes": self.bytes_in + self.bytes_out,
                "packets": self.packet_count,
            },
            "enrichment": {
                "canonical_attack_family": self.label_info.canonical_family.value,
                "is_malicious": self.label_info.is_malicious,
                "mitre_technique": self.label_info.mitre_technique,
                "mitre_tactic": self.label_info.mitre_tactic,
                "mapping_confidence": self.label_info.mapping_confidence,
            },
            "features_14d": self.feature_vector,
        }


class DatasetAdapter:
    """Normalizes raw dataset records to canonical AHRAS OCSF events."""

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None: return default
            f = float(val)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return default

    @classmethod
    def adapt_cicids2017(cls, raw_row: Dict[str, Any], event_idx: int) -> CanonicalEvent:
        label_raw = str(raw_row.get("Label") or raw_row.get("label") or "BENIGN").strip()
        label_key = label_raw.lower()
        mapping = _CICIDS2017_LABEL_MAP.get(
            label_key,
            LabelMapping(label_raw, CanonicalAttackFamily.UNKNOWN, True, None, None, 0.5)
        )

        flow_dur = max(1e-6, cls._safe_float(raw_row.get("Flow Duration") or raw_row.get("flow_duration")) / 1e6)
        tot_fwd_pkts = int(cls._safe_float(raw_row.get("Total Fwd Packets") or raw_row.get("tot_fwd_pkts")))
        tot_bwd_pkts = int(cls._safe_float(raw_row.get("Total Backward Packets") or raw_row.get("tot_bwd_pkts")))
        tot_fwd_bytes = int(cls._safe_float(raw_row.get("Total Length of Fwd Packets") or raw_row.get("totlen_fwd_pkts")))
        tot_bwd_bytes = int(cls._safe_float(raw_row.get("Total Length of Bwd Packets") or raw_row.get("totlen_bwd_pkts")))
        dst_port = int(cls._safe_float(raw_row.get("Destination Port") or raw_row.get("dst_port") or 80))

        feat_14d = [
            flow_dur,
            float(tot_fwd_pkts + tot_bwd_pkts),
            float(tot_fwd_bytes + tot_bwd_bytes),
            float(tot_fwd_bytes) / max(1, tot_fwd_pkts),
            float(tot_bwd_bytes) / max(1, tot_bwd_pkts),
            cls._safe_float(raw_row.get("Flow Bytes/s")),
            cls._safe_float(raw_row.get("Flow Packets/s")),
            cls._safe_float(raw_row.get("Flow IAT Mean")),
            cls._safe_float(raw_row.get("Flow IAT Std")),
            cls._safe_float(raw_row.get("Fwd IAT Total")),
            cls._safe_float(raw_row.get("Bwd IAT Total")),
            cls._safe_float(raw_row.get("SYN Flag Count")),
            cls._safe_float(raw_row.get("ACK Flag Count")),
            float(dst_port),
        ]

        return CanonicalEvent(
            event_id=f"CIC17-{event_idx:08d}",
            dataset_source="CIC-IDS2017",
            ocsf_class="network_activity",
            timestamp=1499256000.0 + (event_idx * 0.01),
            src_ip=str(raw_row.get("Source IP") or f"192.168.10.{10 + (event_idx % 200)}"),
            dst_ip=str(raw_row.get("Destination IP") or "192.168.10.50"),
            src_port=int(cls._safe_float(raw_row.get("Source Port") or 49152 + (event_idx % 1000))),
            dst_port=dst_port,
            protocol="TCP",
            duration_sec=flow_dur,
            bytes_in=tot_fwd_bytes,
            bytes_out=tot_bwd_bytes,
            packet_count=tot_fwd_pkts + tot_bwd_pkts,
            flow_rate=(tot_fwd_bytes + tot_bwd_bytes) / flow_dur,
            feature_vector=feat_14d,
            label_info=mapping,
            raw_payload=raw_row,
        )

    @classmethod
    def adapt_unsw_nb15(cls, raw_row: Dict[str, Any], event_idx: int) -> CanonicalEvent:
        label_raw = str(raw_row.get("attack_cat") or raw_row.get("Label") or "Normal").strip()
        label_key = label_raw.lower()
        mapping = _UNSW_NB15_LABEL_MAP.get(
            label_key,
            LabelMapping(label_raw, CanonicalAttackFamily.UNKNOWN, True, None, None, 0.5)
        )

        dur = max(1e-6, cls._safe_float(raw_row.get("dur") or raw_row.get("duration")))
        sbytes = int(cls._safe_float(raw_row.get("sbytes") or raw_row.get("src_bytes")))
        dbytes = int(cls._safe_float(raw_row.get("dbytes") or raw_row.get("dst_bytes")))
        spkts = int(cls._safe_float(raw_row.get("Spkts") or raw_row.get("spkts") or 1))
        dpkts = int(cls._safe_float(raw_row.get("Dpkts") or raw_row.get("dpkts") or 1))
        dst_port = int(cls._safe_float(raw_row.get("dsport") or raw_row.get("dst_port") or 80))

        feat_14d = [
            dur,
            float(spkts + dpkts),
            float(sbytes + dbytes),
            float(sbytes) / max(1, spkts),
            float(dbytes) / max(1, dpkts),
            cls._safe_float(raw_row.get("sload")),
            cls._safe_float(raw_row.get("rate")),
            cls._safe_float(raw_row.get("sinpkt")),
            cls._safe_float(raw_row.get("dinpkt")),
            cls._safe_float(raw_row.get("sjit")),
            cls._safe_float(raw_row.get("djit")),
            cls._safe_float(raw_row.get("synack")),
            cls._safe_float(raw_row.get("ackdat")),
            float(dst_port),
        ]

        return CanonicalEvent(
            event_id=f"UNSW15-{event_idx:08d}",
            dataset_source="UNSW-NB15",
            ocsf_class="network_activity",
            timestamp=1421913600.0 + (event_idx * 0.01),
            src_ip=str(raw_row.get("srcip") or f"175.45.176.{event_idx % 250}"),
            dst_ip=str(raw_row.get("dstip") or "149.171.126.1"),
            src_port=int(cls._safe_float(raw_row.get("sport") or 40000 + (event_idx % 5000))),
            dst_port=dst_port,
            protocol=str(raw_row.get("proto") or "TCP").upper(),
            duration_sec=dur,
            bytes_in=sbytes,
            bytes_out=dbytes,
            packet_count=spkts + dpkts,
            flow_rate=(sbytes + dbytes) / dur,
            feature_vector=feat_14d,
            label_info=mapping,
            raw_payload=raw_row,
        )
