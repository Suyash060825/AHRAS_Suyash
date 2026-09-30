"""
Behavior and attack profile definitions for scenario generation.
Provides realistic benign and adversarial traffic statistical distributions.
"""

from typing import Dict, Any, List
from .schema import AttackFamily

BENIGN_PROFILES: Dict[str, Dict[str, Any]] = {
    "standard_office_workstation": {
        "web_browsing_ratio": 0.65,
        "dns_query_rate_per_min": 45.0,
        "mean_flow_duration_sec": 4.5,
        "mean_bytes_out": 2400.0,
        "mean_bytes_in": 18500.0,
        "active_hours": (8, 18),
        "typical_protocols": ["TCP/443", "TCP/80", "UDP/53", "TCP/8443"],
    },
    "internal_database_server": {
        "web_browsing_ratio": 0.0,
        "dns_query_rate_per_min": 10.0,
        "mean_flow_duration_sec": 30.0,
        "mean_bytes_out": 120000.0,
        "mean_bytes_in": 15000.0,
        "active_hours": (0, 24),
        "typical_protocols": ["TCP/5432", "TCP/3306", "TCP/1433", "TCP/22"],
    },
    "cloud_microservice_node": {
        "web_browsing_ratio": 0.1,
        "dns_query_rate_per_min": 120.0,
        "mean_flow_duration_sec": 0.8,
        "mean_bytes_out": 4500.0,
        "mean_bytes_in": 4800.0,
        "active_hours": (0, 24),
        "typical_protocols": ["TCP/443", "TCP/8080", "TCP/9092", "TCP/6379"],
    },
    "ot_plc_controller": {
        "web_browsing_ratio": 0.0,
        "dns_query_rate_per_min": 1.0,
        "mean_flow_duration_sec": 12.0,
        "mean_bytes_out": 850.0,
        "mean_bytes_in": 850.0,
        "active_hours": (0, 24),
        "typical_protocols": ["TCP/502", "TCP/102", "TCP/44818"],
    },
}

ATTACK_PROFILES: Dict[AttackFamily, Dict[str, Any]] = {
    AttackFamily.RECONNAISSANCE: {
        "burst_rate_multiplier": 5.0,
        "destination_ip_dispersion": 0.85,
        "destination_port_dispersion": 0.95,
        "mean_packet_size": 64.0,
        "ttp": "T1595.001",
        "anomaly_indicators": ["high_port_variety", "failed_handshakes"],
    },
    AttackFamily.BRUTE_FORCE: {
        "burst_rate_multiplier": 15.0,
        "destination_ip_dispersion": 0.05,
        "destination_port_dispersion": 0.01,
        "mean_packet_size": 180.0,
        "ttp": "T1110",
        "anomaly_indicators": ["auth_failure_spike", "single_target_burst"],
    },
    AttackFamily.EXPLOITATION: {
        "burst_rate_multiplier": 2.0,
        "destination_ip_dispersion": 0.1,
        "destination_port_dispersion": 0.05,
        "mean_packet_size": 1450.0,
        "ttp": "T1190",
        "anomaly_indicators": ["payload_entropy_high", "abnormal_uri"],
    },
    AttackFamily.DENIAL_OF_SERVICE: {
        "burst_rate_multiplier": 50.0,
        "destination_ip_dispersion": 0.01,
        "destination_port_dispersion": 0.01,
        "mean_packet_size": 40.0,
        "ttp": "T1498",
        "anomaly_indicators": ["extreme_volumetric_rate", "incomplete_tcp_sessions"],
    },
    AttackFamily.LATERAL_MOVEMENT: {
        "burst_rate_multiplier": 3.5,
        "destination_ip_dispersion": 0.4,
        "destination_port_dispersion": 0.2,
        "mean_packet_size": 750.0,
        "ttp": "T1021",
        "anomaly_indicators": ["internal_subnet_traversal", "admin_share_access"],
    },
    AttackFamily.EXFILTRATION: {
        "burst_rate_multiplier": 8.0,
        "destination_ip_dispersion": 0.02,
        "destination_port_dispersion": 0.01,
        "mean_packet_size": 1500.0,
        "ttp": "T1048",
        "anomaly_indicators": ["asymmetric_outbound_volume", "high_entropy_dns_or_https"],
    },
    AttackFamily.COMMAND_AND_CONTROL: {
        "burst_rate_multiplier": 1.2,
        "destination_ip_dispersion": 0.01,
        "destination_port_dispersion": 0.01,
        "mean_packet_size": 256.0,
        "ttp": "T1071",
        "anomaly_indicators": ["beaconing_periodicity", "unusual_user_agent"],
    },
}
