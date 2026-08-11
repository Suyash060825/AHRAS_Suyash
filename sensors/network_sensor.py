"""
AHRAS Network Sensor
---------------------
Captures live network packets, aggregates them into flows,
and publishes raw flow records to the message bus.

Two modes:
  LIVE   → uses scapy for real packet capture (requires root + scapy)
  SIM    → generates realistic synthetic flows for dev/test

Flow aggregation:
  A flow = 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)
  Flows are flushed every FLOW_WINDOW_SECONDS seconds.
"""

import uuid
import time
import socket
import random
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    NETWORK_INTERFACE, FLOW_WINDOW_SECONDS, KAFKA_TOPIC_RAW,
)
from pipeline.bus import get_producer

log = logging.getLogger(__name__)

# ── Synthetic traffic profiles for dev/test ──────────────────────────────────
_NORMAL_PROFILES = [
    # (src_port, dst_port, proto, pkt_count_range, byte_range)
    (random.randint(49152, 65535), 443,  "TCP", (5, 50),    (500, 50000)),
    (random.randint(49152, 65535), 80,   "TCP", (3, 30),    (300, 30000)),
    (random.randint(49152, 65535), 53,   "UDP", (1, 5),     (60, 500)),
    (random.randint(49152, 65535), 25,   "TCP", (5, 15),    (500, 5000)),
    (random.randint(49152, 65535), 22,   "TCP", (10, 100),  (1000, 50000)),
]

_ATTACK_PROFILES = [
    # Port scan
    {"type": "port_scan",   "dst_ports": list(range(20, 1025, 7)), "proto": "TCP",
     "flags": ["SYN"], "pkts": 1, "bytes": 60},
    # SSH brute force
    {"type": "ssh_brute",   "dst_port": 22, "proto": "TCP",
     "flags": ["SYN", "ACK"], "pkts": 200, "bytes": 12000},
    # UDP flood
    {"type": "udp_flood",   "dst_port": 80, "proto": "UDP",
     "flags": [], "pkts": 5000, "bytes": 500000},
    # DNS amplification
    {"type": "dns_amp",     "dst_port": 53, "proto": "UDP",
     "flags": [], "pkts": 300, "bytes": 150000},
    # C2 beacon (low and slow)
    {"type": "c2_beacon",   "dst_port": 443, "proto": "TCP",
     "flags": ["SYN", "ACK"], "pkts": 2, "bytes": 256},
]

_INTERNAL_SUBNETS = ["10.0.0.", "192.168.1.", "172.16.0."]
_EXTERNAL_IPS = [
    "45.33.32.156", "198.51.100.23", "203.0.113.5",
    "1.1.1.1", "8.8.8.8", "104.21.64.1",
    "185.220.101.5",   # known Tor exit (for threat intel testing)
]


def _rand_internal() -> str:
    subnet = random.choice(_INTERNAL_SUBNETS)
    return subnet + str(random.randint(2, 254))


def _rand_external() -> str:
    return random.choice(_EXTERNAL_IPS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_flow_event(
    src_ip: str, dst_ip: str, src_port: int, dst_port: int,
    protocol: str, pkt_count: int, byte_count: int,
    duration: float, tcp_flags: list, unique_ports: int,
) -> dict:
    return {
        "event_id":       str(uuid.uuid4()),
        "source":         "network_tap",
        "timestamp":      _now_iso(),
        "src_ip":         src_ip,
        "dst_ip":         dst_ip,
        "src_port":       src_port,
        "dst_port":       dst_port,
        "protocol":       protocol,
        "packet_count":   pkt_count,
        "byte_count":     byte_count,
        "duration_sec":   round(duration, 3),
        "tcp_flags":      tcp_flags,
        "unique_dst_ports": unique_ports,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulated sensor (dev mode)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedNetworkSensor:
    """
    Generates realistic benign + attack traffic flows.
    Identical output format to the live Scapy sensor.
    """

    def __init__(self, flows_per_second: float = 5.0,
                 attack_probability: float = 0.08):
        self._fps   = flows_per_second
        self._atk_p = attack_probability
        self._producer = get_producer()
        self._running  = False

    def _emit_normal(self) -> None:
        sp, dp, proto, pkts_r, bytes_r = random.choice(_NORMAL_PROFILES)
        sp = random.randint(49152, 65535)   # randomize src port
        src = _rand_internal()
        dst = random.choice([_rand_internal(), _rand_external()])
        pkts  = random.randint(*pkts_r)
        bts   = random.randint(*bytes_r)
        flags = ["SYN", "ACK"] if proto == "TCP" else []
        evt = _build_flow_event(src, dst, sp, dp, proto,
                                pkts, bts, random.uniform(0.1, 30.0),
                                flags, 1)
        self._producer.send(KAFKA_TOPIC_RAW, evt)

    def _emit_attack(self) -> None:
        attack = random.choice(_ATTACK_PROFILES)
        src = _rand_internal()
        dst = random.choice([_rand_internal(), _rand_external()])
        a_type = attack["type"]

        if a_type == "port_scan":
            for dp in attack["dst_ports"]:
                evt = _build_flow_event(
                    src, dst, random.randint(49152, 65535), dp,
                    attack["proto"], attack["pkts"], attack["bytes"],
                    0.05, attack["flags"], len(attack["dst_ports"]),
                )
                self._producer.send(KAFKA_TOPIC_RAW, evt)
            log.info(f"[SIM] Port scan: {src} → {dst} ({len(attack['dst_ports'])} ports)")

        elif a_type in ("ssh_brute", "udp_flood", "dns_amp", "c2_beacon"):
            dp = attack.get("dst_port", 80)
            evt = _build_flow_event(
                src, dst, random.randint(49152, 65535), dp,
                attack["proto"], attack["pkts"], attack["bytes"],
                random.uniform(1.0, 60.0), attack["flags"], 1,
            )
            self._producer.send(KAFKA_TOPIC_RAW, evt)
            log.info(f"[SIM] Attack({a_type}): {src} → {dst}:{dp}")

    def start(self, stop_event: threading.Event = None) -> None:
        self._running = True
        interval = 1.0 / self._fps
        log.info(f"[NET-SIM] Simulated sensor: {self._fps} flows/sec, "
                 f"attack_prob={self._atk_p}")

        while self._running:
            if stop_event and stop_event.is_set():
                break
            try:
                if random.random() < self._atk_p:
                    self._emit_attack()
                else:
                    self._emit_normal()
            except Exception as e:
                log.error(f"[NET-SIM] Error: {e}")
            time.sleep(interval)

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Live Scapy sensor (production mode)
# ─────────────────────────────────────────────────────────────────────────────

class LiveNetworkSensor:
    """
    Captures real packets via Scapy. Requires root + scapy installed.
    Aggregates packets into flows, flushes every FLOW_WINDOW_SECONDS.
    """

    def __init__(self, interface: str = NETWORK_INTERFACE):
        self._iface    = interface
        self._producer = get_producer()
        self._flows: dict = defaultdict(lambda: {
            "packet_count": 0, "byte_count": 0,
            "start_time": None, "flags": set(),
            "unique_dst_ports": set(),
        })
        self._lock = threading.Lock()

    def _flush_old_flows(self) -> None:
        now = time.time()
        to_flush = []
        with self._lock:
            for key, flow in self._flows.items():
                if flow["start_time"] and now - flow["start_time"] >= FLOW_WINDOW_SECONDS:
                    to_flush.append((key, dict(flow)))
            for key in [k for k, _ in to_flush]:
                del self._flows[key]

        for key, flow in to_flush:
            src_ip, dst_ip, src_port, dst_port, proto = key
            evt = _build_flow_event(
                src_ip, dst_ip, src_port, dst_port, proto,
                flow["packet_count"], flow["byte_count"],
                now - flow["start_time"],
                list(flow["flags"]),
                len(flow["unique_dst_ports"]),
            )
            self._producer.send(KAFKA_TOPIC_RAW, evt)
            log.debug(f"[NET-LIVE] Flushed flow: {src_ip}→{dst_ip} [{proto}]")

    def _handle_packet(self, pkt) -> None:
        try:
            from scapy.all import IP, TCP, UDP, ICMP
        except ImportError:
            return

        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        proto, sp, dp, flags = "OTHER", 0, 0, set()

        if pkt.haslayer(TCP):
            proto = "TCP"
            sp, dp = pkt[TCP].sport, pkt[TCP].dport
            f = pkt[TCP].flags
            if f & 0x02: flags.add("SYN")
            if f & 0x10: flags.add("ACK")
            if f & 0x01: flags.add("FIN")
            if f & 0x04: flags.add("RST")
        elif pkt.haslayer(UDP):
            proto, sp, dp = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif pkt.haslayer(ICMP):
            proto = "ICMP"

        key = (ip.src, ip.dst, sp, dp, proto)
        with self._lock:
            flow = self._flows[key]
            if flow["start_time"] is None:
                flow["start_time"] = time.time()
            flow["packet_count"] += 1
            flow["byte_count"]   += len(pkt)
            flow["flags"].update(flags)
            flow["unique_dst_ports"].add(dp)

        self._flush_old_flows()

    def start(self, stop_event: threading.Event = None) -> None:
        try:
            from scapy.all import sniff
        except ImportError:
            log.error("[NET-LIVE] scapy not installed — use SimulatedNetworkSensor")
            return

        log.info(f"[NET-LIVE] Sniffing on {self._iface}")
        sniff(
            iface=self._iface,
            prn=self._handle_packet,
            store=False,
            stop_filter=lambda _: stop_event.is_set() if stop_event else False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_network_sensor(simulated: bool = True, **kwargs):
    if simulated:
        return SimulatedNetworkSensor(**kwargs)
    return LiveNetworkSensor()
