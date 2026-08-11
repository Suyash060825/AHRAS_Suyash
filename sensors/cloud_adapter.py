"""
AHRAS Cloud Log Adapter
------------------------
Publishes cloud audit events to the message bus.

Two modes:
  SYNTHETIC → generates realistic AWS CloudTrail-format events
              (dev/test, no AWS credentials required)
  LIVE      → polls real AWS CloudTrail via boto3
              (requires: pip install boto3 + AWS credentials)

Synthetic events include high-privilege and attack-mimicking
actions for evaluation of the detection pipeline.
"""

import uuid
import time
import random
import logging
import threading
from datetime import datetime, timezone

from config.settings import KAFKA_TOPIC_RAW, CLOUD_INTERVAL_SEC
from pipeline.bus import get_producer

log = logging.getLogger(__name__)

# ── Event catalog ─────────────────────────────────────────────────────────────
# (action, severity_hint, weight)
# Higher weight = more frequent in synthetic stream

_EVENTS = [
    # Normal day-to-day
    ("s3:GetObject",                    "low",      40),
    ("s3:PutObject",                    "low",      30),
    ("s3:ListBucket",                   "low",      20),
    ("ec2:DescribeInstances",           "low",      15),
    ("cloudwatch:PutMetricData",        "low",      10),
    ("lambda:InvokeFunction",           "low",      15),
    ("rds:DescribeDBInstances",         "low",      10),
    # Slightly suspicious
    ("iam:ListUsers",                   "medium",    5),
    ("iam:GetUser",                     "medium",    5),
    ("ec2:DescribeSecurityGroups",      "medium",    5),
    ("sts:GetCallerIdentity",           "medium",    5),
    # High privilege
    ("iam:CreateUser",                  "high",      2),
    ("iam:CreateAccessKey",             "high",      2),
    ("iam:AttachUserPolicy",            "high",      2),
    ("iam:PutUserPolicy",               "high",      2),
    ("ec2:AuthorizeSecurityGroupIngress","high",     2),
    ("s3:PutBucketAcl",                 "high",      2),
    # Critical (attacker behavior)
    ("cloudtrail:StopLogging",          "critical",  1),
    ("cloudtrail:DeleteTrail",          "critical",  1),
    ("iam:DeleteAccountPasswordPolicy", "critical",  1),
    ("ec2:CreateVpc",                   "high",      1),
    ("s3:DeleteBucket",                 "high",      1),
    ("iam:DeactivateMFADevice",         "critical",  1),
]

_ACTIONS    = [e[0] for e in _EVENTS]
_SEVERITIES = [e[1] for e in _EVENTS]
_WEIGHTS    = [e[2] for e in _EVENTS]

_USERS = [
    "alice@corp.com", "bob@corp.com",
    "svc-deploy", "svc-backup", "svc-monitoring",
    "admin@corp.com",
    "unknown-external",          # attacker
    "terraform-ci",
]

_USER_WEIGHTS = [15, 15, 10, 10, 10, 5, 3, 8]

_REGIONS     = ["us-east-1", "ap-south-1", "eu-west-1", "us-west-2"]
_USER_AGENTS = [
    "aws-cli/2.13.0",
    "boto3/1.28.0",
    "Terraform/1.5.0",
    "console.amazonaws.com",
    "python-requests/2.31.0",   # suspicious — not a normal AWS agent
]
_EXTERNAL_IPS = [
    "203.0.113.10", "45.33.32.156", "198.51.100.5",
    "104.21.64.1",  "185.220.101.5",
]
_INTERNAL_IPS = [
    "10.0.0.51", "10.0.0.82", "192.168.1.15", "172.16.0.10"
]

_ERROR_CODES = [None, None, None, None, "AccessDenied", "NoSuchKey",
                "InvalidClientTokenId", "UnauthorizedOperation"]


def _rand_event() -> dict:
    idx    = random.choices(range(len(_EVENTS)), weights=_WEIGHTS, k=1)[0]
    action = _ACTIONS[idx]
    sev    = _SEVERITIES[idx]
    user   = random.choices(_USERS, weights=_USER_WEIGHTS, k=1)[0]

    # Attackers come from external IPs
    if user == "unknown-external":
        src_ip = random.choice(_EXTERNAL_IPS)
        agent  = random.choice(["python-requests/2.31.0", "curl/7.88.0"])
    else:
        src_ip = random.choice(_INTERNAL_IPS)
        agent  = random.choice(_USER_AGENTS[:-1])

    # Occasionally inject an error
    error = random.choice(_ERROR_CODES) if sev != "low" else None

    return {
        "event_id":           str(uuid.uuid4()),
        "source":             "cloud_adapter",
        "event_type":         "cloud_api_call",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "provider":           "aws",
        "action":             action,
        "severity_hint":      sev,
        "user_identity":      user,
        "source_ip":          src_ip,
        "region":             random.choice(_REGIONS),
        "user_agent":         agent,
        "request_parameters": {},
        "error_code":         error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic cloud adapter
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCloudAdapter:
    def __init__(self, interval: float = CLOUD_INTERVAL_SEC):
        self._interval = interval
        self._producer = get_producer()

    def start(self, stop_event: threading.Event = None) -> None:
        log.info(f"[CLOUD-SIM] Synthetic adapter started (interval={self._interval}s)")
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                evt = _rand_event()
                self._producer.send(KAFKA_TOPIC_RAW, evt)
                log.debug(
                    f"[CLOUD-SIM] {evt['action']:45s} "
                    f"user={evt['user_identity']:20s} "
                    f"sev={evt['severity_hint']}"
                )
            except Exception as e:
                log.error(f"[CLOUD-SIM] Error: {e}")
            time.sleep(self._interval)


# ─────────────────────────────────────────────────────────────────────────────
# Live AWS CloudTrail adapter (requires boto3)
# ─────────────────────────────────────────────────────────────────────────────

class LiveCloudTrailAdapter:
    """
    Polls AWS CloudTrail for real events.
    Requires: pip install boto3 and valid AWS credentials.
    """

    def __init__(self, region: str = "us-east-1", poll_interval: float = 30.0):
        try:
            import boto3
            self._client   = boto3.client("cloudtrail", region_name=region)
            self._interval = poll_interval
            self._producer = get_producer()
            self._last_ts  = None
            log.info(f"[CLOUD-LIVE] CloudTrail adapter: region={region}")
        except ImportError:
            raise RuntimeError("boto3 not installed: pip install boto3")

    def start(self, stop_event: threading.Event = None) -> None:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                self._poll()
            except Exception as e:
                log.error(f"[CLOUD-LIVE] Poll error: {e}")
            time.sleep(self._interval)

    def _poll(self) -> None:
        kwargs = {"MaxResults": 50}
        if self._last_ts:
            kwargs["StartTime"] = self._last_ts

        resp   = self._client.lookup_events(**kwargs)
        events = resp.get("Events", [])

        for e in events:
            self._last_ts = e.get("EventTime")
            record = {
                "event_id":           str(uuid.uuid4()),
                "source":             "cloud_adapter",
                "event_type":         "cloud_api_call",
                "timestamp":          e.get("EventTime").isoformat()
                                      if e.get("EventTime") else datetime.now(timezone.utc).isoformat(),
                "provider":           "aws",
                "action":             e.get("EventName", ""),
                "severity_hint":      "medium",
                "user_identity":      e.get("Username", ""),
                "source_ip":          "",
                "region":             "",
                "user_agent":         "",
                "request_parameters": {},
                "error_code":         None,
            }
            self._producer.send(KAFKA_TOPIC_RAW, record)
            log.debug(f"[CLOUD-LIVE] {record['action']}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_cloud_adapter(synthetic: bool = True, **kwargs):
    if synthetic:
        return SyntheticCloudAdapter(**kwargs)
    return LiveCloudTrailAdapter(**kwargs)


def start_cloud_adapter_thread(synthetic: bool = True) -> threading.Thread:
    stop_event = threading.Event()
    adapter    = get_cloud_adapter(synthetic=synthetic)
    t = threading.Thread(
        target=adapter.start,
        args=(stop_event,),
        name="ahras-cloud-adapter",
        daemon=True,
    )
    t.start()
    return t
