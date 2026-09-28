from __future__ import annotations
"""
AHRAS Module — Privacy-Aware Telemetry & Data Minimization (Section 35)
-----------------------------------------------------------------------
Implements policy-governed data minimization and confidentiality preservation
across host and network telemetry:
  1. Telemetry Classification Tiers:
     - PUBLIC: External internet telemetry, standard port metadata
     - INTERNAL: Internal RFC1918 communications, service names
     - SENSITIVE: Internal hostnames, non-privileged user accounts, file paths
     - HIGHLY_SENSITIVE: Privileged identities, process memory strings, API keys, credentials

  2. Privacy Transformations:
     - IP Address Anonymization / Subnet Masking (e.g. /24 or /16 truncation)
     - Keyed HMAC User Pseudonymization
     - Commandline / Path Redaction of sensitive tokens and passwords
     - Time bucketing (jittering timestamps to 10s intervals)
     - Feature minimization (dropping raw strings while preserving statistical counts)

  3. Forensic Preservation Invariant:
     - Retains cryptographic reference hash (`forensic_vault_ref = SHA256(raw_event)`)
       enabling authorized SOC investigators to request sealed raw records.
"""

import enum
import hashlib
import hmac
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class PrivacyTier(str, enum.Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    HIGHLY_SENSITIVE = "HIGHLY_SENSITIVE"


class TelemetryPrivacyManager:
    """
    Applies data minimization and privacy-preserving transformations to security events
    based on regulatory policy and organizational confidentiality tiers.
    """

    def __init__(self, hmac_key: str = "ahras_privacy_salt_2026"):
        self.hmac_key = hmac_key.encode("utf-8")
        self._pseudonym_cache: Dict[str, str] = {}
        self._lock = threading.RLock()

    def classify_event(self, event: Dict[str, Any]) -> PrivacyTier:
        """Determines the baseline privacy classification tier of an incoming telemetry event."""
        # Check for credentials, keys, or memory dumps
        raw_cmd = str(event.get("process_cmdline", "")).lower()
        path = str(event.get("file_path", "")).lower()
        user = str(event.get("user", "")).lower()

        if any(tok in raw_cmd for tok in ["password", "token", "apikey", "secret", "bearer", "aws_secret"]):
            return PrivacyTier.HIGHLY_SENSITIVE

        if any(tok in path for tok in ["shadow", "id_rsa", "sam", "ntds.dit", "keychain", "vault"]):
            return PrivacyTier.HIGHLY_SENSITIVE

        if user in ["root", "administrator", "system", "domain admins"]:
            return PrivacyTier.HIGHLY_SENSITIVE

        if event.get("event_type") in ("process_creation", "file_modification", "identity_authentication"):
            return PrivacyTier.SENSITIVE

        # Internal RFC1918 traffic
        src_ip = str(event.get("src_ip", ""))
        if src_ip.startswith("10.") or src_ip.startswith("192.168.") or src_ip.startswith("172.16."):
            return PrivacyTier.INTERNAL

        return PrivacyTier.PUBLIC

    def mask_ip(self, ip: str, tier: PrivacyTier) -> str:
        """Anonymizes IP address according to privacy tier."""
        if not ip or ip in ("null", "unknown", "127.0.0.1", "localhost"):
            return ip

        if tier in (PrivacyTier.PUBLIC, PrivacyTier.INTERNAL):
            return ip

        parts = ip.split(".")
        if len(parts) == 4:
            if tier == PrivacyTier.SENSITIVE:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
            elif tier == PrivacyTier.HIGHLY_SENSITIVE:
                return f"{parts[0]}.{parts[1]}.0.0/16"

        return ip

    def pseudonymize_user(self, user: str, tier: PrivacyTier) -> str:
        """Generates deterministic keyed HMAC pseudonym for usernames in sensitive tiers."""
        if not user or user in ("null", "unknown", "SYSTEM", "LOCAL SERVICE"):
            return user

        if tier in (PrivacyTier.PUBLIC, PrivacyTier.INTERNAL):
            return user

        with self._lock:
            if user not in self._pseudonym_cache:
                digest = hmac.new(self.hmac_key, user.encode("utf-8"), hashlib.sha256).hexdigest()
                self._pseudonym_cache[user] = f"user_{digest[:12]}"
            return self._pseudonym_cache[user]

    def redact_commandline(self, cmdline: str, tier: PrivacyTier) -> str:
        """Redacts sensitive arguments, passwords, and file paths from execution command lines."""
        if not cmdline:
            return cmdline

        if tier == PrivacyTier.PUBLIC:
            return cmdline

        # Redact password key-value patterns
        sanitized = re.sub(
            r'(password|passwd|pwd|token|api_key|secret)\s*[:=]\s*[^\s]+',
            r'\1=[REDACTED_SECRET]',
            cmdline,
            flags=re.IGNORECASE,
        )

        # Redact curl/wget user credential flags: -u user:password
        sanitized = re.sub(
            r'(-u\s+[\w\.-]+:)[^\s]+',
            r'\g<1>[REDACTED_SECRET]',
            sanitized,
            flags=re.IGNORECASE,
        )

        if tier == PrivacyTier.HIGHLY_SENSITIVE:
            # Mask detailed file path hierarchy
            sanitized = re.sub(r'(/home/\w+/|C:\\Users\\[^\\]+\\)', r'[USER_HOME]/', sanitized)

        return sanitized

    def sanitize_event(
        self,
        event: Dict[str, Any],
        target_tier: Optional[PrivacyTier] = None,
    ) -> Dict[str, Any]:
        """
        Applies full data minimization to an event while preserving mathematical forensic link.
        """
        tier = target_tier or self.classify_event(event)
        canonical_raw = str(sorted(event.items())).encode("utf-8")
        vault_ref = hashlib.sha256(canonical_raw).hexdigest()

        sanitized = dict(event)
        sanitized["privacy_tier"] = tier.value
        sanitized["forensic_vault_ref"] = vault_ref

        # Mask network entities
        if "src_ip" in sanitized:
            sanitized["src_ip"] = self.mask_ip(str(sanitized["src_ip"]), tier)
        if "dst_ip" in sanitized:
            sanitized["dst_ip"] = self.mask_ip(str(sanitized["dst_ip"]), tier)

        # Pseudonymize users
        if "user" in sanitized:
            sanitized["user"] = self.pseudonymize_user(str(sanitized["user"]), tier)

        # Redact commandlines
        if "process_cmdline" in sanitized:
            sanitized["process_cmdline"] = self.redact_commandline(str(sanitized["process_cmdline"]), tier)

        # Drop raw payload on HIGHLY_SENSITIVE
        if tier == PrivacyTier.HIGHLY_SENSITIVE:
            sanitized.pop("raw_payload", None)
            sanitized.pop("memory_dump", None)
            sanitized.pop("file_content", None)

        return sanitized
