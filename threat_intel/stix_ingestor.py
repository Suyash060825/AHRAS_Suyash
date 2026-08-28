from __future__ import annotations
"""
AHRAS STIX 2.1 & TAXII 2.1 Threat Feed Ingestor
------------------------------------------------
Parses STIX 2.1 bundles, indicators, observables, and threat-actor patterns
into standardized internal IOC representations.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

# Regex extractors for STIX pattern strings
_IP_PATTERN_RE = re.compile(r"ipv4-addr:value\s*=\s*'([^']+)'", re.IGNORECASE)
_DOMAIN_PATTERN_RE = re.compile(r"domain-name:value\s*=\s*'([^']+)'", re.IGNORECASE)
_HASH_PATTERN_RE = re.compile(r"file:hashes\.(?:SHA-256|MD5|SHA-1)\s*=\s*'([^']+)'", re.IGNORECASE)
_URL_PATTERN_RE = re.compile(r"url:value\s*=\s*'([^']+)'", re.IGNORECASE)


class STIXIngestor:
    """
    Ingests STIX 2.1 JSON bundles and extracts observable IOCs.
    """

    @staticmethod
    def parse_bundle(bundle_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses a STIX 2.1 bundle dict and returns a list of normalized IOC dicts.
        """
        extracted_iocs = []
        objects = bundle_data.get("objects", [])
        
        for obj in objects:
            obj_type = obj.get("type", "")
            
            if obj_type == "indicator":
                pattern = obj.get("pattern", "")
                name = obj.get("name", "Unknown Threat")
                confidence = float(obj.get("confidence", 80)) / 100.0
                labels = obj.get("labels", [])
                
                # Extract IP indicators
                for match in _IP_PATTERN_RE.finditer(pattern):
                    extracted_iocs.append({
                        "ioc_value": match.group(1),
                        "ioc_type": "ip",
                        "threat_name": name,
                        "confidence": confidence,
                        "severity": "HIGH",
                        "source": "STIX_2.1_FEED",
                        "tags": labels,
                    })
                    
                # Extract Domain indicators
                for match in _DOMAIN_PATTERN_RE.finditer(pattern):
                    extracted_iocs.append({
                        "ioc_value": match.group(1),
                        "ioc_type": "domain",
                        "threat_name": name,
                        "confidence": confidence,
                        "severity": "HIGH",
                        "source": "STIX_2.1_FEED",
                        "tags": labels,
                    })
                    
                # Extract Hash indicators
                for match in _HASH_PATTERN_RE.finditer(pattern):
                    extracted_iocs.append({
                        "ioc_value": match.group(1).lower(),
                        "ioc_type": "hash",
                        "threat_name": name,
                        "confidence": confidence,
                        "severity": "CRITICAL",
                        "source": "STIX_2.1_FEED",
                        "tags": labels,
                    })
                    
                # Extract URL indicators
                for match in _URL_PATTERN_RE.finditer(pattern):
                    extracted_iocs.append({
                        "ioc_value": match.group(1),
                        "ioc_type": "url",
                        "threat_name": name,
                        "confidence": confidence,
                        "severity": "HIGH",
                        "source": "STIX_2.1_FEED",
                        "tags": labels,
                    })
                    
            elif obj_type == "observed-data":
                # Handle observed data objects
                objs = obj.get("objects", {})
                for k, obs in objs.items():
                    if obs.get("type") == "ipv4-addr":
                        extracted_iocs.append({
                            "ioc_value": obs.get("value"),
                            "ioc_type": "ip",
                            "threat_name": "Observed Malicious IP",
                            "confidence": 0.85,
                            "severity": "HIGH",
                            "source": "STIX_2.1_OBSERVED",
                        })
                        
        log.info(f"[STIX] Ingested {len(extracted_iocs)} IOCs from STIX bundle")
        return extracted_iocs

    @classmethod
    def from_json_file(cls, filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.parse_bundle(data)
