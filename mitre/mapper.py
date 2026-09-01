from typing import Dict, Any, List

# Hardcoded MITRE ATT&CK Mapping
MITRE_MAPPING = {
    "SSH Bruteforce": {
        "technique_id": "T1110.001",
        "tactic": "Credential Access",
        "name": "Brute Force: Password Guessing",
        "url": "https://attack.mitre.org/techniques/T1110/001/",
        "severity_boost": 0.15
    },
    "FTP Bruteforce": {
        "technique_id": "T1110.001",
        "tactic": "Credential Access",
        "name": "Brute Force: Password Guessing",
        "url": "https://attack.mitre.org/techniques/T1110/001/",
        "severity_boost": 0.15
    },
    "Port Scan": {
        "technique_id": "T1046",
        "tactic": "Discovery",
        "name": "Network Service Discovery",
        "url": "https://attack.mitre.org/techniques/T1046/",
        "severity_boost": 0.05
    },
    "Traffic Flood": {
        "technique_id": "T1498.001",
        "tactic": "Impact",
        "name": "Network Denial of Service: Direct Network Flood",
        "url": "https://attack.mitre.org/techniques/T1498/001/",
        "severity_boost": 0.20
    },
    "Lateral Movement": {
        "technique_id": "T1021",
        "tactic": "Lateral Movement",
        "name": "Remote Services",
        "url": "https://attack.mitre.org/techniques/T1021/",
        "severity_boost": 0.25
    },
    "Web Attack": {
        "technique_id": "T1190",
        "tactic": "Initial Access",
        "name": "Exploit Public-Facing Application",
        "url": "https://attack.mitre.org/techniques/T1190/",
        "severity_boost": 0.30
    },
    "Botnet": {
        "technique_id": "T1071.001",
        "tactic": "Command and Control",
        "name": "Application Layer Protocol: Web Protocols",
        "url": "https://attack.mitre.org/techniques/T1071/001/",
        "severity_boost": 0.35
    }
}

def enrich_with_mitre(attack_type: str) -> Dict[str, Any]:
    """Returns MITRE ATT&CK metadata for a given attack type, or a generic fallback."""
    # Try exact match or substring match
    for key, mapping in MITRE_MAPPING.items():
        if key.lower() in attack_type.lower() or attack_type.lower() in key.lower():
            return mapping
            
    return {
        "technique_id": "T1000",
        "tactic": "Unknown",
        "name": "Uncategorized Anomaly",
        "url": "https://attack.mitre.org/",
        "severity_boost": 0.0
    }

def get_mitre_techniques(attack_types: List[str]) -> List[str]:
    """Returns a list of unique MITRE technique IDs for a list of attack types."""
    techs = set()
    for attack in attack_types:
        mapping = enrich_with_mitre(attack)
        techs.add(mapping["technique_id"])
    return list(techs)
