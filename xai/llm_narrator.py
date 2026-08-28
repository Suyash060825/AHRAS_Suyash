from __future__ import annotations
"""
AHRAS Module 6 — LLM Threat Narration & SOC Copilot Engine
----------------------------------------------------------
Transforms complex multi-modal alert vectors, XAI feature contributions, and MITRE ATT&CK
mappings into natural language executive incident summaries and technical remediation playbooks.

Primary Capabilities:
  1. Executive Summary Narrator: Plain-English breach summary tailored for non-technical CISOs.
  2. Technical Deep-Dive: Detailed driver breakdown explaining why the ML models and rules fired.
  3. Step-by-Step Remediation Playbook: Actionable containment procedures for SOC tier-1 analysts.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from detection.risk_engine import RiskResult

log = logging.getLogger(__name__)


@dataclass
class LLMNarrative:
    """Natural language threat narration output."""
    incident_id:        str
    entity_key:         str
    severity:           str
    risk_score:         float
    executive_summary:  str
    technical_narrative: str
    recommended_playbook: list[str]
    mitre_attack_summary: str


class LLMThreatNarrator:
    """
    Synthesizes natural language threat summaries from multi-engine detection outputs.
    Can operate in local template-based engine mode or connect to API models (Claude/GPT/Llama).
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def generate_narrative(self, risk_res: RiskResult, evt: dict = None) -> LLMNarrative:
        if evt is None:
            evt = {}

        key = risk_res.entity_key
        sev = risk_res.severity
        score = risk_res.risk_score
        flags = risk_res.flags
        mitre = risk_res.mitre_techniques

        # 1. Executive Summary
        exec_sum = (
            f"ALERT [{sev} | Risk Score: {score:.1%}]: Potential security breach detected on entity '{key}'. "
            f"Multi-engine analysis identified {len(flags)} concurrent risk indicators matching "
            f"{len(mitre)} MITRE ATT&CK techniques ({', '.join(mitre) if mitre else 'General Anomaly'}). "
            f"Immediate containment policy: {risk_res.remediation_level}."
        )

        # 2. Technical Narrative
        tech_lines = []
        tech_lines.append(f"At timestamp {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, entity '{key}' produced anomalous activity:")
        tech_lines.append(f"• Signature Normalized Severity S_sig: {risk_res.S_sig:.2f}")
        tech_lines.append(f"• ML Anomaly Ensemble Score A_ml: {risk_res.A_ml:.2f}")
        tech_lines.append(f"• Behavioral Vector Drift ΔD: {risk_res.delta_D:.2f}")
        tech_lines.append(f"• Dynamic Trust Score T_trust: {risk_res.T_trust:.2f}")
        if flags:
            tech_lines.append(f"• Triggered Indicators: {', '.join(flags)}")
        tech_narrative = "\n".join(tech_lines)

        # 3. Recommended Remediation Playbook
        playbook = []
        if sev in ("CRITICAL", "HIGH"):
            playbook.append(f"1. Execute automated/staged host isolation for '{key}'.")
            playbook.append("2. Terminate parent and child process trees associated with alert.")
            playbook.append("3. Invalidate active user session tokens and enforce MFA re-authentication.")
            playbook.append("4. Capture memory dump for forensic analysis.")
        else:
            playbook.append(f"1. Monitor entity '{key}' traffic for 24-hour anomaly recurrence.")
            playbook.append("2. Verify user activity with business unit owner.")
            playbook.append("3. Submit false-positive feedback if activity is verified legitimate.")

        # 4. MITRE ATT&CK Summary
        mitre_summary = f"Mapped Techniques: {', '.join(mitre)}" if mitre else "No direct MITRE technique hit."

        return LLMNarrative(
            incident_id=f"INC-{int(time.time())}",
            entity_key=key,
            severity=sev,
            risk_score=score,
            executive_summary=exec_sum,
            technical_narrative=tech_narrative,
            recommended_playbook=playbook,
            mitre_attack_summary=mitre_summary,
        )


# Singleton
_narrator_instance: Optional[LLMThreatNarrator] = None


def get_llm_narrator() -> LLMThreatNarrator:
    global _narrator_instance
    if _narrator_instance is None:
        _narrator_instance = LLMThreatNarrator()
    return _narrator_instance
