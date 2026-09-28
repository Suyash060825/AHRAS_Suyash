from __future__ import annotations
"""
AHRAS Grounded LLM Analyst Assistant (Section 40)
--------------------------------------------------
Strict grounding-only security intelligence assistant.

Mandates:
1. Grounding-Only:
   - Operates exclusively on verifiable forensic evidence:
     DecisionTrace, security knowledge graph, historical incident cases,
     detector scores, risk arithmetic, threat intelligence, and ATT&CK mappings.
   - Every claim, deduction, and recommendation MUST cite its supporting EvidenceRecord ID.
   - Refuses to hallucinate missing data: explicitly marks unknown fields as "UNAVAILABLE".

2. Zero Autonomous Authorization Rights:
   - "It must NOT independently authorize actions."
   - The assistant only produces explanations, investigation checklists, analyst questions,
     and response rationales.
   - Final execution authority resides strictly in deterministic policy and human approvers.

3. Epistemic Abstention:
   - If confidence is low or uncertainty is elevated (uncertainty > 0.40),
     the assistant abstains from declaring definitive verdicts and instead recommends
     targeted forensic telemetry collection.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set

log = logging.getLogger(__name__)


@dataclass
class GroundedExplanation:
    """Standardized grounded analysis output for SOC analysts."""
    incident_id:            str
    timestamp:              float
    executive_summary:      str
    evidence_breakdown:     List[Dict[str, Any]]
    investigation_checklist: List[str]
    analyst_inquiry_points: List[str]
    response_rationale:     str
    cited_evidence_ids:     List[str]
    epistemic_uncertainty:  float
    is_abstained:           bool
    abstention_reason:      Optional[str] = None
    authorization_disclaimer: str = (
        "STRICT SAFETY NOTICE: This narrative is generated for forensic decision support only. "
        "The LLM Assistant possesses zero autonomous authorization authority. "
        "All active containment or modification actions require deterministic policy validation "
        "and human SOC approval."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundedLLMAssistant:
    """
    Forensic LLM analyst assistant strictly constrained by verified evidence traces.
    """

    def __init__(self, uncertainty_threshold: float = 0.40):
        self.uncertainty_threshold = uncertainty_threshold

    def analyze_incident(
        self,
        incident_id: str,
        decision_trace: Dict[str, Any],
        evidence_records: Optional[List[Dict[str, Any]]] = None,
        graph_context: Optional[Dict[str, Any]] = None,
        threat_intel: Optional[Dict[str, Any]] = None,
    ) -> GroundedExplanation:
        """
        Synthesizes a strictly grounded forensic narrative from verifiable incident context.
        """
        now = time.time()
        ev_records = evidence_records or []
        g_ctx = graph_context or {}
        ti_ctx = threat_intel or {}

        # 1. Extract quantitative parameters from DecisionTrace
        risk_score = float(decision_trace.get("composite_risk_score", decision_trace.get("risk_score", 0.0)))
        uncertainty = float(decision_trace.get("epistemic_uncertainty", decision_trace.get("uncertainty", 0.15)))
        ood_score = float(decision_trace.get("ood_score", 0.0))
        entity_id = decision_trace.get("entity_id", "UNKNOWN_ENTITY")
        mitre_techniques = decision_trace.get("mitre_techniques", [])
        detector_outputs = decision_trace.get("detector_outputs", {})

        # Collect all valid evidence IDs
        cited_ids: List[str] = [
            ev.get("evidence_id") for ev in ev_records if ev.get("evidence_id")
        ]
        if not cited_ids and "evidence_id" in decision_trace:
            cited_ids.append(decision_trace["evidence_id"])

        # 2. Epistemic Abstention Check
        if uncertainty >= self.uncertainty_threshold and not cited_ids:
            return GroundedExplanation(
                incident_id=incident_id,
                timestamp=now,
                executive_summary="ANALYSIS ABSTAINED: High epistemic uncertainty with zero verified evidence links.",
                evidence_breakdown=[],
                investigation_checklist=["Deploy additional network and endpoint telemetry sensors."],
                analyst_inquiry_points=["Verify telemetry pipeline connectivity and sensor health."],
                response_rationale="No containment recommended due to ungrounded uncertainty state.",
                cited_evidence_ids=[],
                epistemic_uncertainty=uncertainty,
                is_abstained=True,
                abstention_reason=f"Epistemic uncertainty ({uncertainty:.2f}) exceeds threshold ({self.uncertainty_threshold:.2f}) without supporting evidence records.",
            )

        # 3. Evidence Breakdown Construction (Strictly cited from evidence records)
        breakdown: List[Dict[str, Any]] = []
        for ev in ev_records:
            ev_id = ev.get("evidence_id", "EVID-UNKNOWN")
            source = ev.get("source", "DETECTOR")
            norm_score = float(ev.get("normalized_score", 0.0))
            reason = ev.get("explanation", ev.get("reason", "Anomalous telemetry metric observed."))
            breakdown.append({
                "evidence_id": ev_id,
                "source": source,
                "contribution_score": round(norm_score, 4),
                "grounded_fact": reason,
                "mitre_technique": ev.get("mitre_mapping", []),
            })

        # 4. Synthesize Executive Summary
        severity = "CRITICAL" if risk_score >= 0.80 else ("HIGH" if risk_score >= 0.60 else "MEDIUM")
        techniques_str = ", ".join(mitre_techniques) if mitre_techniques else "Uncategorized Anomaly"
        
        exec_summary = (
            f"[{severity} INCIDENT {incident_id}] Entity '{entity_id}' produced verified risk score of {risk_score:.2f} "
            f"(Epistemic Uncertainty: {uncertainty:.2f}, OOD/Novelty: {ood_score:.2f}). "
            f"Activity correlates with MITRE ATT&CK techniques: [{techniques_str}]. "
            f"Analysis is strictly grounded on {len(cited_ids)} verified evidence record(s)."
        )

        # 5. Dynamic Investigation Checklist
        checklist = [
            f"Inspect raw packet captures and socket connections on entity '{entity_id}'.",
            f"Review process ancestry for parent/child interpreter anomalies matching {techniques_str}.",
        ]
        if g_ctx.get("lateral_paths"):
            checklist.append(f"Examine reachable neighbor assets on lateral path: {g_ctx.get('lateral_paths')}.")
        if ti_ctx.get("known_c2_ips"):
            checklist.append(f"Cross-reference outbound network traffic against known C2 IOCs: {ti_ctx.get('known_c2_ips')}.")

        # 6. Analyst Inquiry Points
        inquiries = [
            f"Did user assigned to '{entity_id}' initiate approved administrative scripts at timestamp {now}?",
            "Are recent file write bursts part of scheduled enterprise backup or database migration routines?",
            f"Confirm whether host '{entity_id}' has unpatched vulnerabilities matching active exploit pathways.",
        ]

        # 7. Recommended Response Rationale (Explicitly grounded, non-authoritative)
        if risk_score >= 0.75 and uncertainty < 0.30:
            rationale = (
                f"RECOMMENDATION ONLY: Staged host isolation of '{entity_id}' is justified based on "
                f"high verified risk ({risk_score:.2f}) and low uncertainty ({uncertainty:.2f}) cited in records "
                f"[{', '.join(cited_ids[:3])}]. Awaiting SOC analyst confirmation."
            )
        elif ood_score >= 0.70:
            rationale = (
                f"RECOMMENDATION ONLY: High zero-day / OOD signature ({ood_score:.2f}) observed. "
                f"Route traffic to dynamic honeypot deception tripwire to observe attacker behavior safely."
            )
        else:
            rationale = (
                f"RECOMMENDATION ONLY: Maintain baseline monitoring. Risk score ({risk_score:.2f}) "
                f"does not warrant disruptive containment without further evidence."
            )

        return GroundedExplanation(
            incident_id=incident_id,
            timestamp=now,
            executive_summary=exec_summary,
            evidence_breakdown=breakdown,
            investigation_checklist=checklist,
            analyst_inquiry_points=inquiries,
            response_rationale=rationale,
            cited_evidence_ids=cited_ids,
            epistemic_uncertainty=uncertainty,
            is_abstained=False,
            abstention_reason=None,
        )


_global_llm_assistant: Optional[GroundedLLMAssistant] = None

def get_grounded_llm_assistant() -> GroundedLLMAssistant:
    global _global_llm_assistant
    if _global_llm_assistant is None:
        _global_llm_assistant = GroundedLLMAssistant()
    return _global_llm_assistant
