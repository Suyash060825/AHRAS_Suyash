from __future__ import annotations
"""
AHRAS Module — Case-Based Security Memory Engine (Section 48 / Research Frontier P2)
--------------------------------------------------------------------------------------
Maintains a structured historical repository of closed security incidents:
  - Attack graph topologies
  - MITRE ATT&CK technique progression
  - Timeline bounds
  - Cryptographic evidence hashes
  - Applied response actions and verified operational outcomes
  - Model versions active during the incident

Enables historical case retrieval for active incidents to inform:
  1. Incident Investigation & triage prioritization
  2. Grounded Explanation synthesis
  3. Response action recommendation with historical efficacy priors

Strict Temporal Invariant:
  "Never use historical cases containing future information relative to an evaluation event."
  Every query requires an evaluation timestamp; cases closed after this timestamp are strictly filtered out.
"""

import collections
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class HistoricalSecurityCase:
    case_id: str
    incident_id: str
    title: str
    techniques: List[str]                  # MITRE technique IDs
    affected_entities: List[str]           # Affected assets/hosts
    timeline_start: float
    timeline_end: float
    closed_at: float                       # Must be <= query_timestamp to prevent future data leakage
    evidence_ids: List[str]
    applied_responses: List[str]           # Playbook actions taken (e.g. ["ISOLATE_HOST", "KILL_PROCESS"])
    outcome_verdict: str                   # "SUCCESSFUL_CONTAINMENT", "PARTIAL_CONTAINMENT", "COLLATERAL_IMPACT"
    mean_risk_score: float
    model_versions: Dict[str, str] = field(default_factory=dict)
    lessons_learned: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseRetrievalResult:
    case_id: str
    incident_id: str
    title: str
    technique_similarity: float
    structural_similarity: float
    composite_similarity: float
    outcome_verdict: str
    recommended_responses: List[str]
    historical_lessons: str
    case_closed_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "incident_id": self.incident_id,
            "title": self.title,
            "composite_similarity": round(self.composite_similarity, 4),
            "technique_similarity": round(self.technique_similarity, 4),
            "structural_similarity": round(self.structural_similarity, 4),
            "outcome_verdict": self.outcome_verdict,
            "recommended_responses": self.recommended_responses,
            "historical_lessons": self.historical_lessons,
            "case_closed_at": self.case_closed_at,
        }


class CaseBasedSecurityMemory:
    """
    Indexed memory of past security cases enforcing strict temporal causality guards.
    """

    def __init__(self) -> None:
        self._cases: Dict[str, HistoricalSecurityCase] = {}

    def store_case(self, case: HistoricalSecurityCase) -> None:
        """Stores a resolved incident into the case memory."""
        self._cases[case.case_id] = case

    def retrieve_similar_cases(
        self,
        current_techniques: List[str],
        current_entities: List[str],
        query_timestamp: float,
        k: int = 3,
        min_similarity: float = 0.20,
    ) -> List[CaseRetrievalResult]:
        """
        Retrieves top-k historical cases similar to active incident.
        Strictly excludes any cases with closed_at > query_timestamp (temporal safety invariant).
        """
        results: List[CaseRetrievalResult] = []
        set_curr_tech = set(current_techniques)
        set_curr_ent = set(current_entities)

        for case_id, case in self._cases.items():
            # STRICT TEMPORAL INVARIANT CHECK:
            # Never use historical cases containing future information relative to evaluation event!
            if case.closed_at > query_timestamp:
                continue

            set_hist_tech = set(case.techniques)
            set_hist_ent = set(case.affected_entities)

            # Technique Jaccard
            union_tech = set_curr_tech.union(set_hist_tech)
            inter_tech = set_curr_tech.intersection(set_hist_tech)
            tech_sim = len(inter_tech) / max(1, len(union_tech))

            # Entity Overlap
            union_ent = set_curr_ent.union(set_hist_ent)
            inter_ent = set_curr_ent.intersection(set_hist_ent)
            ent_sim = len(inter_ent) / max(1, len(union_ent))

            composite = 0.65 * tech_sim + 0.35 * ent_sim
            if composite < min_similarity:
                continue

            results.append(
                CaseRetrievalResult(
                    case_id=case.case_id,
                    incident_id=case.incident_id,
                    title=case.title,
                    technique_similarity=tech_sim,
                    structural_similarity=ent_sim,
                    composite_similarity=composite,
                    outcome_verdict=case.outcome_verdict,
                    recommended_responses=case.applied_responses,
                    historical_lessons=case.lessons_learned,
                    case_closed_at=case.closed_at,
                )
            )

        results.sort(key=lambda r: r.composite_similarity, reverse=True)
        return results[:k]

    def synthesize_response_recommendation(
        self,
        retrieved_cases: List[CaseRetrievalResult],
    ) -> Dict[str, Any]:
        """Synthesizes high-confidence playbook recommendations based on historical containment outcomes."""
        if not retrieved_cases:
            return {"recommended_actions": [], "confidence": 0.0, "prior_verdicts": {}}

        action_scores: Dict[str, float] = collections.defaultdict(float)
        verdicts = collections.Counter([c.outcome_verdict for c in retrieved_cases])

        for case in retrieved_cases:
            # Successful outcomes boost action score; collateral damage dampens it
            outcome_mult = 1.0 if case.outcome_verdict == "SUCCESSFUL_CONTAINMENT" else (0.5 if case.outcome_verdict == "PARTIAL_CONTAINMENT" else -0.5)
            for action in case.recommended_responses:
                action_scores[action] += case.composite_similarity * outcome_mult

        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        top_actions = [a for a, s in sorted_actions if s > 0]

        mean_sim = sum(c.composite_similarity for c in retrieved_cases) / len(retrieved_cases)
        return {
            "recommended_actions": top_actions,
            "confidence": round(mean_sim, 3),
            "prior_verdicts": dict(verdicts),
            "grounding_case_count": len(retrieved_cases),
        }
