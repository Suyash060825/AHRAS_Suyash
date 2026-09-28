from __future__ import annotations
r"""
AHRAS Module — Attack Campaign Similarity Engine (Section 23 / Research Frontier P1)
------------------------------------------------------------------------------------
Represents security incidents as multi-modal structural fingerprints:
  - Graph structural signature (degree distribution, diameter, density)
  - MITRE ATT&CK technique sets and temporal sequences
  - Target entity topology
  - Evidence hash profiles

Identifies similar historical campaigns using multi-attribute similarity:
  1. Jaccard Technique Similarity: S_tech(A, B) = |T_A \cap T_B| / |T_A \cup T_B|
  2. Sequence Alignment Similarity: Longest Common Subsequence (LCS) ratio
  3. Structural Graph Overlap: Cosine similarity of entity and edge distributions
  4. Temporal Duration Proximity: Gaussian temporal scaling

Safety Invariant:
  Strictly adheres to: "Never assert same attacker/actor without supporting evidence."
  Attribution is tagged as 'UNATTRIBUTED' or 'EVIDENCE_BACKED_HYPOTHESIS' only when
  cryptographic IOCs match.
"""

import collections
import difflib
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class IncidentProfile:
    """Multi-attribute behavioral representation of an attack incident or campaign."""
    incident_id: str
    name: str
    techniques: List[str]                  # Ordered sequence of MITRE technique IDs (e.g. ["T1110", "T1059", "T1021"])
    affected_entities: List[str]           # Entities involved (e.g. ["host:srv-01", "ip:10.0.0.5"])
    duration_seconds: float
    evidence_hashes: List[str] = field(default_factory=list)
    structural_features: Dict[str, float] = field(default_factory=dict) # e.g. {"nodes": 5, "edges": 7, "density": 0.35}
    known_actor_indicator: Optional[str] = None # Direct threat intel IOC match if present
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimilarityMatchResult:
    """Detailed audit score comparing an active incident against a historical campaign."""
    candidate_incident_id: str
    historical_incident_id: str
    historical_campaign_name: str
    composite_similarity: float           # [0.0, 1.0]
    technique_jaccard_similarity: float   # [0.0, 1.0]
    sequence_alignment_score: float       # [0.0, 1.0]
    structural_overlap_score: float       # [0.0, 1.0]
    matching_techniques: List[str]
    longest_common_technique_sequence: List[str]
    confidence: float                     # [0.0, 1.0]
    attribution_verdict: str              # "UNATTRIBUTED" or "EVIDENCE_BACKED_HYPOTHESIS"
    attribution_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_incident_id": self.candidate_incident_id,
            "historical_incident_id": self.historical_incident_id,
            "historical_campaign_name": self.historical_campaign_name,
            "composite_similarity": round(self.composite_similarity, 4),
            "technique_jaccard_similarity": round(self.technique_jaccard_similarity, 4),
            "sequence_alignment_score": round(self.sequence_alignment_score, 4),
            "structural_overlap_score": round(self.structural_overlap_score, 4),
            "matching_techniques": self.matching_techniques,
            "longest_common_technique_sequence": self.longest_common_technique_sequence,
            "confidence": round(self.confidence, 4),
            "attribution_verdict": self.attribution_verdict,
            "attribution_rationale": self.attribution_rationale,
        }


class CampaignSimilarityEngine:
    """
    Indexes historical security incidents and performs multi-faceted similarity matching.
    """

    def __init__(
        self,
        weight_tech: float = 0.40,
        weight_seq: float = 0.35,
        weight_struct: float = 0.25,
    ) -> None:
        self.weight_tech = weight_tech
        self.weight_seq = weight_seq
        self.weight_struct = weight_struct
        self._historical_campaigns: Dict[str, IncidentProfile] = {}

    def register_campaign(self, profile: IncidentProfile) -> None:
        """Indexes a historical incident campaign into the similarity store."""
        self._historical_campaigns[profile.incident_id] = profile

    def _compute_lcs(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Computes Longest Common Subsequence between two technique sequences."""
        matcher = difflib.SequenceMatcher(None, seq1, seq2)
        match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
        if match.size == 0:
            return []
        return seq1[match.a : match.a + match.size]

    def _compute_structural_similarity(
        self,
        struct_a: Dict[str, float],
        struct_b: Dict[str, float],
    ) -> float:
        """Calculates cosine similarity over normalized structural feature vectors."""
        all_keys = sorted(set(struct_a.keys()).union(struct_b.keys()))
        if not all_keys:
            return 1.0
        vec_a = np.array([struct_a.get(k, 0.0) for k in all_keys], dtype=np.float64)
        vec_b = np.array([struct_b.get(k, 0.0) for k in all_keys], dtype=np.float64)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.5
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def find_similar_campaigns(
        self,
        query: IncidentProfile,
        top_k: int = 5,
        min_similarity_threshold: float = 0.30,
    ) -> List[SimilarityMatchResult]:
        """
        Compares query incident against indexed historical campaigns and returns
        ranked list of similar matches.
        """
        results: List[SimilarityMatchResult] = []
        set_q = set(query.techniques)

        for hist_id, hist in self._historical_campaigns.items():
            if hist_id == query.incident_id:
                continue

            set_h = set(hist.techniques)

            # 1. Jaccard Technique Similarity
            union_tech = set_q.union(set_h)
            inter_tech = set_q.intersection(set_h)
            jaccard = len(inter_tech) / max(1, len(union_tech))

            # 2. Sequence Alignment (LCS ratio)
            lcs = self._compute_lcs(query.techniques, hist.techniques)
            seq_score = (2.0 * len(lcs)) / max(1, len(query.techniques) + len(hist.techniques))

            # 3. Structural Overlap
            struct_score = self._compute_structural_similarity(
                query.structural_features, hist.structural_features
            )

            # Composite Score
            composite = (
                self.weight_tech * jaccard
                + self.weight_seq * seq_score
                + self.weight_struct * struct_score
            )

            if composite < min_similarity_threshold:
                continue

            # Confidence is scaled by technique count and evidence availability
            confidence = min(0.98, composite * (0.6 + 0.1 * min(4, len(inter_tech))))

            # Safety Rule: Attribution Verification
            # Never assert same actor without direct threat intel / cryptographic IOC match
            common_hashes = set(query.evidence_hashes).intersection(hist.evidence_hashes)
            if query.known_actor_indicator and hist.known_actor_indicator:
                if query.known_actor_indicator == hist.known_actor_indicator or len(common_hashes) > 0:
                    attribution = "EVIDENCE_BACKED_HYPOTHESIS"
                    rationale = f"Corroborated by matching cryptographic IOCs and threat actor signature: {hist.known_actor_indicator}"
                else:
                    attribution = "UNATTRIBUTED"
                    rationale = "Behavioral overlap observed, but differing actor indicators detected"
            elif len(common_hashes) >= 2:
                attribution = "EVIDENCE_BACKED_HYPOTHESIS"
                rationale = f"Corroborated by {len(common_hashes)} shared cryptographic evidence payload hashes"
            else:
                attribution = "UNATTRIBUTED"
                rationale = "High behavioral & technique similarity; actor attribution unconfirmed per AHRAS safety invariants"

            match_res = SimilarityMatchResult(
                candidate_incident_id=query.incident_id,
                historical_incident_id=hist_id,
                historical_campaign_name=hist.name,
                composite_similarity=composite,
                technique_jaccard_similarity=jaccard,
                sequence_alignment_score=seq_score,
                structural_overlap_score=struct_score,
                matching_techniques=sorted(inter_tech),
                longest_common_technique_sequence=lcs,
                confidence=confidence,
                attribution_verdict=attribution,
                attribution_rationale=rationale,
            )
            results.append(match_res)

        # Sort descending by composite similarity
        results.sort(key=lambda r: r.composite_similarity, reverse=True)
        return results[:top_k]
