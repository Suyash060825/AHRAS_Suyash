"""
AHRAS Feature Importance Mapper
-------------------------------
Computes empirical field indispensability and sensitivity across all 10 ATT&CK tactics,
identifying which schema keys are indispensable vs redundant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from telemetry.telemetry_requirements import (
    TelemetryRequirementsRegistry,
    get_default_requirements_registry,
)


@dataclass
class FieldImportanceSummary:
    """
    Indispensability, sensitivity, and cross-tactic utility of an OCSF field.
    """
    field_name: str
    importance_score: float               # Normalized score 0.0 to 1.0
    tactics_supported: List[str]
    techniques_affected: List[str]
    classification: str                   # INDISPENSABLE, FP_REDUCTION, OPTIONAL_REDUNDANT
    failure_rate_if_removed: float        # Percentage of mapped vectors that collapse if removed


class FeatureImportanceMapper:
    """
    Evaluates field importance across detection rules and requirement profiles.
    """
    def __init__(self, registry: Optional[TelemetryRequirementsRegistry] = None) -> None:
        self.registry = registry or get_default_requirements_registry()

    def compute_field_importance(self) -> List[FieldImportanceSummary]:
        """
        Calculates cross-tactic indispensability metrics for all observed OCSF schema keys.
        """
        profiles = self.registry.get_all_profiles()
        total_profiles = len(profiles)

        # Track appearances
        indispensable_map: Dict[str, Set[str]] = {}  # field -> set(techniques)
        fp_map: Dict[str, Set[str]] = {}
        redundant_map: Dict[str, Set[str]] = {}
        tactics_map: Dict[str, Set[str]] = {}

        all_fields: Set[str] = set()

        for p in profiles:
            for f in p.minimal_necessary_fields:
                indispensable_map.setdefault(f, set()).add(p.technique_id)
                tactics_map.setdefault(f, set()).add(p.tactic)
                all_fields.add(f)
            for f in p.fp_reduction_fields:
                fp_map.setdefault(f, set()).add(p.technique_id)
                tactics_map.setdefault(f, set()).add(p.tactic)
                all_fields.add(f)
            for f in p.optional_redundant_fields:
                redundant_map.setdefault(f, set()).add(p.technique_id)
                tactics_map.setdefault(f, set()).add(p.tactic)
                all_fields.add(f)

        summaries: List[FieldImportanceSummary] = []

        for f in sorted(list(all_fields)):
            indisp_techs = indispensable_map.get(f, set())
            fp_techs = fp_map.get(f, set())
            red_techs = redundant_map.get(f, set())
            tactics = sorted(list(tactics_map.get(f, set())))

            collapse_count = len(indisp_techs)
            failure_rate = round((collapse_count / total_profiles) * 100.0, 2) if total_profiles else 0.0

            # Classification
            if len(indisp_techs) > 0:
                classification = "INDISPENSABLE"
                # Base score proportional to failure rate and cross-tactic spread
                base_score = 0.60 + 0.30 * (len(tactics) / 10.0) + 0.10 * (collapse_count / total_profiles)
            elif len(fp_techs) > 0:
                classification = "FP_REDUCTION"
                base_score = 0.35 + 0.20 * (len(tactics) / 10.0)
            else:
                classification = "OPTIONAL_REDUNDANT"
                base_score = 0.05

            affected_techs = sorted(list(indisp_techs.union(fp_techs)))

            summaries.append(FieldImportanceSummary(
                field_name=f,
                importance_score=round(min(1.0, base_score), 4),
                tactics_supported=tactics,
                techniques_affected=affected_techs,
                classification=classification,
                failure_rate_if_removed=failure_rate,
            ))

        # Sort by importance_score descending, then failure_rate descending
        summaries.sort(key=lambda s: (s.importance_score, s.failure_rate_if_removed), reverse=True)
        return summaries

    def get_top_indispensable_fields(self, top_k: int = 10) -> List[FieldImportanceSummary]:
        """
        Returns top K most critical fields across all tactics.
        """
        all_s = self.compute_field_importance()
        return [s for s in all_s if s.classification == "INDISPENSABLE"][:top_k]
