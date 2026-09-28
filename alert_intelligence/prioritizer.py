from __future__ import annotations
"""
AHRAS Exposure-Aware Alert Prioritizer
======================================
Calibrates incident triage priority based on:
1. Multi-stage attack progression depth across MITRE ATT&CK tactics.
2. Asset business criticality (Tier 1 Crown Jewels vs Tier 3 Workstations).
3. Network zone exposure (Internet-facing / DMZ vs Isolated).
4. Known CVE vulnerabilities and CVSS exploitability.
5. Epistemic uncertainty dampening to prevent false-positive alert escalation.
"""

from typing import Dict, List, Optional
from alert_intelligence.models import (
    ExposureMetric,
    IncidentCluster,
    NetworkZone,
    TriageDecision,
    TriageLevel
)

# Zone exposure multipliers
ZONE_EXPOSURE_WEIGHTS = {
    NetworkZone.INTERNET_FACING: 1.00,
    NetworkZone.DMZ:             0.85,
    NetworkZone.INTERNAL_PROD:   0.70,
    NetworkZone.CORPORATE_LAN:   0.50,
    NetworkZone.ISOLATED_SECURE: 0.30,
}

# Criticality tier multipliers
CRITICALITY_TIER_WEIGHTS = {
    1: 1.00, # Tier 1: Domain Controller, Key Vault, Core DB
    2: 0.70, # Tier 2: Application Cluster, CI/CD Pipeline
    3: 0.40, # Tier 3: Endpoint Workstation, Ephemeral Test Box
}


class ExposureAwarePrioritizer:
    """
    Context-aware prioritization engine that computes calibrated triage scores
    and assigns actionable response recommendations to IncidentClusters.
    """

    def __init__(self, asset_catalog: Optional[Dict[str, ExposureMetric]] = None):
        self.asset_catalog: Dict[str, ExposureMetric] = asset_catalog or {}

    def register_asset(self, metric: ExposureMetric) -> None:
        """Registers or updates asset exposure context."""
        self.asset_catalog[metric.entity_key] = metric

    def prioritize(self, cluster: IncidentCluster) -> TriageDecision:
        """
        Evaluates an IncidentCluster and updates its priority_score, triage_level,
        and returns a structured TriageDecision.
        """
        # 1. Detector Risk & Severity Contribution (Root-Mean-Square of top alerts)
        scores = [a.normalized_score for a in cluster.alerts]
        if scores:
            scores_sorted = sorted(scores, reverse=True)
            top_scores = scores_sorted[:min(5, len(scores_sorted))]
            detector_score = (sum(s ** 2 for s in top_scores) / len(top_scores)) ** 0.5
        else:
            detector_score = 0.0

        # Mean epistemic uncertainty
        uncertainties = [a.uncertainty for a in cluster.alerts]
        mean_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0

        # 2. Progression Depth
        progression_score = cluster.progression_score

        # 3. Asset Criticality and Network Zone Exposure
        max_crit = 0.40 # Default Tier 3
        max_expo = 0.50 # Default Corporate LAN
        max_cve_factor = 0.0

        for ent in cluster.entity_keys:
            metric = self.asset_catalog.get(ent)
            if metric:
                crit_val = CRITICALITY_TIER_WEIGHTS.get(metric.asset_criticality_tier, 0.40)
                expo_val = ZONE_EXPOSURE_WEIGHTS.get(metric.network_zone, 0.50)
                if metric.is_internet_exposed:
                    expo_val = max(expo_val, 0.95)
                cve_val = min(1.0, metric.max_cve_cvss / 10.0)
                
                max_crit = max(max_crit, crit_val)
                max_expo = max(max_expo, expo_val)
                max_cve_factor = max(max_cve_factor, cve_val)

        # Combined exposure factor
        exposure_factor = min(1.0, 0.60 * max_expo + 0.40 * max_cve_factor)

        # 4. Calibrated Multi-Objective Priority Equation
        # Weighting: 35% Detector Severity, 25% Progression, 20% Asset Criticality, 20% Exposure
        raw_priority = (
            0.35 * detector_score +
            0.25 * progression_score +
            0.20 * max_crit +
            0.20 * exposure_factor
        )
        # Apply uncertainty dampening (up to 30% reduction if model is completely uncertain)
        uncertainty_discount = 1.0 - (0.30 * mean_uncertainty)
        calibrated_priority = max(0.0, min(1.0, raw_priority * uncertainty_discount))

        # 5. Triage Tier Classification
        if calibrated_priority >= 0.80:
            level = TriageLevel.CRITICAL
            recommended_action = "IMMEDIATE_CONTAINMENT_DISPATCH"
        elif calibrated_priority >= 0.60:
            level = TriageLevel.HIGH
            recommended_action = "STAGE_AUTOMATION_AND_ENGAGE_HONEYPOT"
        elif calibrated_priority >= 0.35:
            level = TriageLevel.MEDIUM
            recommended_action = "ACTIVE_PROVENANCE_TRACKING"
        elif calibrated_priority >= 0.15:
            level = TriageLevel.LOW
            recommended_action = "CONTINUOUS_MONITORING"
        else:
            level = TriageLevel.INFO
            recommended_action = "RECORD_TELEMETRY"

        # Update cluster in-place
        cluster.exposure_score = round(exposure_factor, 4)
        cluster.aggregate_risk = round(detector_score, 4)
        cluster.priority_score = round(calibrated_priority, 4)
        cluster.triage_level = level
        cluster.recommended_action = recommended_action
        
        # Build explanation rationale
        rationale = (
            f"Triage Priority {calibrated_priority:.3f} ({level.value}) derived from: "
            f"Detector Severity={detector_score:.3f}, Progression={progression_score:.3f} "
            f"({len(cluster.tactics_present)} tactics), Asset Criticality={max_crit:.2f}, "
            f"Zone Exposure={exposure_factor:.2f}, Epistemic Uncertainty={mean_uncertainty:.3f}."
        )
        cluster.explanation_summary = rationale

        # Blast radius estimate: proportional to entity count and asset tier
        blast_radius = min(1.0, (len(cluster.entity_keys) * 0.15) + (max_crit * 0.50))

        return TriageDecision(
            cluster_id=cluster.cluster_id,
            triage_level=level,
            priority_score=round(calibrated_priority, 4),
            recommended_action=recommended_action,
            rationale=rationale,
            mitre_coverage_summary={
                "tactics": list(cluster.tactics_present),
                "techniques": list(cluster.techniques_present)
            },
            evidence_count=len(cluster.evidence_ids),
            blast_radius_estimate=round(blast_radius, 4)
        )
