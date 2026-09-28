from __future__ import annotations
"""
AHRAS Alert Intelligence Pipeline
==================================
Unified end-to-end orchestration pipeline for the Alert Intelligence Layer.
Ingests stream of raw alerts, performs adaptive deduplication, clusters
into multi-stage incidents, and evaluates exposure-aware triage priorities.
"""

from typing import Any, Dict, List, Optional, Tuple
from alert_intelligence.models import (
    ExposureMetric,
    IncidentCluster,
    RawAlert,
    TriageDecision
)
from alert_intelligence.deduplication import AdaptiveAlertDeduplicator
from alert_intelligence.clustering import AlertClusteringEngine
from alert_intelligence.prioritizer import ExposureAwarePrioritizer


class AlertIntelligencePipeline:
    """
    Production-grade Alert Intelligence Engine integrating deduplication,
    spatio-temporal clustering, and contextual exposure prioritization.
    """

    def __init__(
        self,
        dedup_window_seconds: float = 60.0,
        correlation_window_seconds: float = 300.0,
        asset_catalog: Optional[Dict[str, ExposureMetric]] = None
    ):
        self.clustering_engine = AlertClusteringEngine(correlation_window_seconds=correlation_window_seconds)
        self.prioritizer = ExposureAwarePrioritizer(asset_catalog=asset_catalog)
        self.deduplicator = AdaptiveAlertDeduplicator(
            window_seconds=dedup_window_seconds,
            on_evict=self._on_dedup_evict
        )

    def _on_dedup_evict(self, alert: RawAlert) -> None:
        """Invoked when deduplicator evicts an alert window to guarantee zero evidence loss."""
        cluster = self.clustering_engine.correlate(alert)
        self.prioritizer.prioritize(cluster)

    def flush(self) -> List[IncidentCluster]:
        """Flushes remaining deduplicated alerts into clusters and returns all active clusters."""
        flushed_alerts = self.deduplicator.flush()
        for a in flushed_alerts:
            if a.suppressed_count > 0:
                cluster = self.clustering_engine.correlate(a)
                self.prioritizer.prioritize(cluster)
        return self.get_all_incidents()

    def register_asset_exposure(self, metric: ExposureMetric) -> None:
        """Registers asset exposure metadata (criticality tier, network zone, CVEs)."""
        self.prioritizer.register_asset(metric)

    def ingest_alert(self, alert: RawAlert) -> Tuple[Optional[IncidentCluster], Optional[TriageDecision]]:
        """
        Ingests a single RawAlert.
        
        Returns:
            Tuple[Optional[IncidentCluster], Optional[TriageDecision]]:
                - (cluster, triage_decision) if the alert was emitted (new or significant milestone)
                - (None, None) if suppressed by deduplicator.
        """
        emitted_alert, is_new = self.deduplicator.process_alert(alert)
        if emitted_alert is None:
            # Alert duplicate suppressed
            return None, None

        # Correlate into cluster
        cluster = self.clustering_engine.correlate(emitted_alert)

        # Prioritize cluster
        decision = self.prioritizer.prioritize(cluster)

        return cluster, decision

    def ingest_batch(self, alerts: List[RawAlert]) -> List[Tuple[IncidentCluster, TriageDecision]]:
        """Ingests a batch of alerts and returns the list of affected clusters and triage decisions."""
        results = []
        for alert in alerts:
            cluster, decision = self.ingest_alert(alert)
            if cluster and decision:
                results.append((cluster, decision))
        return results

    def get_all_incidents(self) -> List[IncidentCluster]:
        """Returns all currently active incident clusters sorted by priority score (descending)."""
        clusters = self.clustering_engine.get_all_clusters()
        return sorted(clusters, key=lambda c: c.priority_score, reverse=True)

    def get_incident(self, cluster_id: str) -> Optional[IncidentCluster]:
        """Fetches a specific incident cluster by ID."""
        return self.clustering_engine.get_cluster(cluster_id)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns operational telemetry across deduplication, clustering, and triage."""
        dedup_stats = self.deduplicator.metrics
        all_clusters = self.clustering_engine.get_all_clusters()
        
        triage_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for c in all_clusters:
            triage_counts[c.triage_level.value] = triage_counts.get(c.triage_level.value, 0) + 1

        return {
            "deduplication": dedup_stats,
            "active_incident_count": len(all_clusters),
            "triage_distribution": triage_counts,
            "mean_priority_score": (
                round(sum(c.priority_score for c in all_clusters) / len(all_clusters), 4)
                if all_clusters else 0.0
            )
        }
