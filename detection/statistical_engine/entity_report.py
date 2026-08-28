from __future__ import annotations
"""
AHRAS Entity Report Generator
-----------------------------
Synthesizes statistical baselines, peer group comparisons, multi-day trend ramps,
and seasonality shifts into a unified per-entity report.

Primary Uses:
  1. SOC Dashboard: JSON API payload for security analysts investigating an entity.
  2. Paper Evaluation Appendix: Clean Markdown report format detailing multi-modal
     statistical evidence and MITRE ATT&CK mapping for evaluation case studies.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from detection.statistical_engine.stat_engine import get_statistical_engine, StatResult
from detection.statistical_engine.peer_group import get_peer_group_engine, PeerGroupResult
from detection.statistical_engine.trend_engine import get_trend_engine, TrendResult

log = logging.getLogger(__name__)


@dataclass
class EntityReport:
    """Unified per-entity security report."""
    entity_key:              str
    ocsf_class:              str
    generated_at:            str
    overall_anomaly_status:  bool
    overall_confidence:      float
    active_flags:            list[str]
    mitre_techniques:        list[str]
    
    # Engine specific details
    statistical_summary:     dict
    peer_group_summary:      dict
    trend_seasonality_summary: dict
    
    # Plain English narrative
    summary_paragraph:       str

    def to_dict(self) -> dict:
        return {
            "entity_key":             self.entity_key,
            "ocsf_class":             self.ocsf_class,
            "generated_at":           self.generated_at,
            "overall_anomaly_status": self.overall_anomaly_status,
            "overall_confidence":     self.overall_confidence,
            "active_flags":           self.active_flags,
            "mitre_techniques":       self.mitre_techniques,
            "statistical_summary":    self.statistical_summary,
            "peer_group_summary":     self.peer_group_summary,
            "trend_seasonality_summary": self.trend_seasonality_summary,
            "summary_paragraph":      self.summary_paragraph,
        }

    def to_markdown(self) -> str:
        md = []
        md.append(f"# Entity Security Report: `{self.entity_key}`")
        md.append(f"**OCSF Class:** {self.ocsf_class} | **Generated At:** {self.generated_at}")
        md.append(f"**Anomaly Status:** {'🚨 ANOMALOUS' if self.overall_anomaly_status else '✅ NORMAL'} | **Overall Confidence:** {self.overall_confidence:.1%}")
        md.append("")
        
        md.append("## Executive Summary")
        md.append(self.summary_paragraph)
        md.append("")
        
        if self.mitre_techniques:
            md.append("## MITRE ATT&CK Mapping")
            md.append(", ".join([f"`{t}`" for t in self.mitre_techniques]))
            md.append("")
            
        md.append("## Statistical Baseline Metrics")
        for k, v in self.statistical_summary.items():
            md.append(f"- **{k}:** {v}")
        md.append("")

        md.append("## Peer Group Cohort Metrics")
        for k, v in self.peer_group_summary.items():
            md.append(f"- **{k}:** {v}")
        md.append("")

        md.append("## Trend & Seasonality Metrics")
        for k, v in self.trend_seasonality_summary.items():
            md.append(f"- **{k}:** {v}")
        md.append("")

        return "\n".join(md)


class EntityReportGenerator:
    """
    Combines output from StatisticalEngine, PeerGroupEngine, and TrendEngine
    to generate complete EntityReport objects.
    """

    def __init__(self):
        self.stat_engine = get_statistical_engine()
        self.peer_engine = get_peer_group_engine()
        self.trend_engine = get_trend_engine()

    def generate_report(self, cls: str, entity_key: str, last_evt: dict, last_vec: Any = None) -> EntityReport:
        # Run scoring on all three engines for latest event
        stat_res: StatResult = self.stat_engine.score(last_evt, last_vec)
        
        metric_val = 0.0
        if cls == "network_activity":
            dur = float(last_evt.get("traffic", {}).get("duration_sec", 1) or 1)
            metric_val = float(last_evt.get("traffic", {}).get("packets", 0)) / max(dur, 0.001)
        elif cls == "cloud_api":
            metric_val = 1.0

        peer_res: PeerGroupResult = self.peer_engine.update_and_score(last_evt, entity_key, metric_val)
        trend_res: TrendResult = self.trend_engine.update_and_score(last_evt, entity_key, metric_val)

        # Merge active flags and MITRE techniques
        active_flags = sorted(list(set(stat_res.flags + peer_res.flags + trend_res.flags)))
        mitre_techs  = sorted(list(set(stat_res.mitre_techniques + peer_res.mitre_techniques + trend_res.mitre_techniques)))

        is_anom = stat_res.is_anomaly or peer_res.is_peer_anomaly or trend_res.is_trend_ramp or trend_res.is_seasonality_anomaly
        overall_conf = max(stat_res.confidence, peer_res.confidence, trend_res.confidence)

        # Generate plain-English summary paragraph
        summary_p = self._generate_narrative(entity_key, cls, is_anom, active_flags, mitre_techs, stat_res, peer_res, trend_res)

        now_str = datetime.now(timezone.utc).isoformat()

        return EntityReport(
            entity_key=entity_key,
            ocsf_class=cls,
            generated_at=now_str,
            overall_anomaly_status=is_anom,
            overall_confidence=overall_conf,
            active_flags=active_flags,
            mitre_techniques=mitre_techs,
            statistical_summary={
                "zscore":             stat_res.zscore,
                "ewma_deviation":     stat_res.ewma_deviation,
                "temporal_density":   stat_res.temporal_density,
                "behavioral_drift":   stat_res.behavioral_drift,
                "circadian_anomaly":  stat_res.circadian_anomaly,
                "beacon_regularity":  stat_res.beacon_regularity,
                "volume_score":       stat_res.volume_score,
                "baseline_n":         stat_res.baseline_n,
            },
            peer_group_summary={
                "peer_group_key":     peer_res.peer_group_key,
                "peer_mean":          peer_res.peer_mean,
                "peer_std":           peer_res.peer_std,
                "peer_zscore":        peer_res.peer_zscore,
                "peer_sample_count":  peer_res.peer_sample_count,
                "is_peer_anomaly":    peer_res.is_peer_anomaly,
            },
            trend_seasonality_summary={
                "slope":                  trend_res.slope,
                "r2_score":               trend_res.r2_score,
                "is_trend_ramp":          trend_res.is_trend_ramp,
                "is_seasonality_anomaly": trend_res.is_seasonality_anomaly,
                "weekday_mean":           trend_res.weekday_mean,
                "weekend_mean":           trend_res.weekend_mean,
            },
            summary_paragraph=summary_p,
        )

    def _generate_narrative(
        self, key: str, cls: str, is_anom: bool, flags: list[str], mitre: list[str],
        stat: StatResult, peer: PeerGroupResult, trend: TrendResult
    ) -> str:
        if not is_anom:
            return (
                f"Entity '{key}' ({cls}) exhibits normal behavioral patterns consistent with "
                f"established baseline metrics (Z-score: {stat.zscore:.1f}, Drift: {stat.behavioral_drift:.2f}). "
                f"No statistical, peer group, or multi-day trend anomalies detected."
            )

        anom_reasons = []
        if stat.flags:
            anom_reasons.append(f"baseline metric spikes ({', '.join(stat.flags)})")
        if peer.is_peer_anomaly:
            anom_reasons.append(f"deviation from peer cohort {peer.peer_group_key} (Z_peer={peer.peer_zscore:.1f})")
        if trend.is_trend_ramp:
            anom_reasons.append(f"a multi-day gradual trend ramp (slope={trend.slope:.2f}/day, R²={trend.r2_score:.2f})")
        if trend.is_seasonality_anomaly:
            anom_reasons.append(f"unusual weekend activity (weekend avg {trend.weekend_mean:.1f} vs weekday avg {trend.weekday_mean:.1f})")

        reasons_str = "; ".join(anom_reasons)
        mitre_str = f" Mapped to MITRE ATT&CK techniques: {', '.join(mitre)}." if mitre else ""

        return (
            f"ALERT: Entity '{key}' ({cls}) has been flagged as anomalous with a confidence score of "
            f"{max(stat.confidence, peer.confidence, trend.confidence):.1%}. Anomalous behaviors identified: "
            f"{reasons_str}.{mitre_str} SOC analyst review and endpoint isolation recommended."
        )


_report_generator_instance: Optional[EntityReportGenerator] = None


def get_entity_report_generator() -> EntityReportGenerator:
    global _report_generator_instance
    if _report_generator_instance is None:
        _report_generator_instance = EntityReportGenerator()
    return _report_generator_instance
