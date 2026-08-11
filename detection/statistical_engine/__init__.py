from __future__ import annotations
"""
AHRAS Statistical Engine Package
--------------------------------
Exporting public API for baseline tracking, peer cohort analysis,
linear trend & seasonality detection, and unified SOC entity reporting.
"""

from detection.statistical_engine.stat_engine import (
    StatResult,
    StatisticalEngine,
    get_statistical_engine,
    run_statistical_engine,
)

from detection.statistical_engine.peer_group import (
    PeerGroupResult,
    PeerGroupEngine,
    get_peer_group_engine,
)

from detection.statistical_engine.trend_engine import (
    TrendResult,
    TrendEngine,
    get_trend_engine,
)

from detection.statistical_engine.entity_report import (
    EntityReport,
    EntityReportGenerator,
    get_entity_report_generator,
)

__all__ = [
    "StatResult",
    "StatisticalEngine",
    "get_statistical_engine",
    "run_statistical_engine",
    "PeerGroupResult",
    "PeerGroupEngine",
    "get_peer_group_engine",
    "TrendResult",
    "TrendEngine",
    "get_trend_engine",
    "EntityReport",
    "EntityReportGenerator",
    "get_entity_report_generator",
]
