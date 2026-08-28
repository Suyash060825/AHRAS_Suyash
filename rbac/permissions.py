from __future__ import annotations
"""
AHRAS RBAC — Granular Permissions & Role Definitions
------------------------------------------------------
Implements 5 enterprise SOC roles and 40+ granular permissions for zero-trust
access control across ingestion, detection, risk, SOAR, hunting, and management.
"""

from enum import Enum
from typing import Dict, Set, List


class Perm(str, Enum):
    # Detection & Events
    EVENTS_READ          = "events:read"
    EVENTS_INGEST        = "events:ingest"
    DETECTIONS_READ      = "detections:read"
    DETECTIONS_TRIGGER   = "detections:trigger"
    
    # Alerts & Cases
    ALERTS_READ          = "alerts:read"
    ALERTS_ACKNOWLEDGE   = "alerts:acknowledge"
    ALERTS_DISMISS       = "alerts:dismiss"
    CASES_READ           = "cases:read"
    CASES_CREATE         = "cases:create"
    CASES_UPDATE         = "cases:update"
    CASES_CLOSE          = "cases:close"
    
    # Risk & Scoring
    RISK_READ            = "risk:read"
    RISK_SCORE           = "risk:score"
    RISK_EXPLAIN         = "risk:explain"
    RISK_TUNE            = "risk:tune"
    TRUST_READ           = "trust:read"
    TRUST_MODIFY         = "trust:modify"
    
    # Threat Intelligence & IOCs
    IOC_READ             = "ioc:read"
    IOC_CREATE           = "ioc:create"
    IOC_DELETE           = "ioc:delete"
    TI_ENRICH            = "ti:enrich"
    TI_FEEDS_MANAGE      = "ti:feeds_manage"
    
    # Threat Hunting & Forensics
    HUNT_EXECUTE         = "hunt:execute"
    FORENSICS_READ       = "forensics:read"
    FORENSICS_ANALYZE    = "forensics:analyze"
    GRAPH_QUERY          = "graph:query"
    
    # Forecasting & Analytics
    FORECAST_READ        = "forecast:read"
    FORECAST_EXECUTE     = "forecast:execute"
    REPORTS_GENERATE     = "reports:generate"
    METRICS_READ         = "metrics:read"
    
    # Active Defense / SOAR
    SOAR_READ            = "soar:read"
    SOAR_EXECUTE         = "soar:execute"
    SOAR_APPROVE         = "soar:approve"
    FIREWALL_CONTROL     = "firewall:control"
    HONEYPOT_MANAGE      = "honeypot:manage"
    
    # System Administration & RBAC
    USERS_MANAGE         = "users:manage"
    ROLES_MANAGE         = "roles:manage"
    SYSTEM_CONFIG        = "system:config"
    AUDIT_READ           = "audit:read"
    BASELINE_RESET       = "baseline:reset"


class Role(str, Enum):
    ADMIN               = "admin"
    SOC_ANALYST         = "soc_analyst"
    THREAT_HUNTER       = "threat_hunter"
    INCIDENT_RESPONDER  = "incident_responder"
    MANAGER             = "manager"


ROLE_PERMISSIONS: Dict[Role, Set[Perm]] = {
    Role.ADMIN: set(Perm),
    
    Role.SOC_ANALYST: {
        Perm.EVENTS_READ, Perm.EVENTS_INGEST, Perm.DETECTIONS_READ, Perm.DETECTIONS_TRIGGER,
        Perm.ALERTS_READ, Perm.ALERTS_ACKNOWLEDGE, Perm.ALERTS_DISMISS,
        Perm.CASES_READ, Perm.CASES_CREATE, Perm.CASES_UPDATE,
        Perm.RISK_READ, Perm.RISK_SCORE, Perm.RISK_EXPLAIN, Perm.TRUST_READ,
        Perm.IOC_READ, Perm.TI_ENRICH,
        Perm.GRAPH_QUERY, Perm.METRICS_READ, Perm.REPORTS_GENERATE,
        Perm.SOAR_READ, Perm.FORECAST_READ,
    },
    
    Role.THREAT_HUNTER: {
        Perm.EVENTS_READ, Perm.DETECTIONS_READ, Perm.ALERTS_READ,
        Perm.CASES_READ, Perm.CASES_CREATE, Perm.CASES_UPDATE,
        Perm.RISK_READ, Perm.RISK_SCORE, Perm.RISK_EXPLAIN, Perm.TRUST_READ,
        Perm.IOC_READ, Perm.IOC_CREATE, Perm.TI_ENRICH, Perm.TI_FEEDS_MANAGE,
        Perm.HUNT_EXECUTE, Perm.FORENSICS_READ, Perm.FORENSICS_ANALYZE, Perm.GRAPH_QUERY,
        Perm.FORECAST_READ, Perm.FORECAST_EXECUTE, Perm.REPORTS_GENERATE, Perm.METRICS_READ,
    },
    
    Role.INCIDENT_RESPONDER: {
        Perm.EVENTS_READ, Perm.DETECTIONS_READ, Perm.ALERTS_READ, Perm.ALERTS_ACKNOWLEDGE,
        Perm.CASES_READ, Perm.CASES_CREATE, Perm.CASES_UPDATE, Perm.CASES_CLOSE,
        Perm.RISK_READ, Perm.RISK_SCORE, Perm.RISK_EXPLAIN, Perm.TRUST_READ,
        Perm.IOC_READ, Perm.IOC_CREATE, Perm.TI_ENRICH,
        Perm.FORENSICS_READ, Perm.FORENSICS_ANALYZE, Perm.GRAPH_QUERY,
        Perm.SOAR_READ, Perm.SOAR_EXECUTE, Perm.SOAR_APPROVE, Perm.FIREWALL_CONTROL,
        Perm.HONEYPOT_MANAGE, Perm.REPORTS_GENERATE, Perm.METRICS_READ,
    },
    
    Role.MANAGER: {
        Perm.EVENTS_READ, Perm.DETECTIONS_READ, Perm.ALERTS_READ, Perm.CASES_READ,
        Perm.RISK_READ, Perm.TRUST_READ, Perm.FORECAST_READ,
        Perm.REPORTS_GENERATE, Perm.METRICS_READ, Perm.AUDIT_READ,
    },
}


def role_has_permission(role_name: str, perm: Perm) -> bool:
    try:
        r = Role(role_name.lower())
        return perm in ROLE_PERMISSIONS.get(r, set())
    except ValueError:
        return False
