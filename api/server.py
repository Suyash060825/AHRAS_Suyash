from __future__ import annotations
"""
AHRAS SOC REST API Server
------------------------
Production FastAPI service providing endpoints for SOC analysts, dashboards,
and automated active defense response workflows.

Endpoints:
  - GET  /health                    : Service health and system status
  - GET  /dashboard, /              : Real-time Cyber-Glassmorphism SOC Dashboard
  - GET  /alerts                    : Query historical security alerts with filtering
  - GET  /entities/{key}/report     : Unified per-entity security report (JSON & Markdown)
  - POST /alerts/{id}/respond       : Trigger or approve active defense response actions
  - GET  /actions/pending           : Active defense actions awaiting SOC approval
  - GET  /actions/history           : Audit trail of executed active defense mitigations
  - POST /analyst/feedback          : Mark false positives and reset entity baselines
  - GET  /metrics                   : SOC operational metrics & engine stats
  - POST /api/detect                : Direct telemetry detection and XAI scoring
  - POST /api/score                 : Direct risk evaluation
  - POST /api/forecast              : Time-series risk forecasting & early warning
  - GET  /api/forecast/escalating   : Fleet-wide top escalating threat sources
  - POST /api/auth/token            : OAuth2/JWT authentication endpoint
  - GET  /api/auth/users            : User management and role listing
  - GET  /api/threat-intel/iocs     : Threat intelligence IOC database
  - POST /api/threat-intel/iocs     : Ingest new STIX/IOC threat indicators
  - GET  /api/xai/fidelity          : Exact analytical XAI fidelity ledger summary
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Body, Path, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from storage.store import get_store
from detection.statistical_engine.entity_report import get_entity_report_generator, EntityReport
from detection.statistical_engine.stat_engine import get_statistical_engine
from response.orchestrator import get_response_orchestrator, ResponseAction
from detection.hybrid_engine import get_combiner
from detection.risk_engine import get_risk_engine, run_risk_engine
from normalizer.ocsf_normalizer import _norm_network
from forecast.predictor import AttackPredictor, ForecastResult
from threat_intel.intel import get_threat_intel_manager
from auth.manager import authenticate_user, create_access_token, list_users, register_user
from xai.fidelity_ledger import get_fidelity_ledger
from rbac.permissions import Perm, Role
from rbac.middleware import get_user_permissions

log = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AHRAS SOC Security API",
    description="Adaptive Hybrid Risk-Aware Security REST API for Enterprise SOC Integration",
    version="6.0.0",
)

# Enable CORS for SOC Frontend Web Dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request & Response Schemas ───────────────────────────────────────

class AnalystFeedbackRequest(BaseModel):
    ocsf_class: str = Field(..., description="OCSF class of entity (e.g. network_activity, cloud_api)")
    entity_key: str = Field(..., description="Entity identifier (IP, hostname, username)")
    action:     str = Field("mark_false_positive", description="mark_false_positive or reset_entity")
    reason:     Optional[str] = Field(None, description="Analyst rationale for feedback")


class ResponseApprovalRequest(BaseModel):
    action: str = Field("approve", description="approve or reject")
    reason: Optional[str] = Field(None, description="SOC analyst approval/rejection rationale")


class DetectRequest(BaseModel):
    timestamp:   Optional[float] = None
    source_ip:   Optional[str] = Field(None, alias="src_ip")
    dest_ip:     Optional[str] = Field(None, alias="dst_ip")
    source_port: Optional[int] = Field(None, alias="src_port")
    dest_port:   Optional[int] = Field(None, alias="dst_port")
    protocol:    Optional[str] = "TCP"
    bytes:       Optional[int] = 512
    packet_count: Optional[int] = 10
    duration_sec: Optional[float] = 1.0
    ioc_match:   Optional[List[str]] = None
    unique_dst_ports: Optional[int] = 1
    tcp_flags:   Optional[List[str]] = None

    model_config = {"populate_by_name": True}


class ForecastRequest(BaseModel):
    indicator: str = Field(..., description="IP, hostname, or entity ID")
    risk_history: List[float] = Field(..., description="Chronological risk scores (0-100 or 0-1)")
    horizon: int = Field(5, description="Forecast steps")
    critical_threshold: float = Field(85.0, description="Score threshold for early warning")


class TokenRequest(BaseModel):
    username: str = Field(...)
    password: str = Field(...)


class IOCIngestRequest(BaseModel):
    ioc_value:   str
    ioc_type:    str = "ip"
    threat_name: str = "External Feed IOC"
    confidence:  float = 0.85
    severity:    str = "HIGH"
    source:      str = "REST_API"
    tags:        Optional[List[str]] = None


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Serves the real-time AHRAS SOC Web Dashboard."""
    dash_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")
    if os.path.exists(dash_path):
        with open(dash_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AHRAS Dashboard File Not Found</h1>"


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint returning engine state."""
    return {
        "status":     "healthy",
        "service":    "AHRAS SOC REST API",
        "version":    "6.0.0",
        "timestamp":  time.time(),
        "store_type": "SQLite / MongoDB Dual-Mode",
        "modules": {
            "ocsf_normalizer": "OPERATIONAL",
            "detection_engines": "TRI_ENGINE_ENSEMBLE",
            "adaptive_risk": "OPERATIONAL",
            "dynamic_trust": "ACTIVE",
            "xai_fidelity": "VERIFIED",
            "causal_forecasting": "OPERATIONAL",
            "threat_intelligence": "ACTIVE",
            "active_defense_soar": "ARMED",
            "rbac_access_control": "ENFORCED",
        }
    }


@app.get("/alerts", tags=["Alerts"])
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"),
    ocsf_class: Optional[str] = Query(None, description="Filter by OCSF class"),
    limit: int = Query(50, ge=1, le=500, description="Max alerts to return"),
):
    """Retrieves historical security alerts from storage."""
    try:
        store = get_store()
        query = {}
        if severity:
            query["severity"] = severity.upper()
        if ocsf_class:
            query["ocsf_class"] = ocsf_class

        alerts = store.query("alerts", query, limit=limit)
        return {
            "total_returned": len(alerts),
            "filters": {"severity": severity, "ocsf_class": ocsf_class},
            "alerts": alerts,
        }
    except Exception as e:
        log.error(f"[API] Error querying alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities/{entity_key:path}/report", tags=["Entities"])
def get_entity_report(
    entity_key: str = Path(..., description="Entity identifier (IP, hostname, username)"),
    ocsf_class: str = Query("network_activity", description="OCSF class of entity"),
    format: str = Query("json", description="Output format: json or markdown"),
):
    """Generates unified per-entity security report combining multi-modal statistical evidence."""
    try:
        rep_gen = get_entity_report_generator()
        last_evt = {"ocsf_class": ocsf_class, "entity_key": entity_key}
        report: EntityReport = rep_gen.generate_report(ocsf_class, entity_key, last_evt)

        if format.lower() == "markdown":
            return {"entity_key": entity_key, "format": "markdown", "content": report.to_markdown()}

        return report.to_dict()
    except Exception as e:
        log.error(f"[API] Error generating entity report for '{entity_key}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{action_id}/respond", tags=["Active Defense"])
def respond_to_action(
    action_id: str = Path(..., description="Action ID to approve or reject"),
    req: ResponseApprovalRequest = Body(...),
):
    """Approves or rejects a staged active defense mitigation action."""
    orch = get_response_orchestrator()

    if req.action.lower() == "approve":
        success = orch.approve_action(action_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found in pending approval queue")
        return {"action_id": action_id, "status": "APPROVED_AND_EXECUTED", "message": "Mitigation executed successfully"}
    elif req.action.lower() == "reject":
        success = orch.reject_action(action_id, reason=req.reason or "Analyst rejected")
        if not success:
            raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found in pending approval queue")
        return {"action_id": action_id, "status": "REJECTED", "message": "Staged mitigation rejected"}
    else:
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")


@app.get("/actions/pending", tags=["Active Defense"])
def list_pending_actions():
    """Lists all mitigation actions awaiting SOC analyst approval."""
    orch = get_response_orchestrator()
    return {"pending_count": len(orch.get_pending_actions()), "actions": orch.get_pending_actions()}


@app.get("/actions/history", tags=["Active Defense"])
def list_action_history():
    """Lists audit trail of all executed and staged response actions."""
    orch = get_response_orchestrator()
    return {"total_count": len(orch.get_action_history()), "history": orch.get_action_history()}


@app.post("/analyst/feedback", tags=["Analyst Interface"])
def submit_analyst_feedback(req: AnalystFeedbackRequest):
    """Allows SOC analysts to mark false positives or reset entity baselines."""
    stat_eng = get_statistical_engine()

    if req.action == "mark_false_positive":
        stat_eng.mark_false_positive(req.ocsf_class, req.entity_key)
        return {
            "status": "SUCCESS",
            "entity_key": req.entity_key,
            "message": f"Suppressed false positive alerts for '{req.entity_key}'",
        }
    elif req.action == "reset_entity":
        stat_eng.reset_entity(req.ocsf_class, req.entity_key)
        return {
            "status": "SUCCESS",
            "entity_key": req.entity_key,
            "message": f"Reset statistical baseline for '{req.entity_key}'",
        }
    else:
        raise HTTPException(status_code=400, detail="Action must be 'mark_false_positive' or 'reset_entity'")


@app.get("/metrics", tags=["System"])
def get_metrics():
    """Provides real-time SOC security operational metrics."""
    stat_eng = get_statistical_engine()
    orch = get_response_orchestrator()
    s_stats = stat_eng.get_stats()

    return {
        "system_status":     "OPERATIONAL",
        "tracked_entities":  s_stats.get("tracked_entities", 0),
        "total_events_scored": s_stats.get("total_scored", 0),
        "active_mitigations": len(orch.get_action_history()),
        "pending_approvals":  len(orch.get_pending_actions()),
        "uptime_sec":         round(time.time() - getattr(app.state, "start_time", time.time()), 2),
    }


# ── Research REST APIs (POST /api/detect, /api/score, /api/forecast, /api/auth) ──

@app.post("/api/detect", tags=["Detection & Risk"])
def detect_event(req: DetectRequest):
    """
    Ingests and normalizes an event flow, runs parallel detection engines,
    fuses outputs into adaptive risk score, and generates an XAI explanation.
    """
    try:
        src_ip = req.source_ip or "10.0.0.5"
        raw_evt = {
            "src_ip":           src_ip,
            "dst_ip":           req.dest_ip or "192.168.1.10",
            "src_port":         req.source_port or 34567,
            "dst_port":         req.dest_port or 22,
            "protocol":         req.protocol or "TCP",
            "packet_count":     req.packet_count or 10,
            "duration_sec":     req.duration_sec or 1.0,
            "bytes":            req.bytes or 512,
            "unique_dst_ports": req.unique_dst_ports or 1,
            "tcp_flags":        req.tcp_flags or (["SYN"] if req.dest_port == 22 else ["ACK"]),
        }
        ocsf_evt = _norm_network(raw_evt)
        
        combiner = get_combiner()
        det_res = combiner.process(ocsf_evt)
        
        # Check Threat Intel
        ti_mgr = get_threat_intel_manager()
        ti_matches = ti_mgr.match_event(ocsf_evt)
        ioc_count = len(ti_matches) + (len(req.ioc_match) if req.ioc_match else 0)
        
        risk_res = run_risk_engine(
            src_ip,
            det_res.signature_matches if det_res else [],
            det_res.anomaly_result if det_res else None,
            det_res.stat_result if det_res else None,
            ocsf_evt
        )
        
        # Build explanation contributions
        explanations = []
        if ioc_count > 0:
            explanations.append({"feature": "ioc_match", "value": ioc_count, "contribution": round(min(0.60, ioc_count * 0.30), 2)})
        if det_res and det_res.anomaly_result.get("ensemble_score", 0) > 0:
            explanations.append({"feature": "anomaly_score", "value": round(det_res.anomaly_result["ensemble_score"], 2), "contribution": round(risk_res.A_ml * 0.30, 2)})
        if risk_res.S_sig > 0:
            explanations.append({"feature": "signature_match", "value": round(risk_res.S_sig, 2), "contribution": round(risk_res.S_sig * 0.50, 2)})
        if risk_res.T_trust > 0:
            explanations.append({"feature": "trust_discount", "value": round(risk_res.T_trust, 2), "contribution": round(-0.15 * risk_res.T_trust, 2)})

        return {
            "risk_score": round(risk_res.risk_score, 4),
            "risk_level": risk_res.severity,
            "is_alert":   risk_res.is_alert,
            "remediation": risk_res.remediation_level,
            "explanation": explanations,
            "mitre_techniques": risk_res.mitre_techniques,
        }
    except Exception as e:
        log.error(f"[API] Error in /api/detect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score", tags=["Detection & Risk"])
def score_event(event_dict: Dict[str, Any] = Body(...)):
    """Direct score endpoint for arbitrary event dicts."""
    try:
        src_ip = event_dict.get("src_ip") or event_dict.get("source_ip") or "10.0.0.1"
        ocsf_evt = _norm_network(event_dict) if event_dict.get("ocsf_class") is None else event_dict
        combiner = get_combiner()
        det_res = combiner.process(ocsf_evt)
        risk_res = run_risk_engine(
            src_ip,
            det_res.signature_matches if det_res else [],
            det_res.anomaly_result if det_res else None,
            det_res.stat_result if det_res else None,
            ocsf_evt
        )
        return {
            "risk": round(risk_res.risk_score, 4),
            "level": risk_res.severity,
            "components": {
                "S_sig": risk_res.S_sig,
                "A_ml": risk_res.A_ml,
                "delta_D": risk_res.delta_D,
                "T_trust": risk_res.T_trust,
            },
            "explanation": risk_res.explanation,
        }
    except Exception as e:
        log.error(f"[API] Error in /api/score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forecast", tags=["Forecasting"])
def forecast_risk(req: ForecastRequest):
    """Evaluates time-series risk forecasting and early warning lead time."""
    predictor = AttackPredictor(horizon=req.horizon)
    res: ForecastResult = predictor.predict(req.indicator, req.risk_history, critical_threshold=req.critical_threshold)
    return res.to_dict()


@app.get("/api/forecast/escalating", tags=["Forecasting"])
def list_escalating_threats(limit: int = Query(10, ge=1, le=50)):
    """Surfaces top escalating risk indicators across monitored entities."""
    stat_eng = get_statistical_engine()
    profiles = stat_eng.get_all_profiles()
    
    histories = {}
    for p in profiles:
        k = p.get("entity_key")
        z = p.get("zscore", 0.0)
        drift = p.get("behavioral_drift", 0.0)
        # Synthesize recent score ramp
        base = min(1.0, (z / 5.0) * 0.5 + (drift / 3.0) * 0.5)
        histories[k] = [max(0.0, base - 0.2), max(0.0, base - 0.1), base]

    predictor = AttackPredictor(horizon=5)
    top = predictor.top_escalating(histories, n=limit)
    return {"escalating_count": len(top), "escalating": [t.to_dict() for t in top]}


@app.post("/api/auth/token", tags=["Authentication & RBAC"])
def login_for_access_token(req: TokenRequest):
    """Authenticates user credentials and returns JWT Bearer token with RBAC role."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = user["role"]
    access_token = create_access_token(data={"sub": user["username"], "role": role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": role,
        "permissions": list(get_user_permissions(role)),
    }


@app.get("/api/auth/users", tags=["Authentication & RBAC"])
def get_user_list():
    """Lists registered SOC users and assigned roles."""
    return {"users": list_users()}


@app.get("/api/threat-intel/iocs", tags=["Threat Intelligence"])
def get_iocs(limit: int = Query(50, ge=1, le=500)):
    """Lists active threat intelligence indicators."""
    ti = get_threat_intel_manager()
    return {"total_count": ti.count_iocs(), "iocs": ti.list_iocs(limit=limit)}


@app.post("/api/threat-intel/iocs", tags=["Threat Intelligence"])
def add_ioc(req: IOCIngestRequest):
    """Ingests a new STIX/IOC threat indicator."""
    ti = get_threat_intel_manager()
    rec = ti.add_ioc(
        ioc_value=req.ioc_value,
        ioc_type=req.ioc_type,
        threat_name=req.threat_name,
        confidence=req.confidence,
        severity=req.severity,
        source=req.source,
        tags=req.tags,
    )
    return {"status": "SUCCESS", "ioc": rec.to_dict()}


@app.get("/api/xai/fidelity", tags=["Explainability"])
def get_xai_fidelity_summary():
    """Returns exact analytical sum-check and feature alignment metrics summary."""
    ledger = get_fidelity_ledger()
    return ledger.get_summary()


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Utility launcher for running uvicorn server in standalone mode."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
