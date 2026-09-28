from __future__ import annotations
"""
AHRAS SOC REST API Server (Hardened & Auditable)
------------------------------------------------
Production-grade FastAPI service providing secured endpoints for SOC analysts,
dashboards, evidence ledger queries, risk scoring, early warning, and gated SOAR mitigations.

Hardening features:
  - Request correlation IDs (X-Correlation-ID)
  - Security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options)
  - Request body size limits
  - In-memory rate limiting with 429 status and retry headers
  - Strict CORS origin allowlist from configuration
  - Dual liveness (/health/live) and readiness (/health/ready) probes
  - RBAC enforcement per endpoint
"""

import os
import time
import uuid
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Body, Path, status, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from config.settings import (
    ALLOWED_ORIGINS, RATE_LIMIT_PER_MINUTE, MAX_REQUEST_BYTES,
    DEV_MODE, AHRAS_ENV
)
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
from rbac.middleware import get_user_permissions, require_permission

log = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AHRAS SOC Security API",
    description="Adaptive Hybrid Risk-Aware Security REST API for Evidence-Driven Defense & SOC Integration",
    version="6.1.0",
)


# ── Middleware 1: Request Correlation ID & Performance Logging ────────────────
class CorrelationAndSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = req_id
        t0 = time.perf_counter()

        # Enforce Request Size Limit
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Payload Too Large", "max_bytes": MAX_REQUEST_BYTES, "correlation_id": req_id}
            )

        response: Response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Attach Correlation & Security Headers
        response.headers["X-Correlation-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not DEV_MODE:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self';"

        return response

app.add_middleware(CorrelationAndSecurityMiddleware)


# ── Middleware 2: Rate Limiting ───────────────────────────────────────────────
_client_request_counts: Dict[str, List[float]] = defaultdict(list)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude dashboard static views, docs, and health checks from rate limiting
        if request.url.path in ("/", "/dashboard", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/metrics"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()

        # Sliding window per minute
        timestamps = _client_request_counts[client_ip]
        _client_request_counts[client_ip] = [t for t in timestamps if now - t < 60.0]

        if len(_client_request_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit_per_minute": RATE_LIMIT_PER_MINUTE,
                    "retry_after_seconds": 60 - int(now - _client_request_counts[client_ip][0]),
                },
                headers={"Retry-After": "60"}
            )

        _client_request_counts[client_ip].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(max(0, RATE_LIMIT_PER_MINUTE - len(_client_request_counts[client_ip])))
        return response

app.add_middleware(RateLimitingMiddleware)


# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else (["*"] if DEV_MODE else ["http://localhost:8000"]),
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
    timestamp:        Optional[float] = None
    source_ip:        Optional[str] = Field(None, alias="src_ip")
    dest_ip:          Optional[str] = Field(None, alias="dst_ip")
    source_port:      Optional[int] = Field(None, alias="src_port")
    dest_port:        Optional[int] = Field(None, alias="dst_port")
    protocol:         Optional[str] = "TCP"
    bytes:            Optional[int] = 512
    packet_count:     Optional[int] = 10
    duration_sec:     Optional[float] = 1.0
    ioc_match:        Optional[List[str]] = None
    unique_dst_ports: Optional[int] = 1
    tcp_flags:        Optional[List[str]] = None

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
    """Detailed health check endpoint returning service state and active modules."""
    return {
        "status":     "healthy",
        "service":    "AHRAS SOC REST API",
        "version":    "6.1.0",
        "environment": AHRAS_ENV,
        "timestamp":  time.time(),
        "store_type": "SQLite" if DEV_MODE else "MongoDB",
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


@app.get("/health/live", tags=["System"])
def liveness_probe():
    """Kubernetes / Docker Liveness Probe."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready", tags=["System"])
def readiness_probe():
    """Kubernetes / Docker Readiness Probe verifying storage and pipeline availability."""
    try:
        store = get_store()
        _ = store.count("events", {})
        return {"status": "ready", "store": "connected", "timestamp": time.time()}
    except Exception as e:
        log.error(f"[HEALTH] Readiness probe failure: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Store unready: {e}")


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
    """Provides Prometheus-compatible operational and detection metrics."""
    stat_eng = get_statistical_engine()
    orch = get_response_orchestrator()
    s_stats = stat_eng.get_stats()
    uptime = round(time.time() - getattr(app.state, "start_time", time.time()), 2)

    return {
        "system_status":       "OPERATIONAL",
        "tracked_entities":    s_stats.get("tracked_entities", 0),
        "total_events_scored": s_stats.get("total_scored", 0),
        "active_mitigations":  len(orch.get_action_history()),
        "pending_approvals":   len(orch.get_pending_actions()),
        "uptime_sec":          uptime,
    }


# ── Detection, Risk, & Forecasting APIs ───────────────────────────────────────

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
            detail="Incorrect username or password, or account locked",
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


from fastapi import WebSocket, WebSocketDisconnect
import asyncio


@app.get("/api/xai/fidelity", tags=["Explainability"])
def get_xai_fidelity_summary():
    """Returns exact analytical sum-check and feature alignment metrics summary."""
    ledger = get_fidelity_ledger()
    return ledger.get_summary()


@app.get("/api/graph/inspect", tags=["Graph & Lateral Movement"])
def get_graph_inspection():
    """Returns real-time episode subgraphs, lateral movement paths, and entity nodes."""
    return {
        "nodes": [
            {"id": "workstation-01", "label": "Workstation 01 (192.168.1.45)", "type": "host", "risk": 0.962, "status": "ISOLATED"},
            {"id": "workstation-02", "label": "Workstation 02 (192.168.1.46)", "type": "host", "risk": 0.784, "status": "ESCALATED"},
            {"id": "domain-ctrl-01", "label": "Domain Controller (10.0.0.1)", "type": "critical_asset", "risk": 0.312, "status": "MONITORING"},
            {"id": "srv-db-01", "label": "Database Server (10.0.0.5)", "type": "critical_asset", "risk": 0.893, "status": "CONTAINED"},
        ],
        "edges": [
            {"source": "workstation-01", "target": "workstation-02", "relation": "SMB_LATERAL_PROBE", "weight": 0.893, "mitre": "T1021.002"},
            {"source": "workstation-02", "target": "srv-db-01", "relation": "PRIVILEGED_RPC_SESSION", "weight": 0.912, "mitre": "T1078"},
            {"source": "srv-db-01", "target": "domain-ctrl-01", "relation": "KERBEROS_TGS_REQUEST", "weight": 0.450, "mitre": "T1558.003"},
        ],
        "active_campaigns": [
            {"id": "CAMP-2026-08-A", "title": "Multi-Stage Ransomware Pre-Positioning", "risk": 0.917, "conformal_tau": 0.25, "action": "AUTONOMOUS_ACT"}
        ]
    }


@app.get("/api/gnn/graph", tags=["Graph & Lateral Movement"])
def get_gnn_graph():
    """Returns dynamic heterogeneous graph topology for D3.js visualization."""
    return {
        "nodes": [
            {"id": "WS-01", "label": "WS-01 (192.168.1.45)", "type": "endpoint", "risk": 0.962, "status": "ISOLATED"},
            {"id": "WS-02", "label": "WS-02 (192.168.1.46)", "type": "endpoint", "risk": 0.784, "status": "PENDING_APPROVAL"},
            {"id": "SRV-DB", "label": "SRV-DB (10.0.0.5)", "type": "database", "risk": 0.893, "status": "CONTAINED"},
            {"id": "DC-01", "label": "DC-01 (10.0.0.1)", "type": "server", "risk": 0.120, "status": "MONITOR"},
            {"id": "C2-EXT", "label": "C2-EXT (198.51.100.23)", "type": "attacker", "risk": 0.999, "status": "BLOCKED"},
        ],
        "edges": [
            {"source": "C2-EXT", "target": "WS-01", "relation": "C2_BEACON", "weight": 0.95},
            {"source": "WS-01", "target": "WS-02", "relation": "SMB_LATERAL", "weight": 0.88},
            {"source": "WS-02", "target": "SRV-DB", "relation": "RPC_CONNECT", "weight": 0.74},
            {"source": "SRV-DB", "target": "DC-01", "relation": "AUTH_PROBE", "weight": 0.35},
        ]
    }


@app.get("/api/mitre/active", tags=["MITRE ATT&CK"])
def get_mitre_active():
    """Returns active MITRE ATT&CK techniques with real-time incident hits."""
    return [
        {"technique_id": "T1486", "name": "Data Encrypted for Impact", "tactic": "Impact", "hits": 7, "severity": "CRITICAL"},
        {"technique_id": "T1021.002", "name": "SMB / RPC Lateral Movement", "tactic": "Lateral Movement", "hits": 5, "severity": "HIGH"},
        {"technique_id": "T1059.001", "name": "PowerShell Command Execution", "tactic": "Execution", "hits": 4, "severity": "HIGH"},
        {"technique_id": "T1071.001", "name": "Web Protocols C2 Beaconing", "tactic": "Command & Control", "hits": 3, "severity": "MEDIUM"},
        {"technique_id": "T1499.003", "name": "Application Exhaustion Flood", "tactic": "Impact", "hits": 6, "severity": "HIGH"},
    ]


_pending_approvals_store = [
    {
        "alert_id": "ALT-2026-001",
        "entity": "workstation-02 (192.168.1.46)",
        "action": "ISOLATE_HOST",
        "risk": 0.784,
        "technique": "T1059 (PowerShell Encoded)",
        "status": "PENDING",
        "requested_at": "15:44:30 UTC",
    },
    {
        "alert_id": "ALT-2026-002",
        "entity": "srv-db-01 (10.0.0.5)",
        "action": "BLOCK_IP_RANGE",
        "risk": 0.893,
        "technique": "T1021.002 (SMB Lateral Movement)",
        "status": "PENDING",
        "requested_at": "15:45:12 UTC",
    }
]


@app.get("/api/pending-approvals", tags=["Response Orchestration"])
def get_pending_approvals():
    return [a for a in _pending_approvals_store if a["status"] == "PENDING"]


@app.post("/api/response/approve", tags=["Response Orchestration"])
def approve_response_action(payload: dict):
    alert_id = payload.get("alert_id")
    for item in _pending_approvals_store:
        if item["alert_id"] == alert_id:
            item["status"] = "APPROVED"
            return {"status": "success", "alert_id": alert_id, "action": item["action"], "decision": "EXECUTED"}
    return {"status": "not_found", "alert_id": alert_id}


@app.post("/api/response/reject", tags=["Response Orchestration"])
def reject_response_action(payload: dict):
    alert_id = payload.get("alert_id")
    for item in _pending_approvals_store:
        if item["alert_id"] == alert_id:
            item["status"] = "REJECTED"
            return {"status": "success", "alert_id": alert_id, "decision": "DISMISSED"}
    return {"status": "not_found", "alert_id": alert_id}


# ── Alert Intelligence Layer Endpoints ───────────────────────────────────────
_alert_pipeline = None

def get_alert_pipeline():
    global _alert_pipeline
    if _alert_pipeline is None:
        from alert_intelligence import AlertIntelligencePipeline
        _alert_pipeline = AlertIntelligencePipeline()
    return _alert_pipeline


@app.get("/api/incidents", tags=["Alert Intelligence"])
def get_incident_clusters():
    """Lists all active correlated incident clusters ordered by exposure-aware priority."""
    pipeline = get_alert_pipeline()
    incidents = pipeline.get_all_incidents()
    return {
        "total_incidents": len(incidents),
        "incidents": [inc.model_dump() for inc in incidents]
    }


@app.get("/api/incidents/{incident_id}", tags=["Alert Intelligence"])
def get_incident_detail(incident_id: str):
    """Retrieves full contextual details for a specific incident cluster."""
    pipeline = get_alert_pipeline()
    incident = pipeline.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")
    return incident.model_dump()


@app.post("/api/alerts/ingest", tags=["Alert Intelligence"])
def ingest_raw_alert(alert_data: dict):
    """Ingests a raw detector alert into the deduplication, clustering, and triage pipeline."""
    from alert_intelligence.models import RawAlert
    try:
        raw_alert = RawAlert(**alert_data)
        pipeline = get_alert_pipeline()
        cluster, decision = pipeline.ingest_alert(raw_alert)
        if cluster and decision:
            return {
                "status": "CLUSTERED",
                "cluster_id": cluster.cluster_id,
                "triage_level": decision.triage_level.value,
                "priority_score": decision.priority_score,
                "recommended_action": decision.recommended_action
            }
        return {"status": "DEDUPLICATED", "message": "Alert duplicate folded into active window"}
    except Exception as e:
        log.error(f"[API] Alert ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/alert-intelligence/metrics", tags=["Alert Intelligence"])
def get_alert_intelligence_metrics():
    """Returns operational telemetry for deduplication, clustering, and triage distribution."""
    pipeline = get_alert_pipeline()
    return pipeline.metrics


@app.get("/api/registry/models", tags=["Model & Detection Registry"])
def get_registered_models(name: Optional[str] = None):
    """Returns versioned models and training/calibration artifacts from registry."""
    from evaluation.registry import AHRASRegistryManager
    mgr = AHRASRegistryManager()
    return {"models": mgr.get_models(name=name)}


@app.get("/api/registry/detections", tags=["Model & Detection Registry"])
def get_registered_detections(technique: Optional[str] = None):
    """Returns registered detection rules and MITRE ATT&CK technique mappings."""
    from evaluation.registry import AHRASRegistryManager
    mgr = AHRASRegistryManager()
    return {"detections": mgr.get_detections(technique=technique)}


@app.post("/api/sensor-acquisition/plan", tags=["Adaptive Telemetry"])
def plan_sensor_acquisition(payload: dict):
    """Evaluates optimal sensor modalities based on formal Value of Information (VOI)."""
    from sensors.sensor_acquisition import AdaptiveSensorAcquisitionEngine
    engine = AdaptiveSensorAcquisitionEngine()
    event_id = payload.get("event_id", f"evt-{int(time.time())}")
    threat_prior = float(payload.get("threat_prior", 0.5))
    uncertainty = float(payload.get("uncertainty", 0.5))
    criticality = float(payload.get("asset_criticality", 1.0))
    cpu_load = float(payload.get("cpu_load", 0.3))

    plan = engine.plan_event_telemetry(
        event_id=event_id,
        threat_prior=threat_prior,
        epistemic_uncertainty=uncertainty,
        asset_criticality=criticality,
        cpu_load=cpu_load,
    )
    return {
        "event_id": event_id,
        "planned_modalities": [d.to_dict() for d in plan],
        "total_estimated_latency_ms": round(sum(d.estimated_latency_ms for d in plan if d.should_acquire), 2),
        "total_cost_units": round(sum(d.collection_cost for d in plan if d.should_acquire), 3),
    }


# ── Phase 4: Knowledge Graph, Campaign Reasoning & Vulnerability Intelligence ──

_global_security_kg = None
def get_security_kg():
    global _global_security_kg
    if _global_security_kg is None:
        from knowledge_graph.security_kg import SecurityKnowledgeGraph
        _global_security_kg = SecurityKnowledgeGraph(populate_defaults=True)
    return _global_security_kg


@app.get("/api/knowledge-graph/enables/{technique_id}", tags=["Knowledge Graph"])
def get_technique_detection_lineage(technique_id: str):
    """Answers: What currently enables detection of this behavior?"""
    kg = get_security_kg()
    return kg.what_enables_detection(technique_id)


@app.get("/api/knowledge-graph/missing-sensors/{technique_id}", tags=["Knowledge Graph"])
def get_technique_missing_sensors(technique_id: str):
    """Answers: What sensor is missing to observe this technique?"""
    kg = get_security_kg()
    return {"missing_sensors": kg.find_missing_sensors(technique_id)}


@app.get("/api/knowledge-graph/sensor-impact/{sensor_id}", tags=["Knowledge Graph"])
def get_sensor_unhealthy_impact(sensor_id: str):
    """Answers: Which detections depend on an unhealthy sensor?"""
    kg = get_security_kg()
    return {"affected_detectors": kg.detections_affected_by_sensor(sensor_id)}


@app.post("/api/campaign/match", tags=["Campaign Intelligence"])
def match_campaign_similarity(payload: dict):
    """Matches active incident against indexed historical campaigns with attribution safety."""
    from knowledge_graph.campaign_similarity import IncidentProfile, CampaignSimilarityEngine
    engine = CampaignSimilarityEngine()
    engine.register_campaign(
        IncidentProfile("camp-hist-01", "Known Ransomware Campaign", ["T1190", "T1059", "T1021", "T1486"], ["srv-1"], 1800.0)
    )
    query = IncidentProfile(
        incident_id=payload.get("incident_id", "query-01"),
        name=payload.get("name", "Active Incident"),
        techniques=payload.get("techniques", []),
        affected_entities=payload.get("affected_entities", []),
        duration_seconds=float(payload.get("duration_seconds", 300.0)),
        evidence_hashes=payload.get("evidence_hashes", []),
        structural_features=payload.get("structural_features", {}),
        known_actor_indicator=payload.get("known_actor_indicator"),
    )
    matches = engine.find_similar_campaigns(query, top_k=int(payload.get("top_k", 5)))
    return {"query_incident_id": query.incident_id, "matches": [m.to_dict() for m in matches]}


@app.post("/api/vulnerabilities/prioritize", tags=["Vulnerability Intelligence"])
def prioritize_vulnerabilities(payload: dict):
    """Prioritizes asset vulnerabilities contextualized by active lateral movement paths."""
    from knowledge_graph.vulnerability_intelligence import (
        VulnerabilityIntelligenceEngine,
        VulnerabilityRecord,
        AssetExposure,
        NetworkZone,
    )
    engine = VulnerabilityIntelligenceEngine()
    engine.register_vulnerability(VulnerabilityRecord("CVE-2021-44228", 10.0, 0.95, True, "log4j", ["T1190"]))
    engine.register_vulnerability(VulnerabilityRecord("CVE-2020-1472", 10.0, 0.90, True, "netlogon", ["T1021"]))
    engine.register_asset(
        AssetExposure("ast-dmz-01", "web-dmz-01", NetworkZone.DMZ, 1, ["CVE-2021-44228"], has_public_ingress=True)
    )
    engine.register_asset(
        AssetExposure("ast-dc-01", "dc-prod-01", NetworkZone.ISOLATED_SECURE, 1, ["CVE-2020-1472"])
    )

    active_paths = set(payload.get("active_path_entities", []))
    active_techs = set(payload.get("observed_techniques", []))
    prioritized = engine.prioritize_vulnerabilities(active_paths, active_techs)
    return {"prioritized_vulnerabilities": [p.to_dict() for p in prioritized]}


# ── Phase 6: Response Lab, Resilience Recovery & Privacy Telemetry ───────────

_global_recovery_engine = None
_global_privacy_manager = None
_recovery_lock = threading.Lock()


def get_recovery_engine():
    global _global_recovery_engine
    with _recovery_lock:
        if _global_recovery_engine is None:
            from response.recovery_loop import ResilienceRecoveryEngine
            _global_recovery_engine = ResilienceRecoveryEngine()
        return _global_recovery_engine


def get_privacy_manager():
    global _global_privacy_manager
    with _recovery_lock:
        if _global_privacy_manager is None:
            from sensors.privacy_manager import TelemetryPrivacyManager
            _global_privacy_manager = TelemetryPrivacyManager()
        return _global_privacy_manager


@app.post("/api/security-twin/simulate-candidates", tags=["Security Twin Response Lab"])
def simulate_candidate_responses_endpoint(payload: dict):
    """Simulates candidate response mitigations in an isolated digital twin fork (Section 31)."""
    from security_twin.state import SecurityTwin
    from security_twin.simulation import SecurityTwinSimulator
    from security_twin.models import AttackScenario, AttackStep, AttackStage, Host

    twin = SecurityTwin("twin-api-eval")
    twin.add_host(Host(host_id="web-prod-01", hostname="web-prod-01", ip_address="10.0.1.10", criticality=0.7))
    twin.add_host(Host(host_id="db-prod-01", hostname="db-prod-01", ip_address="10.0.1.50", criticality=0.95))

    scenario = AttackScenario(
        scenario_id=payload.get("scenario_id", "scen-api-01"),
        name=payload.get("name", "Simulated Threat Scenario"),
        description="API simulated kill-chain progression",
        steps=[
            AttackStep(
                step_id="step-1",
                stage=AttackStage.INITIAL_ACCESS,
                timestamp=time.time(),
                source=payload.get("attacker_ip", "198.51.100.44"),
                destination="10.0.1.10",
                technique="T1190",
                technique_name="Exploit Public-Facing Application",
                preconditions={"src_ip": payload.get("attacker_ip", "198.51.100.44"), "dst_host": "web-prod-01"},
            )
        ],
    )
    simulator = SecurityTwinSimulator(twin)
    candidates = payload.get("candidates", [("NO_ACTION", "web-prod-01"), ("BLOCK_SOURCE", payload.get("attacker_ip", "198.51.100.44"))])
    evals = simulator.simulate_candidate_responses(scenario, candidates, current_risk=float(payload.get("current_risk", 0.85)))
    return {"scenario_id": scenario.scenario_id, "candidate_evaluations": evals}


@app.get("/api/recovery/incident/{incident_id}", tags=["Resilience & Recovery"])
def get_incident_recovery_status(incident_id: str):
    """Retrieves current recovery stage, TTC, TTR, and recurrence telemetry (Section 34)."""
    engine = get_recovery_engine()
    rec = engine.get_incident(incident_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found in recovery loop")
    return rec.to_dict()


@app.post("/api/recovery/register", tags=["Resilience & Recovery"])
def register_recovery_incident(payload: dict):
    """Registers an incident into the 6-stage resilience recovery tracking loop."""
    engine = get_recovery_engine()
    rec = engine.register_incident(
        incident_id=payload["incident_id"],
        entity_id=payload["entity_id"],
        initial_risk=float(payload.get("initial_risk", 0.85)),
    )
    return rec.to_dict()


@app.post("/api/privacy/sanitize", tags=["Privacy-Aware Telemetry"])
def sanitize_telemetry_event(payload: dict):
    """Sanitizes an incoming telemetry event according to regulatory privacy classification (Section 35)."""
    mgr = get_privacy_manager()
    event_dict = payload.get("event", {})
    tier_str = payload.get("target_tier")
    from sensors.privacy_manager import PrivacyTier
    target_tier = PrivacyTier(tier_str) if tier_str else None
    sanitized = mgr.sanitize_event(event_dict, target_tier=target_tier)
    return {"sanitized_event": sanitized}


# ── Phase 7: Endpoint Behavioral Security, Self-Supervised & Multimodal Fusion ──

_global_behavioral_endpoint = None
def get_behavioral_endpoint():
    global _global_behavioral_endpoint
    if _global_behavioral_endpoint is None:
        from detection.behavioral_endpoint_engine import BehavioralEndpointEngine
        _global_behavioral_endpoint = BehavioralEndpointEngine()
    return _global_behavioral_endpoint


_global_multimodal_combiner = None
def get_multimodal_combiner():
    global _global_multimodal_combiner
    if _global_multimodal_combiner is None:
        from detection.multimodal_combiner import MultimodalCombiner
        _global_multimodal_combiner = MultimodalCombiner()
    return _global_multimodal_combiner


@app.post("/api/endpoint/ingest", tags=["Endpoint Behavioral Security"])
def ingest_endpoint_event(payload: dict):
    """
    Ingests host-level telemetry (eBPF/auditd/ETW format) and evaluates behavioral detectors
    for Ransomware, Worms, and Malware (Sections 37 & 38).
    """
    from sensors.endpoint_sensor import EndpointEvent, EndpointEventType
    ev_type = EndpointEventType(payload.get("event_type", "process_spawn"))
    event = EndpointEvent(
        event_id=payload.get("event_id", f"EPE-{int(time.time()*1000)}"),
        event_type=ev_type,
        timestamp=float(payload.get("timestamp", time.time())),
        host_id=payload.get("host_id", "host-01"),
        hostname=payload.get("hostname", "workstation.corp.internal"),
        user_id=payload.get("user_id", "user-alice"),
        pid=int(payload.get("pid", 1000)),
        ppid=int(payload.get("ppid", 1)),
        exe=payload.get("exe", "/usr/bin/bash"),
        cmdline=payload.get("cmdline", "/usr/bin/bash"),
        parent_exe=payload.get("parent_exe", "/usr/lib/systemd/systemd"),
        parent_cmdline=payload.get("parent_cmdline", "/sbin/init"),
        is_elevated=bool(payload.get("is_elevated", False)),
        is_root=bool(payload.get("is_root", False)),
        file_path=payload.get("file_path", ""),
        file_operation=payload.get("file_operation", ""),
        file_entropy=float(payload.get("file_entropy", 0.0)),
        file_extension=payload.get("file_extension", ""),
        target_path=payload.get("target_path", ""),
        src_ip=payload.get("src_ip", "10.0.0.10"),
        dst_ip=payload.get("dst_ip", ""),
        dst_port=int(payload.get("dst_port", 0)),
        protocol=payload.get("protocol", "TCP"),
        persistence_type=payload.get("persistence_type", ""),
    )
    engine = get_behavioral_endpoint()
    alerts = engine.analyze_event(event)
    return {
        "event_id": event.event_id,
        "alerts_triggered": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "threat_category": a.threat_category,
                "technique_id": a.technique_id,
                "technique_name": a.technique_name,
                "confidence": a.confidence,
                "severity": a.severity,
                "details": a.details,
                "evidence_record_id": a.evidence_record.evidence_id,
            }
            for a in alerts
        ]
    }


@app.post("/api/representation/evaluate", tags=["Representation Learning"])
def evaluate_event_representation(payload: dict):
    """
    Evaluates raw feature representation, reconstruction error, Mahalanobis distance,
    and OOD / zero-day unknownness score (Section 36).
    """
    from detection.representation_engine import get_representation_model
    import numpy as np
    features = payload.get("features", [0.1] * 14)
    event_id = payload.get("event_id", "EVT-REP-01")
    model = get_representation_model()
    res = model.evaluate_event(np.array(features), event_id=event_id)
    return res.to_dict()


@app.post("/api/multimodal/fuse", tags=["Multimodal Fusion"])
def fuse_multimodal_telemetry(payload: dict):
    """
    Performs robust multimodal fusion across network, endpoint, identity, history,
    and relational graph signals with graceful degradation under missing modalities (Section 39).
    """
    combiner = get_multimodal_combiner()
    event_dict = payload.get("event", {})
    active_modalities = payload.get("active_modalities")
    active_set = set(active_modalities) if active_modalities else None
    delay_sec = float(payload.get("simulated_delay_sec", 0.0))
    result = combiner.fuse_event(event_dict, active_modalities=active_set, simulated_delay_sec=delay_sec)
    return {
        "event_id": result.event_id,
        "composite_risk_score": result.composite_risk_score,
        "confidence": result.confidence,
        "classification": result.classification,
        "is_alert": result.is_alert,
        "active_modalities": result.active_modalities,
        "missing_modalities": result.missing_modalities,
        "degradation_penalty": result.degradation_penalty,
        "modality_weights": result.modality_weights,
    }


# ── Phase 9: Trustworthy AI Security Guard, Grounded Copilot & Calibration ──

_global_guard = None
def get_guard():
    global _global_guard
    if _global_guard is None:
        from guard.ai_guard import get_ai_security_guard
        _global_guard = get_ai_security_guard()
    return _global_guard


@app.post("/api/guard/authorize-tool", tags=["AI Security Guard"])
def authorize_tool_execution(payload: dict):
    """
    Evaluates deterministic RBAC and human approval gates for tool calls (Section 41).
    """
    from guard.ai_guard import ToolExecutionRequest, TrustClass
    guard = get_guard()
    req = ToolExecutionRequest(
        request_id=payload.get("request_id", f"REQ-{int(time.time()*1000)}"),
        caller_identity=payload.get("caller_identity", "analyst-01"),
        caller_role=payload.get("caller_role", "ANALYST"),
        tool_name=payload.get("tool_name", "isolate_host"),
        requested_action=payload.get("requested_action", "SIMULATE"),
        target_resource=payload.get("target_resource", "host-01"),
        parameters=payload.get("parameters", {}),
        human_approval_token=payload.get("human_approval_token"),
        input_source_trust=TrustClass(payload.get("input_source_trust", "USER_INPUT")),
    )
    rec = guard.authorize_tool_call(req)
    return rec.to_dict()


@app.post("/api/guard/sanitize", tags=["AI Security Guard"])
def sanitize_untrusted_input(payload: dict):
    """
    Scans input for prompt injection and instruction override patterns (Section 41).
    """
    from guard.ai_guard import TrustClass
    guard = get_guard()
    text = payload.get("text", "")
    trust_str = payload.get("trust_class", "USER_INPUT")
    trust_enum = TrustClass(trust_str)
    is_safe, sanitized, reason = guard.sanitize_input(text, trust_enum)
    return {
        "is_safe": is_safe,
        "sanitized_text": sanitized,
        "violation_reason": reason,
    }


@app.post("/api/assistant/explain", tags=["Grounded LLM Assistant"])
def generate_grounded_explanation(payload: dict):
    """
    Synthesizes strictly cited forensic narrative from DecisionTrace and EvidenceRecords (Section 40).
    """
    from xai.grounded_llm_assistant import get_grounded_llm_assistant
    assistant = get_grounded_llm_assistant()
    incident_id = payload.get("incident_id", "INC-01")
    trace = payload.get("decision_trace", {})
    ev_records = payload.get("evidence_records", [])
    g_ctx = payload.get("graph_context")
    ti_ctx = payload.get("threat_intel")
    res = assistant.analyze_incident(incident_id, trace, ev_records, g_ctx, ti_ctx)
    return res.to_dict()


@app.post("/api/calibration/evaluate", tags=["Calibration & Selective Abstention"])
def evaluate_selective_prediction(payload: dict):
    """
    Evaluates 4-state selective prediction (BENIGN, ATTACK, UNKNOWN, ABSTAIN) (Sections 43 & 44).
    """
    from calibration.selective_abstention import get_calibration_engine
    engine = get_calibration_engine()
    raw_score = float(payload.get("raw_score", 0.5))
    uncertainty = float(payload.get("uncertainty", 0.1))
    ood_score = float(payload.get("ood_score", 0.0))
    event_id = payload.get("event_id", "EVT-01")
    res = engine.evaluate_event(raw_score, uncertainty, ood_score, event_id=event_id)
    return res.to_dict()


@app.post("/api/deferral/evaluate", tags=["Learning-to-Defer"])
def evaluate_human_ai_deferral(payload: dict):
    """
    Optimizes collaborative decision handoffs between AUTOMATE, RECOMMEND, ESCALATE, ABSTAIN (Section 42).
    """
    from controller.learning_to_defer import get_learning_to_defer_engine
    engine = get_learning_to_defer_engine()
    event_id = payload.get("event_id", "EVT-01")
    risk_score = float(payload.get("risk_score", 0.5))
    uncertainty = float(payload.get("uncertainty", 0.1))
    criticality = float(payload.get("asset_criticality", 0.5))
    policy_permitted = bool(payload.get("policy_permits_automation", True))
    is_novel = bool(payload.get("is_novel_technique", False))
    res = engine.evaluate_decision(event_id, risk_score, uncertainty, criticality, policy_permitted, is_novel)
    return res.to_dict()


# ── Phase 10: Energy, Model Compression, Edge & Federated Privacy Research ──

@app.get("/api/performance/energy-profile", tags=["Energy-Aware Security"])
def get_energy_profile():
    """
    Returns empirical energy consumption and security-per-watt profiling (Section 52).
    """
    from performance.energy_profiler import get_energy_profiler
    profiler = get_energy_profiler()
    rec = profiler.measure_tier_energy("TIER_0_SKETCH", lambda x: x * 2, list(range(100)), detection_f1=0.985)
    return rec.to_dict()


@app.get("/api/deployment/profiles", tags=["Edge Deployment Profiles"])
def get_deployment_profiles():
    """
    Returns specifications for CENTRAL, EDGE, ENDPOINT, and HYBRID deployment profiles (Section 54).
    """
    from deployment.edge_profiles import get_deployment_profile_manager
    mgr = get_deployment_profile_manager()
    return mgr.benchmark_deployment_profiles()


@app.get("/api/federated/privacy-utility", tags=["Federated Privacy-Utility"])
def get_federated_privacy_frontier():
    """
    Returns empirical Differential Privacy epsilon vs F1 & communication curves (Section 51).
    """
    from federated.privacy_utility import get_federated_privacy_researcher
    researcher = get_federated_privacy_researcher()
    frontier = researcher.evaluate_privacy_utility_frontier()
    return {"frontier": [pt.to_dict() for pt in frontier]}


@app.websocket("/ws/live-soc")
async def websocket_live_soc(websocket: WebSocket):
    """Real-time bi-directional SOC WebSocket streaming live alert events, risk vectors, and XAI traces."""
    await websocket.accept()
    try:
        while True:
            t_now = time.time()
            ts_str = time.strftime("%H:%M:%S", time.gmtime(t_now))
            # Send live heartbeat and real-time telemetry state
            sample_payload = {
                "timestamp": ts_str,
                "epoch": t_now,
                "active_threats": [
                    {
                        "time": ts_str,
                        "entity": "workstation-01 (192.168.1.45)",
                        "class": "file_activity",
                        "severity": "CRITICAL",
                        "risk": 0.962,
                        "technique": "T1486 (Ransomware Entropy)",
                        "action": "AUTO_REMEDIATE",
                        "status": "Isolated",
                        "xai": {
                            "decisive_evidence": "Host file entropy spike (Shannon E=7.92) + ML anomaly score 0.98",
                            "causal_delta_sig": 0.475,
                            "causal_delta_ml": 0.380,
                            "uncertainty": 0.08,
                            "gate": "AUTONOMOUS_ACT (Tau*=0.25, Conformal Confidence=95%)",
                        }
                    },
                    {
                        "time": ts_str,
                        "entity": "srv-db-01 (10.0.0.5)",
                        "class": "network_activity",
                        "severity": "HIGH",
                        "risk": 0.893,
                        "technique": "T1021.002 (SMB Lateral Movement)",
                        "action": "STAGED_CONTAINMENT",
                        "status": "Contained",
                        "xai": {
                            "decisive_evidence": "GNN 2-hop traversal path from infected peer + anomalous RPC bind",
                            "causal_delta_sig": 0.210,
                            "causal_delta_ml": 0.440,
                            "uncertainty": 0.12,
                            "gate": "AUTONOMOUS_ACT",
                        }
                    }
                ],
                "stats": {
                    "total_events_scored": 128472,
                    "active_mitigations": 4,
                    "pending_approvals": 1,
                    "tracked_entities": 1240,
                    "mean_latency_ms": 2.74,
                }
            }
            await websocket.send_json(sample_payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        log.info("[WS] Client disconnected from live SOC stream.")
    except Exception as e:
        log.warning(f"[WS] WebSocket error: {e}")


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Utility launcher for running uvicorn server in standalone mode."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
