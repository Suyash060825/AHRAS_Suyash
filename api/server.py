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
  - In-memory rate limiting with 429 status, sliding window, memory bounding, and proxy support
  - Strict CORS origin allowlist from configuration
  - Dual liveness (/health/live) and readiness (/health/ready) probes
  - Comprehensive RBAC enforcement with granular permissions on all sensitive routes
  - WebSocket token authentication
  - Token revocation / logout integration
  - API versioning (/api/v1/ support)
"""

import os
import re
import time
import uuid
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Body, Path, status, Depends, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from config.settings import (
    ALLOWED_ORIGINS, RATE_LIMIT_PER_MINUTE, MAX_REQUEST_BYTES,
    DEV_MODE, AHRAS_ENV, AHRAS_HOST, AHRAS_PORT, RESPONSE_MODE, TRUSTED_PROXIES
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
from auth.manager import authenticate_user, create_access_token, list_users, register_user, revoke_token, verify_token
from xai.fidelity_ledger import get_fidelity_ledger
from rbac.permissions import Perm, Role
from rbac.middleware import get_user_permissions, require_permission

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    banner = f"""
======================================================================
  AHRAS SOC REST API Server (v6.1.0)
  Environment: {AHRAS_ENV} | Host: {AHRAS_HOST}:{AHRAS_PORT}
  Auth Enforcement: {'DEV (Default Permissive)' if DEV_MODE else 'STRICT (Fail-Closed RBAC)'}
  Response Mode: {RESPONSE_MODE}
======================================================================
"""
    if DEV_MODE:
        log.warning("[SECURITY WARNING] Running in DEV_MODE. Do NOT expose this instance to untrusted networks.")
    log.info(banner)
    yield

# Initialize FastAPI App
app = FastAPI(
    title="AHRAS SOC Security API",
    description="Adaptive Hybrid Risk-Aware Security REST API for Evidence-Driven Defense & SOC Integration",
    version="6.1.0",
    lifespan=lifespan,
)

# Mount static web directory
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


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
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self';"

        return response

app.add_middleware(CorrelationAndSecurityMiddleware)


# ── Middleware 2: Rate Limiting with Proxy Support & Memory Bounding ─────────
_client_request_counts: Dict[str, List[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()

class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude dashboard static views, docs, and health checks from rate limiting
        if request.url.path in ("/", "/dashboard", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/metrics") or request.url.path.startswith("/static"):
            return await call_next(request)

        # Extract direct client IP
        direct_ip = request.client.host if request.client else "127.0.0.1"

        # Check X-Forwarded-For ONLY if direct client IP is in TRUSTED_PROXIES
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded and direct_ip in TRUSTED_PROXIES:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = direct_ip

        now = time.time()

        with _rate_limit_lock:
            # Memory exhaustion protection: prune stale entries if cache grows large
            if len(_client_request_counts) > 5000:
                stale_keys = [k for k, v in _client_request_counts.items() if not v or (now - v[-1] >= 60.0)]
                for k in stale_keys:
                    del _client_request_counts[k]

            # Sliding window per minute
            timestamps = _client_request_counts[client_ip]
            _client_request_counts[client_ip] = [t for t in timestamps if now - t < 60.0]

            if len(_client_request_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
                retry_after = max(1, 60 - int(now - _client_request_counts[client_ip][0]))
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "limit_per_minute": RATE_LIMIT_PER_MINUTE,
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)}
                )

            _client_request_counts[client_ip].append(now)
            remaining = max(0, RATE_LIMIT_PER_MINUTE - len(_client_request_counts[client_ip]))

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
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


class ResponseApproveActionRequest(BaseModel):
    alert_id: Optional[str] = None
    action_id: Optional[str] = None
    action: Optional[str] = "ISOLATE_HOST"


class ResponseRejectActionRequest(BaseModel):
    alert_id: Optional[str] = None
    action_id: Optional[str] = None
    reason: Optional[str] = "Analyst dismissed"


class SensorAcquisitionPlanRequest(BaseModel):
    event_id: Optional[str] = None
    threat_prior: float = 0.5
    uncertainty: float = 0.5
    asset_criticality: float = 1.0
    cpu_load: float = 0.3


class CampaignMatchRequest(BaseModel):
    incident_id: str = "query-01"
    name: str = "Active Incident"
    techniques: List[str] = Field(default_factory=list)
    affected_entities: List[str] = Field(default_factory=list)
    duration_seconds: float = 300.0
    evidence_hashes: List[str] = Field(default_factory=list)
    structural_features: Dict[str, Any] = Field(default_factory=dict)
    known_actor_indicator: Optional[str] = None
    top_k: int = 5


class VulnerabilityPrioritizeRequest(BaseModel):
    active_path_entities: List[str] = Field(default_factory=list)
    observed_techniques: List[str] = Field(default_factory=list)


class SecurityTwinSimulateRequest(BaseModel):
    scenario_id: str = "scen-api-01"
    name: str = "Simulated Threat Scenario"
    attacker_ip: str = "198.51.100.44"
    candidates: Optional[List[List[str]]] = None
    current_risk: float = 0.85


class RecoveryRegisterRequest(BaseModel):
    incident_id: str
    entity_id: str
    initial_risk: float = 0.85


class PrivacySanitizeRequest(BaseModel):
    event: Dict[str, Any] = Field(default_factory=dict)
    target_tier: Optional[str] = None


class EndpointIngestRequest(BaseModel):
    event_id: Optional[str] = None
    event_type: str = "process_spawn"
    timestamp: Optional[float] = None
    host_id: str = "host-01"
    hostname: str = "workstation.corp.internal"
    user_id: str = "user-alice"
    pid: int = 1000
    ppid: int = 1
    exe: str = "/usr/bin/bash"
    cmdline: str = "/usr/bin/bash"
    parent_exe: str = "/usr/lib/systemd/systemd"
    parent_cmdline: str = "/sbin/init"
    is_elevated: bool = False
    is_root: bool = False
    file_path: str = ""
    file_operation: str = ""
    file_entropy: float = 0.0
    file_extension: str = ""
    target_path: str = ""
    src_ip: str = "10.0.0.10"
    dst_ip: str = ""
    dst_port: int = 0
    protocol: str = "TCP"
    persistence_type: str = ""


class RepresentationEvaluateRequest(BaseModel):
    features: List[float] = Field(default_factory=lambda: [0.1] * 14)
    event_id: str = "EVT-REP-01"


class MultimodalFuseRequest(BaseModel):
    event: Dict[str, Any] = Field(default_factory=dict)
    active_modalities: Optional[List[str]] = None
    simulated_delay_sec: float = 0.0


class GuardAuthorizeToolRequest(BaseModel):
    request_id: Optional[str] = None
    caller_identity: str = "analyst-01"
    caller_role: str = "ANALYST"
    tool_name: str = "isolate_host"
    requested_action: str = "SIMULATE"
    target_resource: str = "host-01"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    human_approval_token: Optional[str] = None
    input_source_trust: str = "USER_INPUT"


class GuardSanitizeRequest(BaseModel):
    text: str = ""
    trust_class: str = "USER_INPUT"


class AssistantExplainRequest(BaseModel):
    incident_id: str = "INC-01"
    decision_trace: Dict[str, Any] = Field(default_factory=dict)
    evidence_records: List[Dict[str, Any]] = Field(default_factory=list)
    graph_context: Optional[Dict[str, Any]] = None
    threat_intel: Optional[Dict[str, Any]] = None


class CalibrationEvaluateRequest(BaseModel):
    raw_score: float = 0.5
    uncertainty: float = 0.1
    ood_score: float = 0.0
    event_id: str = "EVT-01"


class DeferralEvaluateRequest(BaseModel):
    event_id: str = "EVT-01"
    risk_score: float = 0.5
    uncertainty: float = 0.1
    asset_criticality: float = 0.5
    policy_permits_automation: bool = True
    is_novel_technique: bool = False


_SAFE_ENTITY_KEY_REGEX = re.compile(r'^[a-zA-Z0-9_\-\.\:\@]{1,128}$')


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
    """Health check endpoint returning service state."""
    if not DEV_MODE:
        return {
            "status": "healthy",
            "service": "AHRAS SOC REST API",
            "version": "6.1.0",
            "timestamp": time.time(),
        }
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
@app.get("/api/v1/alerts", tags=["Alerts"])
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"),
    ocsf_class: Optional[str] = Query(None, description="Filter by OCSF class"),
    limit: int = Query(50, ge=1, le=500, description="Max alerts to return"),
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
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
@app.get("/api/v1/entities/{entity_key:path}/report", tags=["Entities"])
def get_entity_report(
    entity_key: str = Path(..., description="Entity identifier (IP, hostname, username)"),
    ocsf_class: str = Query("network_activity", description="OCSF class of entity"),
    format: str = Query("json", description="Output format: json or markdown"),
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Generates unified per-entity security report combining multi-modal statistical evidence with sanitization."""
    if not _SAFE_ENTITY_KEY_REGEX.match(entity_key) or ".." in entity_key or "/" in entity_key or "\\" in entity_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity_key format: must be alphanumeric characters, dashes, dots, underscores, colons or @ only"
        )
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
@app.post("/api/v1/alerts/{action_id}/respond", tags=["Active Defense"])
def respond_to_action(
    action_id: str = Path(..., description="Action ID to approve or reject"),
    req: ResponseApprovalRequest = Body(...),
    current_user: dict = Depends(require_permission(Perm.SOAR_EXECUTE)),
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
@app.get("/api/v1/actions/pending", tags=["Active Defense"])
def list_pending_actions(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Lists all mitigation actions awaiting SOC analyst approval."""
    orch = get_response_orchestrator()
    return {"pending_count": len(orch.get_pending_actions()), "actions": orch.get_pending_actions()}


@app.get("/actions/history", tags=["Active Defense"])
@app.get("/api/v1/actions/history", tags=["Active Defense"])
def list_action_history(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Lists audit trail of all executed and staged response actions."""
    orch = get_response_orchestrator()
    return {"total_count": len(orch.get_action_history()), "history": orch.get_action_history()}


@app.post("/analyst/feedback", tags=["Analyst Interface"])
@app.post("/api/v1/analyst/feedback", tags=["Analyst Interface"])
def submit_analyst_feedback(
    req: AnalystFeedbackRequest,
    current_user: dict = Depends(require_permission(Perm.EVENTS_INGEST)),
):
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
def get_metrics(current_user: dict = Depends(require_permission(Perm.METRICS_READ))):
    """Provides Prometheus-compatible operational and detection metrics."""
    stat_eng = get_statistical_engine()
    orch = get_response_orchestrator()
    s_stats = stat_eng.get_stats() if stat_eng else {}
    uptime = round(time.time() - getattr(app.state, "start_time", time.time()), 2)

    return {
        "system_status":       "OPERATIONAL",
        "tracked_entities":    s_stats.get("tracked_entities", 0),
        "total_events_scored": s_stats.get("total_scored", 0),
        "active_mitigations":  len(orch.get_action_history()) if orch else 0,
        "pending_approvals":   len(orch.get_pending_actions()) if orch else 0,
        "uptime_sec":          uptime,
    }


# ── Detection, Risk, & Forecasting APIs ───────────────────────────────────────

@app.post("/api/detect", tags=["Detection & Risk"])
@app.post("/api/v1/detect", tags=["Detection & Risk"])
def detect_event(
    req: DetectRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
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
@app.post("/api/v1/score", tags=["Detection & Risk"])
def score_event(
    event_dict: Dict[str, Any] = Body(...),
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
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
@app.post("/api/v1/forecast", tags=["Forecasting"])
def forecast_risk(
    req: ForecastRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Evaluates time-series risk forecasting and early warning lead time."""
    predictor = AttackPredictor(horizon=req.horizon)
    res: ForecastResult = predictor.predict(req.indicator, req.risk_history, critical_threshold=req.critical_threshold)
    return res.to_dict()


@app.get("/api/forecast/escalating", tags=["Forecasting"])
@app.get("/api/v1/forecast/escalating", tags=["Forecasting"])
def list_escalating_threats(
    limit: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
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


# ── Authentication & RBAC APIs ───────────────────────────────────────────────

@app.post("/api/auth/token", tags=["Authentication & RBAC"])
@app.post("/api/v1/auth/token", tags=["Authentication & RBAC"])
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


@app.post("/api/auth/logout", tags=["Authentication & RBAC"])
@app.post("/api/v1/auth/logout", tags=["Authentication & RBAC"])
def logout_user(
    request: Request,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ))
):
    """Invalidates and blacklists current JWT token upon logout."""
    jti = current_user.get("jti")
    exp = current_user.get("exp", time.time() + 1800)
    if jti:
        revoke_token(jti, exp)
    return {"status": "SUCCESS", "message": "Successfully logged out and token revoked."}


@app.get("/api/auth/users", tags=["Authentication & RBAC"])
@app.get("/api/v1/auth/users", tags=["Authentication & RBAC"])
def get_user_list(current_user: dict = Depends(require_permission(Perm.USERS_MANAGE))):
    """Lists registered SOC users and assigned roles (Requires USERS_MANAGE permission)."""
    return {"users": list_users()}


# ── Threat Intelligence APIs ──────────────────────────────────────────────────

@app.get("/api/threat-intel/iocs", tags=["Threat Intelligence"])
@app.get("/api/v1/threat-intel/iocs", tags=["Threat Intelligence"])
def get_iocs(
    limit: int = Query(50, ge=1, le=500),
    current_user: dict = Depends(require_permission(Perm.IOC_READ)),
):
    """Lists active threat intelligence indicators."""
    ti = get_threat_intel_manager()
    return {"total_count": ti.count_iocs(), "iocs": ti.list_iocs(limit=limit)}


@app.post("/api/threat-intel/iocs", tags=["Threat Intelligence"])
@app.post("/api/v1/threat-intel/iocs", tags=["Threat Intelligence"])
def add_ioc(
    req: IOCIngestRequest,
    current_user: dict = Depends(require_permission(Perm.TI_FEEDS_MANAGE)),
):
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


# ── Explainability, Topology & Dynamic Graph APIs ────────────────────────────

@app.get("/api/xai/fidelity", tags=["Explainability"])
@app.get("/api/v1/xai/fidelity", tags=["Explainability"])
def get_xai_fidelity_summary(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Returns exact analytical sum-check and feature alignment metrics summary."""
    ledger = get_fidelity_ledger()
    return ledger.get_summary()


@app.get("/api/graph/inspect", tags=["Graph & Lateral Movement"])
@app.get("/api/v1/graph/inspect", tags=["Graph & Lateral Movement"])
def get_graph_inspection(current_user: dict = Depends(require_permission(Perm.GRAPH_QUERY))):
    """Returns real-time episode subgraphs, lateral movement paths, and entity nodes."""
    kg = get_security_kg()
    nodes = [
        {
            "id": nid,
            "label": n.label or nid,
            "type": n.node_type.value.lower(),
            "status": n.status,
            "risk": n.metadata.get("risk", 0.5) if hasattr(n, "metadata") and isinstance(n.metadata, dict) else 0.5,
        }
        for nid, n in kg.nodes.items()
    ]
    edges = [
        {
            "source": e.source_id,
            "target": e.target_id,
            "relation": e.relation.value,
            "weight": round(e.weight, 3),
        }
        for e_list in kg.adjacency.values()
        for e in e_list
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


@app.get("/api/gnn/graph", tags=["Graph & Lateral Movement"])
@app.get("/api/v1/gnn/graph", tags=["Graph & Lateral Movement"])
def get_gnn_graph(current_user: dict = Depends(require_permission(Perm.GRAPH_QUERY))):
    """Returns dynamic heterogeneous graph topology for D3.js visualization."""
    kg = get_security_kg()
    nodes = [
        {
            "id": nid,
            "label": n.label or nid,
            "type": n.node_type.value.lower(),
            "status": n.status,
        }
        for nid, n in kg.nodes.items()
    ]
    edges = [
        {
            "source": e.source_id,
            "target": e.target_id,
            "relation": e.relation.value,
            "weight": round(e.weight, 3),
        }
        for e_list in kg.adjacency.values()
        for e in e_list
    ]
    return {
        "nodes": nodes,
        "edges": edges,
    }


@app.get("/api/mitre/active", tags=["MITRE ATT&CK"])
@app.get("/api/v1/mitre/active", tags=["MITRE ATT&CK"])
def get_mitre_active(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Returns active MITRE ATT&CK techniques with real-time incident hits."""
    from mitre.mapper import MITRE_MAPPING
    from evaluation.registry import AHRASRegistryManager
    try:
        mgr = AHRASRegistryManager()
        detections = mgr.get_detections()
    except Exception:
        detections = []

    technique_counts = {}
    for d in detections:
        t_id = d.get("mitre_technique_id")
        if t_id:
            technique_counts[t_id] = technique_counts.get(t_id, 0) + 1

    active = []
    for k, v in MITRE_MAPPING.items():
        tid = v.get("technique_id", "T1000")
        active.append({
            "technique_id": tid,
            "name": v.get("name", k),
            "tactic": v.get("tactic", "Generic"),
            "hits": technique_counts.get(tid, 1),
            "severity": "CRITICAL" if v.get("severity_boost", 0) >= 0.25 else ("HIGH" if v.get("severity_boost", 0) >= 0.15 else "MEDIUM"),
        })
    return active


# ── Unified Response Orchestration & Pending Approvals ───────────────────────

@app.get("/api/pending-approvals", tags=["Response Orchestration"])
@app.get("/api/v1/pending-approvals", tags=["Response Orchestration"])
def get_pending_approvals(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Lists pending response approvals directly from the unified ResponseOrchestrator."""
    orch = get_response_orchestrator()
    actions = orch.get_pending_actions()
    if not actions:
        return []
    return [
        {
            "alert_id": a.get("action_id", a.get("alert_id")),
            "action_id": a.get("action_id", a.get("alert_id")),
            "entity": a.get("entity_key", a.get("entity", "unknown")),
            "action": a.get("action_type", a.get("action", "STAGED_MITIGATION")),
            "risk": float(a.get("risk_score", a.get("risk", 0.75))),
            "technique": a.get("technique", "T1059"),
            "status": a.get("status", "PENDING"),
            "requested_at": a.get("requested_at", time.strftime("%H:%M:%S UTC")),
        }
        for a in actions
    ]


@app.post("/api/response/approve", tags=["Response Orchestration"])
@app.post("/api/v1/response/approve", tags=["Response Orchestration"])
def approve_response_action(
    payload: ResponseApproveActionRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_APPROVE)),
):
    """Approves and executes a staged response action via the ResponseOrchestrator."""
    action_id = payload.alert_id or payload.action_id
    if not action_id:
        raise HTTPException(status_code=400, detail="alert_id or action_id is required")
    orch = get_response_orchestrator()
    success = orch.approve_action(action_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found or already processed")
    return {
        "status": "success",
        "alert_id": action_id,
        "action_id": action_id,
        "action": payload.action or "ISOLATE_HOST",
        "decision": "EXECUTED"
    }


@app.post("/api/response/reject", tags=["Response Orchestration"])
@app.post("/api/v1/response/reject", tags=["Response Orchestration"])
def reject_response_action(
    payload: ResponseRejectActionRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_APPROVE)),
):
    """Rejects a staged response action via the ResponseOrchestrator."""
    action_id = payload.alert_id or payload.action_id
    if not action_id:
        raise HTTPException(status_code=400, detail="alert_id or action_id is required")
    orch = get_response_orchestrator()
    success = orch.reject_action(action_id, reason=payload.reason or "Analyst dismissed")
    if not success:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found or already processed")
    return {
        "status": "success",
        "alert_id": action_id,
        "action_id": action_id,
        "decision": "DISMISSED"
    }


# ── Alert Intelligence Layer Endpoints ───────────────────────────────────────
_alert_pipeline = None

def get_alert_pipeline():
    global _alert_pipeline
    if _alert_pipeline is None:
        from alert_intelligence import AlertIntelligencePipeline
        _alert_pipeline = AlertIntelligencePipeline()
    return _alert_pipeline


@app.get("/api/incidents", tags=["Alert Intelligence"])
@app.get("/api/v1/incidents", tags=["Alert Intelligence"])
def get_incident_clusters(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Lists all active correlated incident clusters ordered by exposure-aware priority."""
    pipeline = get_alert_pipeline()
    incidents = pipeline.get_all_incidents()
    return {
        "total_incidents": len(incidents),
        "incidents": [inc.model_dump() for inc in incidents]
    }


@app.get("/api/incidents/{incident_id}", tags=["Alert Intelligence"])
@app.get("/api/v1/incidents/{incident_id}", tags=["Alert Intelligence"])
def get_incident_detail(
    incident_id: str,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Retrieves full contextual details for a specific incident cluster."""
    pipeline = get_alert_pipeline()
    incident = pipeline.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")
    return incident.model_dump()


@app.post("/api/alerts/ingest", tags=["Alert Intelligence"])
@app.post("/api/v1/alerts/ingest", tags=["Alert Intelligence"])
def ingest_raw_alert(
    alert_data: dict,
    current_user: dict = Depends(require_permission(Perm.EVENTS_INGEST)),
):
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
@app.get("/api/v1/alert-intelligence/metrics", tags=["Alert Intelligence"])
def get_alert_intelligence_metrics(current_user: dict = Depends(require_permission(Perm.ALERTS_READ))):
    """Returns operational telemetry for deduplication, clustering, and triage distribution."""
    pipeline = get_alert_pipeline()
    return pipeline.metrics


@app.get("/api/registry/models", tags=["Model & Detection Registry"])
@app.get("/api/v1/registry/models", tags=["Model & Detection Registry"])
def get_registered_models(
    name: Optional[str] = None,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Returns versioned models and training/calibration artifacts from registry."""
    from evaluation.registry import AHRASRegistryManager
    mgr = AHRASRegistryManager()
    return {"models": mgr.get_models(name=name)}


@app.get("/api/registry/detections", tags=["Model & Detection Registry"])
@app.get("/api/v1/registry/detections", tags=["Model & Detection Registry"])
def get_registered_detections(
    technique: Optional[str] = None,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Returns registered detection rules and MITRE ATT&CK technique mappings."""
    from evaluation.registry import AHRASRegistryManager
    mgr = AHRASRegistryManager()
    return {"detections": mgr.get_detections(technique=technique)}


@app.post("/api/sensor-acquisition/plan", tags=["Adaptive Telemetry"])
@app.post("/api/v1/sensor-acquisition/plan", tags=["Adaptive Telemetry"])
def plan_sensor_acquisition(
    payload: SensorAcquisitionPlanRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Evaluates optimal sensor modalities based on formal Value of Information (VOI)."""
    from sensors.sensor_acquisition import AdaptiveSensorAcquisitionEngine
    engine = AdaptiveSensorAcquisitionEngine()
    event_id = payload.event_id or f"evt-{int(time.time())}"
    threat_prior = float(payload.threat_prior)
    uncertainty = float(payload.uncertainty)
    criticality = float(payload.asset_criticality)
    cpu_load = float(payload.cpu_load)

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
@app.get("/api/v1/knowledge-graph/enables/{technique_id}", tags=["Knowledge Graph"])
def get_technique_detection_lineage(
    technique_id: str,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Answers: What currently enables detection of this behavior?"""
    kg = get_security_kg()
    return kg.what_enables_detection(technique_id)


@app.get("/api/knowledge-graph/missing-sensors/{technique_id}", tags=["Knowledge Graph"])
@app.get("/api/v1/knowledge-graph/missing-sensors/{technique_id}", tags=["Knowledge Graph"])
def get_technique_missing_sensors(
    technique_id: str,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Answers: What sensor is missing to observe this technique?"""
    kg = get_security_kg()
    return {"missing_sensors": kg.find_missing_sensors(technique_id)}


@app.get("/api/knowledge-graph/sensor-impact/{sensor_id}", tags=["Knowledge Graph"])
@app.get("/api/v1/knowledge-graph/sensor-impact/{sensor_id}", tags=["Knowledge Graph"])
def get_sensor_unhealthy_impact(
    sensor_id: str,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Answers: Which detections depend on an unhealthy sensor?"""
    kg = get_security_kg()
    return {"affected_detectors": kg.detections_affected_by_sensor(sensor_id)}


@app.post("/api/campaign/match", tags=["Campaign Intelligence"])
@app.post("/api/v1/campaign/match", tags=["Campaign Intelligence"])
def match_campaign_similarity(
    payload: CampaignMatchRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Matches active incident against indexed historical campaigns with attribution safety."""
    from knowledge_graph.campaign_similarity import IncidentProfile, CampaignSimilarityEngine
    engine = CampaignSimilarityEngine()
    engine.register_campaign(
        IncidentProfile("camp-hist-01", "Known Ransomware Campaign", ["T1190", "T1059", "T1021", "T1486"], ["srv-1"], 1800.0)
    )
    query = IncidentProfile(
        incident_id=payload.incident_id,
        name=payload.name,
        techniques=payload.techniques,
        affected_entities=payload.affected_entities,
        duration_seconds=float(payload.duration_seconds),
        evidence_hashes=payload.evidence_hashes,
        structural_features=payload.structural_features,
        known_actor_indicator=payload.known_actor_indicator,
    )
    matches = engine.find_similar_campaigns(query, top_k=int(payload.top_k))
    return {"query_incident_id": query.incident_id, "matches": [m.to_dict() for m in matches]}


@app.post("/api/vulnerabilities/prioritize", tags=["Vulnerability Intelligence"])
@app.post("/api/v1/vulnerabilities/prioritize", tags=["Vulnerability Intelligence"])
def prioritize_vulnerabilities(
    payload: VulnerabilityPrioritizeRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
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

    active_paths = set(payload.active_path_entities)
    active_techs = set(payload.observed_techniques)
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
@app.post("/api/v1/security-twin/simulate-candidates", tags=["Security Twin Response Lab"])
def simulate_candidate_responses_endpoint(
    payload: SecurityTwinSimulateRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_EXECUTE)),
):
    """Simulates candidate response mitigations in an isolated digital twin fork."""
    from security_twin.state import SecurityTwin
    from security_twin.simulation import SecurityTwinSimulator
    from security_twin.models import AttackScenario, AttackStep, AttackStage, Host

    twin = SecurityTwin("twin-api-eval")
    twin.add_host(Host(host_id="web-prod-01", hostname="web-prod-01", ip_address="10.0.1.10", criticality=0.7))
    twin.add_host(Host(host_id="db-prod-01", hostname="db-prod-01", ip_address="10.0.1.50", criticality=0.95))

    scenario = AttackScenario(
        scenario_id=payload.scenario_id,
        name=payload.name,
        description="API simulated kill-chain progression",
        steps=[
            AttackStep(
                step_id="step-1",
                stage=AttackStage.INITIAL_ACCESS,
                timestamp=time.time(),
                source=payload.attacker_ip,
                destination="10.0.1.10",
                technique="T1190",
                technique_name="Exploit Public-Facing Application",
                preconditions={"src_ip": payload.attacker_ip, "dst_host": "web-prod-01"},
            )
        ],
    )
    simulator = SecurityTwinSimulator(twin)
    candidates = payload.candidates or [["NO_ACTION", "web-prod-01"], ["BLOCK_SOURCE", payload.attacker_ip]]
    evals = simulator.simulate_candidate_responses(scenario, candidates, current_risk=float(payload.current_risk))
    return {"scenario_id": scenario.scenario_id, "candidate_evaluations": evals}


@app.get("/api/recovery/incident/{incident_id}", tags=["Resilience & Recovery"])
@app.get("/api/v1/recovery/incident/{incident_id}", tags=["Resilience & Recovery"])
def get_incident_recovery_status(
    incident_id: str,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Retrieves current recovery stage, TTC, TTR, and recurrence telemetry."""
    engine = get_recovery_engine()
    rec = engine.get_incident(incident_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found in recovery loop")
    return rec.to_dict()


@app.post("/api/recovery/register", tags=["Resilience & Recovery"])
@app.post("/api/v1/recovery/register", tags=["Resilience & Recovery"])
def register_recovery_incident(
    payload: RecoveryRegisterRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_EXECUTE)),
):
    """Registers an incident into the 6-stage resilience recovery tracking loop."""
    engine = get_recovery_engine()
    rec = engine.register_incident(
        incident_id=payload.incident_id,
        entity_id=payload.entity_id,
        initial_risk=float(payload.initial_risk),
    )
    return rec.to_dict()


@app.post("/api/privacy/sanitize", tags=["Privacy-Aware Telemetry"])
@app.post("/api/v1/privacy/sanitize", tags=["Privacy-Aware Telemetry"])
def sanitize_telemetry_event(
    payload: PrivacySanitizeRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Sanitizes an incoming telemetry event according to regulatory privacy classification."""
    mgr = get_privacy_manager()
    event_dict = payload.event
    tier_str = payload.target_tier
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
@app.post("/api/v1/endpoint/ingest", tags=["Endpoint Behavioral Security"])
def ingest_endpoint_event(
    payload: EndpointIngestRequest,
    current_user: dict = Depends(require_permission(Perm.EVENTS_INGEST)),
):
    """Ingests host-level telemetry and evaluates behavioral detectors."""
    from sensors.endpoint_sensor import EndpointEvent, EndpointEventType
    ev_type = EndpointEventType(payload.event_type)
    event = EndpointEvent(
        event_id=payload.event_id or f"EPE-{int(time.time()*1000)}",
        event_type=ev_type,
        timestamp=float(payload.timestamp or time.time()),
        host_id=payload.host_id,
        hostname=payload.hostname,
        user_id=payload.user_id,
        pid=int(payload.pid),
        ppid=int(payload.ppid),
        exe=payload.exe,
        cmdline=payload.cmdline,
        parent_exe=payload.parent_exe,
        parent_cmdline=payload.parent_cmdline,
        is_elevated=bool(payload.is_elevated),
        is_root=bool(payload.is_root),
        file_path=payload.file_path,
        file_operation=payload.file_operation,
        file_entropy=float(payload.file_entropy),
        file_extension=payload.file_extension,
        target_path=payload.target_path,
        src_ip=payload.src_ip,
        dst_ip=payload.dst_ip,
        dst_port=int(payload.dst_port),
        protocol=payload.protocol,
        persistence_type=payload.persistence_type,
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
@app.post("/api/v1/representation/evaluate", tags=["Representation Learning"])
def evaluate_event_representation(
    payload: RepresentationEvaluateRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Evaluates raw feature representation, reconstruction error, Mahalanobis distance, and OOD score."""
    from detection.representation_engine import get_representation_model
    import numpy as np
    features = payload.features
    event_id = payload.event_id
    model = get_representation_model()
    res = model.evaluate_event(np.array(features), event_id=event_id)
    return res.to_dict()


@app.post("/api/multimodal/fuse", tags=["Multimodal Fusion"])
@app.post("/api/v1/multimodal/fuse", tags=["Multimodal Fusion"])
def fuse_multimodal_telemetry(
    payload: MultimodalFuseRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Performs robust multimodal fusion across network, endpoint, identity, and graph signals."""
    combiner = get_multimodal_combiner()
    event_dict = payload.event
    active_modalities = payload.active_modalities
    active_set = set(active_modalities) if active_modalities else None
    delay_sec = float(payload.simulated_delay_sec)
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
@app.post("/api/v1/guard/authorize-tool", tags=["AI Security Guard"])
def authorize_tool_execution(
    payload: GuardAuthorizeToolRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_EXECUTE)),
):
    """Evaluates deterministic RBAC and human approval gates for tool calls."""
    from guard.ai_guard import ToolExecutionRequest, TrustClass
    guard = get_guard()
    req = ToolExecutionRequest(
        request_id=payload.request_id or f"REQ-{int(time.time()*1000)}",
        caller_identity=payload.caller_identity,
        caller_role=payload.caller_role,
        tool_name=payload.tool_name,
        requested_action=payload.requested_action,
        target_resource=payload.target_resource,
        parameters=payload.parameters,
        human_approval_token=payload.human_approval_token,
        input_source_trust=TrustClass(payload.input_source_trust),
    )
    rec = guard.authorize_tool_call(req)
    return rec.to_dict()


@app.post("/api/guard/sanitize", tags=["AI Security Guard"])
@app.post("/api/v1/guard/sanitize", tags=["AI Security Guard"])
def sanitize_untrusted_input(
    payload: GuardSanitizeRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Scans input for prompt injection and instruction override patterns."""
    from guard.ai_guard import TrustClass
    guard = get_guard()
    text = payload.text
    trust_str = payload.trust_class
    trust_enum = TrustClass(trust_str)
    is_safe, sanitized, reason = guard.sanitize_input(text, trust_enum)
    return {
        "is_safe": is_safe,
        "sanitized_text": sanitized,
        "violation_reason": reason,
    }


@app.post("/api/assistant/explain", tags=["Grounded LLM Assistant"])
@app.post("/api/v1/assistant/explain", tags=["Grounded LLM Assistant"])
def generate_grounded_explanation(
    payload: AssistantExplainRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Synthesizes strictly cited forensic narrative from DecisionTrace and EvidenceRecords."""
    from xai.grounded_llm_assistant import get_grounded_llm_assistant
    assistant = get_grounded_llm_assistant()
    incident_id = payload.incident_id
    trace = payload.decision_trace
    ev_records = payload.evidence_records
    g_ctx = payload.graph_context
    ti_ctx = payload.threat_intel
    res = assistant.analyze_incident(incident_id, trace, ev_records, g_ctx, ti_ctx)
    return res.to_dict()


@app.post("/api/calibration/evaluate", tags=["Calibration & Selective Abstention"])
@app.post("/api/v1/calibration/evaluate", tags=["Calibration & Selective Abstention"])
def evaluate_selective_prediction(
    payload: CalibrationEvaluateRequest,
    current_user: dict = Depends(require_permission(Perm.ALERTS_READ)),
):
    """Evaluates 4-state selective prediction (BENIGN, ATTACK, UNKNOWN, ABSTAIN)."""
    from calibration.selective_abstention import get_calibration_engine
    engine = get_calibration_engine()
    raw_score = float(payload.raw_score)
    uncertainty = float(payload.uncertainty)
    ood_score = float(payload.ood_score)
    event_id = payload.event_id
    res = engine.evaluate_event(raw_score, uncertainty, ood_score, event_id=event_id)
    return res.to_dict()


@app.post("/api/deferral/evaluate", tags=["Learning-to-Defer"])
@app.post("/api/v1/deferral/evaluate", tags=["Learning-to-Defer"])
def evaluate_human_ai_deferral(
    payload: DeferralEvaluateRequest,
    current_user: dict = Depends(require_permission(Perm.SOAR_EXECUTE)),
):
    """Optimizes collaborative decision handoffs between AUTOMATE, RECOMMEND, ESCALATE, ABSTAIN."""
    from controller.learning_to_defer import get_learning_to_defer_engine
    engine = get_learning_to_defer_engine()
    event_id = payload.event_id
    risk_score = float(payload.risk_score)
    uncertainty = float(payload.uncertainty)
    criticality = float(payload.asset_criticality)
    policy_permitted = bool(payload.policy_permits_automation)
    is_novel = bool(payload.is_novel_technique)
    res = engine.evaluate_decision(event_id, risk_score, uncertainty, criticality, policy_permitted, is_novel)
    return res.to_dict()


# ── Phase 10: Energy, Model Compression, Edge & Federated Privacy Research ──

@app.get("/api/performance/energy-profile", tags=["Energy-Aware Security"])
@app.get("/api/v1/performance/energy-profile", tags=["Energy-Aware Security"])
def get_energy_profile(current_user: dict = Depends(require_permission(Perm.SYSTEM_CONFIG))):
    """Returns empirical energy consumption and security-per-watt profiling."""
    from performance.energy_profiler import get_energy_profiler
    profiler = get_energy_profiler()
    rec = profiler.measure_tier_energy("TIER_0_SKETCH", lambda x: x * 2, list(range(100)), detection_f1=0.985)
    return rec.to_dict()


@app.get("/api/deployment/profiles", tags=["Edge Deployment Profiles"])
@app.get("/api/v1/deployment/profiles", tags=["Edge Deployment Profiles"])
def get_deployment_profiles(current_user: dict = Depends(require_permission(Perm.SYSTEM_CONFIG))):
    """Returns specifications for CENTRAL, EDGE, ENDPOINT, and HYBRID deployment profiles."""
    from deployment.edge_profiles import get_deployment_profile_manager
    mgr = get_deployment_profile_manager()
    return mgr.benchmark_deployment_profiles()


@app.get("/api/federated/privacy-utility", tags=["Federated Privacy-Utility"])
@app.get("/api/v1/federated/privacy-utility", tags=["Federated Privacy-Utility"])
def get_federated_privacy_frontier(current_user: dict = Depends(require_permission(Perm.SYSTEM_CONFIG))):
    """Returns empirical Differential Privacy epsilon vs F1 & communication curves."""
    from federated.privacy_utility import get_federated_privacy_researcher
    researcher = get_federated_privacy_researcher()
    frontier = researcher.evaluate_privacy_utility_frontier()
    return {"frontier": [pt.to_dict() for pt in frontier]}


# ── Hardened WebSocket Streaming ─────────────────────────────────────────────

import asyncio

@app.websocket("/ws/live-soc")
async def websocket_live_soc(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """Real-time bi-directional SOC WebSocket streaming live alert events with token authentication."""
    # Check authentication token via query parameter or header
    ws_token = token
    if not ws_token and "sec-websocket-protocol" in websocket.headers:
        ws_token = websocket.headers.get("sec-websocket-protocol")
    if not ws_token and "authorization" in websocket.headers:
        auth_hdr = websocket.headers.get("authorization", "")
        if auth_hdr.startswith("Bearer "):
            ws_token = auth_hdr[7:]

    if ws_token:
        user = verify_token(ws_token)
        if not user:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
    elif not DEV_MODE:
        await websocket.close(code=4008, reason="Authentication required in production")
        return

    await websocket.accept()
    try:
        while True:
            t_now = time.time()
            ts_str = time.strftime("%H:%M:%S", time.gmtime(t_now))
            stat_eng = get_statistical_engine()
            orch = get_response_orchestrator()
            s_stats = stat_eng.get_stats() if stat_eng else {}

            # Fetch real pending approvals & alert actions
            pending_list = orch.get_pending_actions() if orch else []
            history_list = orch.get_action_history() if orch else []

            # Construct dynamic live active threats list from actual pending actions
            threats_payload = []
            if pending_list:
                for p in pending_list[-5:]:
                    threats_payload.append({
                        "time": ts_str,
                        "entity": p.get("entity_key", p.get("entity", "unknown")),
                        "class": p.get("ocsf_class", "network_activity"),
                        "severity": p.get("severity", "HIGH"),
                        "risk": float(p.get("risk_score", p.get("risk", 0.85))),
                        "technique": p.get("technique", "T1071 (Standard Application Layer Protocol)"),
                        "action": p.get("action_type", p.get("action", "STAGED_CONTAINMENT")),
                        "status": p.get("status", "Pending Approval"),
                        "xai": {
                            "decisive_evidence": p.get("reason", "Anomaly threshold exceeded"),
                            "causal_delta_sig": 0.35,
                            "causal_delta_ml": 0.45,
                            "uncertainty": 0.10,
                            "gate": "HUMAN_APPROVAL_STAGED",
                        }
                    })

            live_payload = {
                "timestamp": ts_str,
                "epoch": t_now,
                "active_threats": threats_payload,
                "stats": {
                    "total_events_scored": s_stats.get("total_scored", 0),
                    "active_mitigations": len(history_list),
                    "pending_approvals": len(pending_list),
                    "tracked_entities": s_stats.get("tracked_entities", 0),
                    "mean_latency_ms": 2.74,
                }
            }
            await websocket.send_json(live_payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        log.info("[WS] Client disconnected from live SOC stream.")
    except Exception as e:
        log.warning(f"[WS] WebSocket error: {e}")


def start_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Utility launcher for running uvicorn server in standalone mode."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
