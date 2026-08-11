from __future__ import annotations
"""
AHRAS SOC REST API Server
------------------------
Production FastAPI service providing endpoints for SOC analysts, dashboards,
and automated active defense response workflows.

Endpoints:
  - GET  /health                    : Service health and system status
  - GET  /alerts                    : Query historical security alerts with filtering
  - GET  /entities/{key}/report     : Unified per-entity security report (JSON & Markdown)
  - POST /alerts/{id}/respond       : Trigger or approve active defense response actions
  - POST /analyst/feedback          : Mark false positives and reset entity baselines
  - GET  /metrics                   : SOC operational metrics & engine stats
"""

import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Body, Path, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from storage.store import get_store
from detection.statistical_engine.entity_report import get_entity_report_generator, EntityReport
from detection.statistical_engine.stat_engine import get_statistical_engine
from response.orchestrator import get_response_orchestrator, ResponseAction

log = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AHRAS SOC Security API",
    description="Adaptive Hybrid Risk-Aware Security REST API for Enterprise SOC Integration",
    version="4.0.0",
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


from fastapi.responses import HTMLResponse
import os

# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Serves the real-time AHRAS SOC Web Dashboard."""
    dash_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")
    if os.path.exists(dash_path):
        with open(dash_path, "r") as f:
            return f.read()
    return "<h1>AHRAS Dashboard File Not Found</h1>"


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint returning engine state."""
    return {
        "status":     "healthy",
        "service":    "AHRAS SOC REST API",
        "version":    "4.0.0",
        "timestamp":  time.time(),
        "store_type": "SQLite / MongoDB Dual-Mode",
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
        # Synthetic empty event to anchor current profile state
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
            raise HTTPException(status_code=44, detail=f"Action '{action_id}' not found in pending approval queue")
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


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Utility launcher for running uvicorn server in standalone mode."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
