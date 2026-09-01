import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict

log = logging.getLogger(__name__)

class AccessScope(Enum):
    FULL = "FULL"
    RESTRICTED = "RESTRICTED"
    MINIMAL = "MINIMAL"
    REVOKED = "REVOKED"

@dataclass
class ZTREState:
    entity_key: str
    current_scope: AccessScope
    ttl_seconds: int
    last_risk_score: float
    last_update_time: float
    sustained_high_risk_ticks: int = 0

class ZeroTrustRiskEngine:
    """
    Session-level continuous trust evaluation based on risk score DELTA and thresholds.
    Called after every risk_engine scoring pass to dictate auth/RBAC scope.
    """
    def __init__(self):
        self.sessions: Dict[str, ZTREState] = {}
        
        # Policy configurations
        self.DEFAULT_TTL = 86400  # 24 hours
        self.RESTRICTED_TTL = 3600  # 1 hour
        self.MINIMAL_TTL = 300      # 5 minutes
        self.REVOKED_TTL = 0
        
    def evaluate_session(self, entity_key: str, new_risk_score: float) -> ZTREState:
        now = time.time()
        
        if entity_key not in self.sessions:
            self.sessions[entity_key] = ZTREState(
                entity_key=entity_key,
                current_scope=AccessScope.FULL,
                ttl_seconds=self.DEFAULT_TTL,
                last_risk_score=new_risk_score,
                last_update_time=now
            )
            
        state = self.sessions[entity_key]
        
        # Calculate risk delta
        risk_delta = (new_risk_score - state.last_risk_score) * 100.0  # Scale to 0-100 pts
        abs_risk_pts = new_risk_score * 100.0
        
        # Sustained tracking
        if abs_risk_pts > 60:
            state.sustained_high_risk_ticks += 1
        else:
            state.sustained_high_risk_ticks = max(0, state.sustained_high_risk_ticks - 1)
            
        # ZTRE Policy Rules
        old_scope = state.current_scope
        
        if abs_risk_pts >= 90:
            # Absolute critical risk -> Revoke
            state.current_scope = AccessScope.REVOKED
            state.ttl_seconds = self.REVOKED_TTL
        elif risk_delta > 25:
            # Massive sudden jump -> Drop to minimal (containment)
            state.current_scope = AccessScope.MINIMAL
            state.ttl_seconds = self.MINIMAL_TTL
        elif state.sustained_high_risk_ticks > 3 or abs_risk_pts > 60:
            # Sustained elevated risk -> Restricted
            state.current_scope = AccessScope.RESTRICTED
            state.ttl_seconds = self.RESTRICTED_TTL
        elif abs_risk_pts < 30 and risk_delta <= 0:
            # Risk falling back to normal -> Restore
            state.current_scope = AccessScope.FULL
            state.ttl_seconds = self.DEFAULT_TTL
            
        state.last_risk_score = new_risk_score
        state.last_update_time = now
        
        if state.current_scope != old_scope:
            log.info(f"[ZTRE] Session scope for {entity_key} transitioned: {old_scope.name} -> {state.current_scope.name} (Risk={abs_risk_pts:.1f} pts, Delta={risk_delta:+.1f} pts)")
            
        return state

# Singleton
_ztre_instance = None
def get_ztre() -> ZeroTrustRiskEngine:
    global _ztre_instance
    if _ztre_instance is None:
        _ztre_instance = ZeroTrustRiskEngine()
    return _ztre_instance
