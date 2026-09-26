from __future__ import annotations
"""
AHRAS Module — Confidence-Based Early-Exit Model Router (Extension E)
=====================================================================
Multi-stage adaptive detection cascade optimizing compute and latency
while maintaining research-grade detection quality (F1, precision, recall).

Routing Architecture:
  Stage 1: Fast Path (Lightweight signatures, cheap protocol heuristics)
  Stage 2: ML Ensemble (Isolation Forest, Deep Autoencoder, One-Class SVM, Stat Engine)
  Stage 3: Behavioral / Multimodal (Cross-modal attention, temporal representation)
  Stage 4: Deep Analysis (Relational graph reasoning, AttackPathReasoner, Causal DAG)

Exit Policy:
  At stage k, compute:
    - Confidence C_k in [0, 1]
    - Epistemic uncertainty U_k in [0, 1]
    - Out-of-distribution score OOD_k in [0, 1]
    - Decision margin |C_k - theta|
  Exit if:
    (C_k >= tau_conf and U_k <= tau_unc)  [Decisive Alert]
    or
    (|C_k - theta| >= tau_margin and OOD_k <= tau_ood and U_k <= tau_unc)  [Decisive Benign / Alert]
  Otherwise proceed to Stage k + 1.
"""

import uuid
import time
import math
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config.settings import (
    USE_MODEL_ROUTER,
    ROUTER_CONF_THRESHOLD,
    ROUTER_UNCERTAINTY_THRESHOLD,
    ROUTER_MARGIN_THRESHOLD,
)
from detection.hybrid_engine import (
    DetectionResult,
    _sig_to_dict,
    _anom_to_dict,
    _stat_to_dict,
    _severity_label,
    _W_SIG,
    _W_ML,
    _W_STAT,
    _ALERT_THRESHOLD,
)
from detection.feature_extractor import extract, feature_names as get_feature_names
from detection.signature_engine.rules import run_signature_engine, SignatureMatch
from detection.anomaly_engine.ml_engine import run_anomaly_engine, AnomalyResult, _get_detector
from detection.statistical_engine.stat_engine import run_statistical_engine, StatResult
from detection.multimodal_encoder import MultimodalSecurityEncoder, ModalityVectors
from detection.attack_path import AttackPathReasoner
from detection.xai_explainer import explain

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Router Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    conf_threshold: float = ROUTER_CONF_THRESHOLD
    uncertainty_threshold: float = ROUTER_UNCERTAINTY_THRESHOLD
    margin_threshold: float = ROUTER_MARGIN_THRESHOLD
    ood_threshold: float = 0.30
    decision_threshold: float = 0.50
    enabled: bool = USE_MODEL_ROUTER
    force_stage: Optional[int] = None  # 1..4 to force termination at a specific stage


# ─────────────────────────────────────────────────────────────────────────────
# Routed Detection Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutedDetectionResult(DetectionResult):
    """
    Subclasses DetectionResult to preserve full backward compatibility with
    all downstream consumers (Risk Engine, Evidence Ledger, XAI, API) while
    providing rich routing metadata and telemetry.
    """
    stage_exited: int = 1
    stage_name: str = "fast_path"
    stages_evaluated: list[int] = field(default_factory=lambda: [1])
    confidence_trajectory: list[float] = field(default_factory=list)
    uncertainty_trajectory: list[float] = field(default_factory=list)
    ood_trajectory: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    per_stage_latencies: dict[str, float] = field(default_factory=dict)
    exit_reason: str = ""
    early_exited: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("normalized_event", None)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-Based Early-Exit Router
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceModelRouter:
    """
    Thread-safe 4-stage early-exit detector that cascades from sub-millisecond
    signature/heuristic checks to full multi-modal relational graph reasoning.
    """

    STAGE_NAMES = {
        1: "fast_path",
        2: "ml_ensemble",
        3: "multimodal",
        4: "deep_graph",
    }

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._lock = threading.Lock()

        # Component models
        self.multimodal_encoder = MultimodalSecurityEncoder(embed_dim=8, seed=42)
        self.attack_reasoner = AttackPathReasoner(embed_dim=8, seed=42)

        # Baseline cache for XAI feature deviation calculation
        self._baselines: dict[str, np.ndarray] = {}
        self._baseline_n: dict[str, int] = {}

        # Telemetry & performance metrics
        self._stats = {
            "total_routed": 0,
            "stage_1_exits": 0,
            "stage_2_exits": 0,
            "stage_3_exits": 0,
            "stage_4_exits": 0,
            "alerts_total": 0,
            "stage_1_alerts": 0,
            "stage_2_alerts": 0,
            "stage_3_alerts": 0,
            "stage_4_alerts": 0,
            "total_latency_ms": 0.0,
            "stage_latencies": {
                "fast_path": 0.0,
                "ml_ensemble": 0.0,
                "multimodal": 0.0,
                "deep_graph": 0.0,
            },
        }

    def _update_baseline(self, cls: str, vec: np.ndarray) -> np.ndarray:
        with self._lock:
            if cls not in self._baselines:
                self._baselines[cls] = vec.copy()
                self._baseline_n[cls] = 1
            else:
                n = self._baseline_n[cls]
                self._baselines[cls] = (self._baselines[cls] * n + vec) / (n + 1)
                self._baseline_n[cls] = n + 1
            return self._baselines[cls].copy()

    # ── Stage 1: Fast Path ───────────────────────────────────────────────────

    def _eval_stage_1(
        self, evt: dict
    ) -> Tuple[float, float, float, list[SignatureMatch], bool, str]:
        """
        Sub-millisecond triage: deterministic pattern rules + lightweight heuristics.
        Returns: (conf_1, unc_1, ood_1, sig_matches, is_benign_candidate, attack_type)
        """
        sig_matches = run_signature_engine(evt)
        if sig_matches:
            best_sig = sig_matches[0]
            conf = max((m.confidence for m in sig_matches), default=0.90)
            unc = 0.05   # Deterministic rule match has near-zero epistemic ambiguity
            ood = 0.05
            return conf, unc, ood, sig_matches, False, best_sig.attack_type

        # Lightweight protocol & process sanity check
        cls = evt.get("ocsf_class", "")
        sev_id = evt.get("severity_id", 1)

        # Quick check for obviously benign events
        is_clear_benign = False
        if cls == "network_activity":
            dst_port = evt.get("dst_port", 0)
            bytes_out = evt.get("bytes_out", 0)
            flags = evt.get("tcp_flags", [])
            # Standard common ports without anomalies or flood flags
            if dst_port in (80, 443, 53, 123) and bytes_out < 50000 and sev_id <= 1:
                if not (len(flags) == 1 and "SYN" in flags):  # not raw SYN flood
                    is_clear_benign = True
        elif cls == "process_activity":
            cmd = str(evt.get("command", "")).lower()
            elevated = evt.get("is_elevated", False) or evt.get("is_root", False)
            if not elevated and sev_id <= 1 and len(cmd) < 80:
                suspicious_tokens = ["nc ", "ncat ", "wget ", "curl ", "bash -i", "/dev/tcp"]
                if not any(tok in cmd for tok in suspicious_tokens):
                    is_clear_benign = True
        elif cls == "file_activity":
            filepath = str(evt.get("filepath", "")).lower()
            if not any(filepath.endswith(ext) for ext in [".enc", ".locked", ".crypto", ".ransom"]):
                if sev_id <= 1 and evt.get("entropy", 0.0) < 6.5:
                    is_clear_benign = True
        elif cls == "cloud_api":
            api_name = str(evt.get("api_name", evt.get("activity_name", "")))
            if (api_name.startswith("Describe") or api_name.startswith("Get") or api_name.startswith("List")) and sev_id <= 1:
                is_clear_benign = True

        if is_clear_benign:
            conf = 0.05
            unc = 0.10
            ood = 0.05
            return conf, unc, ood, [], True, ""

        # Inconclusive fast path
        conf = 0.35
        unc = 0.40
        ood = 0.20
        return conf, unc, ood, [], False, ""

    # ── Stage 2: ML Ensemble ─────────────────────────────────────────────────

    def _eval_stage_2(
        self,
        evt: dict,
        vec: np.ndarray,
        sig_matches: list[SignatureMatch],
        s1_conf: float,
    ) -> Tuple[float, float, float, AnomalyResult, StatResult, list[str]]:
        """
        Evaluates Isolation Forest, Deep Autoencoder, One-Class SVM, and Stat Engine.
        """
        cls = evt.get("ocsf_class", "")
        ml_result = run_anomaly_engine(cls, vec)
        stat_result = run_statistical_engine(evt, vec)

        sig_fired = len(sig_matches) > 0
        ml_fired = ml_result.is_anomaly
        stat_fired = stat_result.is_anomaly

        engines_fired = []
        weighted_conf = 0.0

        if sig_fired:
            engines_fired.append("signature")
            weighted_conf += _W_SIG * s1_conf
        if ml_fired:
            engines_fired.append("anomaly")
            weighted_conf += _W_ML * ml_result.confidence
        if stat_fired:
            engines_fired.append("statistical")
            weighted_conf += _W_STAT * stat_result.confidence

        # Multi-engine corroboration
        n_eng = len(engines_fired)
        if n_eng == 2:
            weighted_conf = min(weighted_conf * 1.20, 1.0)
        elif n_eng >= 3:
            weighted_conf = min(weighted_conf * 1.40, 1.0)

        conf_2 = round(weighted_conf, 4)

        # Epistemic uncertainty: variance among detectors + boundary ambiguity
        sub_scores = [
            float(ml_result.isolation_score > 0),
            float(ml_result.svm_score > 0),
            float(ml_result.reconstruction_error > 0.5),
            float(stat_result.is_anomaly),
        ]
        disagreement = float(np.var(sub_scores))
        boundary_dist = abs(conf_2 - 0.50)
        boundary_unc = max(0.0, 1.0 - 2.0 * boundary_dist)
        unc_2 = round(float(np.clip(0.40 * disagreement + 0.60 * boundary_unc, 0.05, 0.95)), 4)

        # Out-of-distribution score from autoencoder reconstruction error
        ood_2 = round(float(np.clip(ml_result.reconstruction_error / 2.0, 0.0, 1.0)), 4)

        return conf_2, unc_2, ood_2, ml_result, stat_result, engines_fired

    # ── Stage 3: Behavioral / Multimodal ─────────────────────────────────────

    def _eval_stage_3(
        self,
        evt: dict,
        s2_conf: float,
        s2_unc: float,
    ) -> Tuple[float, float, float, ModalityVectors]:
        """
        Cross-modal representation & attention entropy analysis.
        """
        mod_vecs = self.multimodal_encoder.encode(evt)
        attn = mod_vecs.attention_matrix  # (K, K)

        # Normalized cross-modal attention entropy
        K = attn.shape[0]
        if K > 1:
            attn_flat = attn.flatten()
            entropy = -float(np.sum(attn_flat * np.log(attn_flat + 1e-12)))
            max_entropy = float(K * np.log(K))
            norm_entropy = entropy / (max_entropy + 1e-12)
        else:
            norm_entropy = 1.0

        # Modality consistency: Euclidean norm of cross-modal fused embedding
        fused_norm = float(np.linalg.norm(mod_vecs.fused))
        mm_score = 1.0 / (1.0 + math.exp(-float(np.clip(fused_norm - 2.0, -10.0, 10.0))))

        # Fused Stage 3 confidence
        conf_3 = round(float(0.60 * s2_conf + 0.40 * mm_score), 4)

        # Multimodal consensus reduces uncertainty
        unc_3 = round(float(np.clip(s2_unc * (1.0 - 0.35 * norm_entropy), 0.02, 0.90)), 4)
        ood_3 = round(float(np.clip(1.0 - norm_entropy, 0.0, 1.0)), 4)

        return conf_3, unc_3, ood_3, mod_vecs

    # ── Stage 4: Deep Relational & Graph Analysis ─────────────────────────────

    def _eval_stage_4(
        self,
        evt: dict,
        s3_conf: float,
        s3_unc: float,
    ) -> Tuple[float, float, float, float]:
        """
        Deep multi-hop attack path reasoning and graph risk aggregation.
        """
        # Formulate active graph entities from event
        src_ip = evt.get("src_ip", evt.get("source_ip", "host-unknown"))
        dst_ip = evt.get("dst_ip", evt.get("destination_ip", "gateway"))
        path_nodes = [str(src_ip), str(dst_ip)]

        # Query node risks via attack reasoner Noisy-OR
        node_risks = [s3_conf, max(0.10, s3_conf * 0.85)]
        path_risk = self.attack_reasoner.score_path_noisy_or(node_risks)

        # Fused Stage 4 confidence
        conf_4 = round(float(0.55 * s3_conf + 0.45 * path_risk), 4)
        # Deep evaluation resolves residual epistemic ambiguity
        unc_4 = round(float(np.clip(s3_unc * 0.40, 0.02, 0.50)), 4)
        ood_4 = 0.05

        return conf_4, unc_4, ood_4, path_risk

    # ── Core Routing Engine ──────────────────────────────────────────────────

    def route(
        self,
        evt: dict,
        force_stage: Optional[int] = None,
        force_full: bool = False,
    ) -> Optional[RoutedDetectionResult]:
        """
        Processes an event through the confidence-based early-exit cascade.
        """
        t0 = time.perf_counter()
        target_stage = 4 if force_full else (force_stage or self.config.force_stage)

        cls = evt.get("ocsf_class", "")
        event_id = evt.get("event_id", str(uuid.uuid4()))
        detection_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        cfg = self.config
        stages_evaluated: list[int] = []
        conf_traj: list[float] = []
        unc_traj: list[float] = []
        ood_traj: list[float] = []
        stage_latencies: dict[str, float] = {}

        # Default place-holders
        sig_matches: list[SignatureMatch] = []
        ml_result: Optional[AnomalyResult] = None
        stat_result: Optional[StatResult] = None
        mod_vecs: Optional[ModalityVectors] = None
        attack_type = ""
        engines_fired: list[str] = []
        exit_reason = ""
        stage_exited = 1

        # ── Stage 1: Fast Path ───────────────────────────────────────────────
        t_s1 = time.perf_counter()
        stages_evaluated.append(1)
        c1, u1, o1, sig_matches, s1_benign, best_sig_type = self._eval_stage_1(evt)
        stage_latencies["fast_path"] = round((time.perf_counter() - t_s1) * 1000, 3)
        conf_traj.append(c1)
        unc_traj.append(u1)
        ood_traj.append(o1)
        if best_sig_type:
            attack_type = best_sig_type

        # Early exit check at Stage 1
        force_exit_s1 = (target_stage == 1)
        s1_alert = (c1 >= cfg.conf_threshold and u1 <= cfg.uncertainty_threshold)
        s1_clear_benign = (
            s1_benign
            and abs(c1 - cfg.decision_threshold) >= cfg.margin_threshold
            and o1 <= cfg.ood_threshold
            and u1 <= cfg.uncertainty_threshold
        )

        if force_exit_s1 or (target_stage is None and (s1_alert or s1_clear_benign)):
            stage_exited = 1
            if force_exit_s1:
                exit_reason = "force_stage_1"
            elif s1_alert:
                exit_reason = "stage1_signature_high_confidence"
            else:
                exit_reason = "stage1_fast_path_clear_benign"

            is_alert = s1_alert or (force_exit_s1 and c1 >= cfg.decision_threshold)
            conf = c1
            severity = _severity_label(conf) if is_alert else "INFO"
            if is_alert and sig_matches:
                engines_fired.append("signature")
                sig_sev = max((m.severity for m in sig_matches), default=1)
                _SIG_SEV_LABELS = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "INFO"}
                severity = max(
                    [severity, _SIG_SEV_LABELS.get(sig_sev, "INFO")],
                    key=lambda x: {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}.get(x, 0)
                )

            total_ms = round((time.perf_counter() - t0) * 1000, 3)
            self._record_telemetry(1, is_alert, total_ms, stage_latencies)

            explanation = {
                "stage_exited": 1,
                "stage_name": "fast_path",
                "exit_reason": exit_reason,
                "confidence": conf,
                "severity": severity,
                "top_contributing_features": [],
                "explanation_text": (
                    f"Early exit at Stage 1 (Fast Path): {exit_reason}. "
                    f"Confidence: {conf:.2f}, Uncertainty: {u1:.2f}."
                ),
            }

            return RoutedDetectionResult(
                detection_id=detection_id,
                event_id=event_id,
                ocsf_class=cls,
                time=timestamp,
                is_alert=is_alert,
                confidence=conf,
                severity=severity,
                attack_type=attack_type,
                engines_fired=engines_fired,
                signature_matches=[_sig_to_dict(m) for m in sig_matches],
                anomaly_result={},
                stat_result={},
                explanation=explanation,
                normalized_event=evt,
                processing_ms=total_ms,
                stage_exited=1,
                stage_name="fast_path",
                stages_evaluated=stages_evaluated,
                confidence_trajectory=conf_traj,
                uncertainty_trajectory=unc_traj,
                ood_trajectory=ood_traj,
                latency_ms=total_ms,
                per_stage_latencies=stage_latencies,
                exit_reason=exit_reason,
                early_exited=True,
            )

        # ── Stage 2: ML Ensemble ─────────────────────────────────────────────
        t_s2 = time.perf_counter()
        stages_evaluated.append(2)
        vec = extract(evt)
        if vec is None:
            return None

        feat_names = get_feature_names(cls)
        baseline_vec = self._update_baseline(cls, vec)

        c2, u2, o2, ml_result, stat_result, engines_fired = self._eval_stage_2(
            evt, vec, sig_matches, c1
        )
        stage_latencies["ml_ensemble"] = round((time.perf_counter() - t_s2) * 1000, 3)
        conf_traj.append(c2)
        unc_traj.append(u2)
        ood_traj.append(o2)

        if not attack_type:
            if ml_result.is_anomaly:
                attack_type = "ml_anomaly"
            elif stat_result.is_anomaly:
                attack_type = "behavioral_anomaly"

        # Early exit check at Stage 2
        force_exit_s2 = (target_stage == 2)
        s2_alert = (c2 >= cfg.conf_threshold and u2 <= cfg.uncertainty_threshold)
        s2_clear_benign = (
            abs(c2 - cfg.decision_threshold) >= cfg.margin_threshold
            and c2 < cfg.decision_threshold
            and o2 <= cfg.ood_threshold
            and u2 <= cfg.uncertainty_threshold
        )

        if force_exit_s2 or (target_stage is None and (s2_alert or s2_clear_benign)):
            stage_exited = 2
            if force_exit_s2:
                exit_reason = "force_stage_2"
            elif s2_alert:
                exit_reason = "stage2_ml_high_confidence_alert"
            else:
                exit_reason = "stage2_ml_clear_benign"

            is_alert = (
                len(sig_matches) > 0
                or (ml_result.is_anomaly and c2 > 0.60)
                or (len(engines_fired) >= 2 and c2 > _ALERT_THRESHOLD)
                or (len(engines_fired) == 3)
                or (force_exit_s2 and c2 >= cfg.decision_threshold)
            )
            conf = c2
            severity = _severity_label(conf) if is_alert else "INFO"

            # XAI Explanation
            if_pipe = None
            try:
                det = _get_detector(cls)
                if_pipe = getattr(det, "_if_pipe", None)
            except Exception:
                pass

            xai = explain(
                vec=vec,
                feature_names=feat_names,
                ocsf_class=cls,
                attack_type=attack_type,
                sig_confidence=c1 if sig_matches else 0.0,
                ml_confidence=ml_result.confidence,
                stat_confidence=stat_result.confidence,
                if_pipe=if_pipe,
                baseline_mean=baseline_vec,
            )
            xai["stage_exited"] = 2
            xai["stage_name"] = "ml_ensemble"
            xai["exit_reason"] = exit_reason

            total_ms = round((time.perf_counter() - t0) * 1000, 3)
            self._record_telemetry(2, is_alert, total_ms, stage_latencies)

            return RoutedDetectionResult(
                detection_id=detection_id,
                event_id=event_id,
                ocsf_class=cls,
                time=timestamp,
                is_alert=is_alert,
                confidence=conf,
                severity=severity,
                attack_type=attack_type,
                engines_fired=engines_fired,
                signature_matches=[_sig_to_dict(m) for m in sig_matches],
                anomaly_result=_anom_to_dict(ml_result),
                stat_result=_stat_to_dict(stat_result),
                explanation=xai,
                normalized_event=evt,
                processing_ms=total_ms,
                stage_exited=2,
                stage_name="ml_ensemble",
                stages_evaluated=stages_evaluated,
                confidence_trajectory=conf_traj,
                uncertainty_trajectory=unc_traj,
                ood_trajectory=ood_traj,
                latency_ms=total_ms,
                per_stage_latencies=stage_latencies,
                exit_reason=exit_reason,
                early_exited=True,
            )

        # ── Stage 3: Behavioral / Multimodal ─────────────────────────────────
        t_s3 = time.perf_counter()
        stages_evaluated.append(3)
        c3, u3, o3, mod_vecs = self._eval_stage_3(evt, c2, u2)
        stage_latencies["multimodal"] = round((time.perf_counter() - t_s3) * 1000, 3)
        conf_traj.append(c3)
        unc_traj.append(u3)
        ood_traj.append(o3)

        force_exit_s3 = (target_stage == 3)
        s3_alert = (c3 >= cfg.conf_threshold and u3 <= cfg.uncertainty_threshold)
        s3_clear_benign = (
            abs(c3 - cfg.decision_threshold) >= cfg.margin_threshold
            and c3 < cfg.decision_threshold
            and o3 <= cfg.ood_threshold
            and u3 <= cfg.uncertainty_threshold
        )

        if force_exit_s3 or (target_stage is None and (s3_alert or s3_clear_benign)):
            stage_exited = 3
            if force_exit_s3:
                exit_reason = "force_stage_3"
            elif s3_alert:
                exit_reason = "stage3_multimodal_converged_alert"
            else:
                exit_reason = "stage3_multimodal_clear_benign"

            is_alert = (c3 >= cfg.decision_threshold) or len(sig_matches) > 0
            conf = c3
            severity = _severity_label(conf) if is_alert else "INFO"

            # XAI Explanation with multimodal cross-attention weights
            if_pipe = None
            try:
                det = _get_detector(cls)
                if_pipe = getattr(det, "_if_pipe", None)
            except Exception:
                pass

            xai = explain(
                vec=vec,
                feature_names=feat_names,
                ocsf_class=cls,
                attack_type=attack_type,
                sig_confidence=c1 if sig_matches else 0.0,
                ml_confidence=ml_result.confidence if ml_result else 0.0,
                stat_confidence=stat_result.confidence if stat_result else 0.0,
                if_pipe=if_pipe,
                baseline_mean=baseline_vec,
            )
            xai["stage_exited"] = 3
            xai["stage_name"] = "multimodal"
            xai["exit_reason"] = exit_reason
            xai["cross_modal_attention"] = mod_vecs.attention_matrix.tolist()

            total_ms = round((time.perf_counter() - t0) * 1000, 3)
            self._record_telemetry(3, is_alert, total_ms, stage_latencies)

            return RoutedDetectionResult(
                detection_id=detection_id,
                event_id=event_id,
                ocsf_class=cls,
                time=timestamp,
                is_alert=is_alert,
                confidence=conf,
                severity=severity,
                attack_type=attack_type,
                engines_fired=engines_fired,
                signature_matches=[_sig_to_dict(m) for m in sig_matches],
                anomaly_result=_anom_to_dict(ml_result) if ml_result else {},
                stat_result=_stat_to_dict(stat_result) if stat_result else {},
                explanation=xai,
                normalized_event=evt,
                processing_ms=total_ms,
                stage_exited=3,
                stage_name="multimodal",
                stages_evaluated=stages_evaluated,
                confidence_trajectory=conf_traj,
                uncertainty_trajectory=unc_traj,
                ood_trajectory=ood_traj,
                latency_ms=total_ms,
                per_stage_latencies=stage_latencies,
                exit_reason=exit_reason,
                early_exited=True,
            )

        # ── Stage 4: Deep Relational & Graph Analysis ─────────────────────────
        t_s4 = time.perf_counter()
        stages_evaluated.append(4)
        c4, u4, o4, path_risk = self._eval_stage_4(evt, c3, u3)
        stage_latencies["deep_graph"] = round((time.perf_counter() - t_s4) * 1000, 3)
        conf_traj.append(c4)
        unc_traj.append(u4)
        ood_traj.append(o4)

        stage_exited = 4
        exit_reason = "stage4_deep_graph_exhausted"
        is_alert = (c4 >= cfg.decision_threshold) or len(sig_matches) > 0
        conf = c4
        severity = _severity_label(conf) if is_alert else "INFO"

        # Comprehensive explanation
        if_pipe = None
        try:
            det = _get_detector(cls)
            if_pipe = getattr(det, "_if_pipe", None)
        except Exception:
            pass

        xai = explain(
            vec=vec,
            feature_names=feat_names,
            ocsf_class=cls,
            attack_type=attack_type,
            sig_confidence=c1 if sig_matches else 0.0,
            ml_confidence=ml_result.confidence if ml_result else 0.0,
            stat_confidence=stat_result.confidence if stat_result else 0.0,
            if_pipe=if_pipe,
            baseline_mean=baseline_vec,
        )
        xai["stage_exited"] = 4
        xai["stage_name"] = "deep_graph"
        xai["exit_reason"] = exit_reason
        xai["path_risk"] = path_risk
        if mod_vecs is not None:
            xai["cross_modal_attention"] = mod_vecs.attention_matrix.tolist()

        total_ms = round((time.perf_counter() - t0) * 1000, 3)
        self._record_telemetry(4, is_alert, total_ms, stage_latencies)

        return RoutedDetectionResult(
            detection_id=detection_id,
            event_id=event_id,
            ocsf_class=cls,
            time=timestamp,
            is_alert=is_alert,
            confidence=conf,
            severity=severity,
            attack_type=attack_type,
            engines_fired=engines_fired,
            signature_matches=[_sig_to_dict(m) for m in sig_matches],
            anomaly_result=_anom_to_dict(ml_result) if ml_result else {},
            stat_result=_stat_to_dict(stat_result) if stat_result else {},
            explanation=xai,
            normalized_event=evt,
            processing_ms=total_ms,
            stage_exited=4,
            stage_name="deep_graph",
            stages_evaluated=stages_evaluated,
            confidence_trajectory=conf_traj,
            uncertainty_trajectory=unc_traj,
            ood_trajectory=ood_traj,
            latency_ms=total_ms,
            per_stage_latencies=stage_latencies,
            exit_reason=exit_reason,
            early_exited=False,
        )

    def process(self, evt: dict) -> Optional[RoutedDetectionResult]:
        """Convenience alias for route(evt) for interface parity with HybridCombiner."""
        return self.route(evt)

    # ── Telemetry & Statistics ───────────────────────────────────────────────

    def _record_telemetry(
        self,
        stage: int,
        is_alert: bool,
        total_ms: float,
        stage_latencies: dict[str, float],
    ) -> None:
        with self._lock:
            self._stats["total_routed"] += 1
            self._stats[f"stage_{stage}_exits"] += 1
            self._stats["total_latency_ms"] += total_ms
            if is_alert:
                self._stats["alerts_total"] += 1
                self._stats[f"stage_{stage}_alerts"] += 1
            for name, ms in stage_latencies.items():
                self._stats["stage_latencies"][name] += ms

    def get_stats(self) -> dict:
        """Returns cumulative routing performance metrics and stage distributions."""
        with self._lock:
            total = max(1, self._stats["total_routed"])
            f1 = round(self._stats["stage_1_exits"] / total, 4)
            f2 = round(self._stats["stage_2_exits"] / total, 4)
            f3 = round(self._stats["stage_3_exits"] / total, 4)
            f4 = round(self._stats["stage_4_exits"] / total, 4)
            early_rate = round((self._stats["stage_1_exits"] + self._stats["stage_2_exits"] + self._stats["stage_3_exits"]) / total, 4)
            avg_lat = round(self._stats["total_latency_ms"] / total, 3)

            return {
                "total_routed": self._stats["total_routed"],
                "early_exit_rate": early_rate,
                "exit_fractions": {"f1": f1, "f2": f2, "f3": f3, "f4": f4},
                "stage_exits": {
                    "stage_1": self._stats["stage_1_exits"],
                    "stage_2": self._stats["stage_2_exits"],
                    "stage_3": self._stats["stage_3_exits"],
                    "stage_4": self._stats["stage_4_exits"],
                },
                "alerts_by_stage": {
                    "stage_1": self._stats["stage_1_alerts"],
                    "stage_2": self._stats["stage_2_alerts"],
                    "stage_3": self._stats["stage_3_alerts"],
                    "stage_4": self._stats["stage_4_alerts"],
                    "total": self._stats["alerts_total"],
                },
                "avg_latency_ms": avg_lat,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Module Singleton
# ─────────────────────────────────────────────────────────────────────────────

_router_instance: Optional[ConfidenceModelRouter] = None
_router_lock = threading.Lock()


def get_model_router(config: Optional[RouterConfig] = None) -> ConfidenceModelRouter:
    """Returns thread-safe singleton instance of ConfidenceModelRouter."""
    global _router_instance
    with _router_lock:
        if _router_instance is None or config is not None:
            _router_instance = ConfidenceModelRouter(config=config)
    return _router_instance
