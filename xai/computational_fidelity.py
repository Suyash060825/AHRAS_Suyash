from __future__ import annotations
"""
AHRAS Module — Computational Fidelity & Multi-Path Sum-Check Engine
-------------------------------------------------------------------
Implements Stage 2 of the AHRAS Research Platform:
  1. Per-event error metrics:
       E_abs = |R_engine - R_reconstructed|
       E_rel = |R_engine - R_reconstructed| / max(|R_engine|, epsilon)
  2. Aggregate distributional statistics:
       MAE, Median Absolute Error, P95 Error, Max Error, Pass Rate, Failure Rate.
  3. Separate 10-path subsystem evaluations:
       Path 1:  Base Scoring (Signature + ML baseline)
       Path 2:  Adaptive Weighting (Evidence Quality Modifiers Q_i)
       Path 3:  Multipliers (Signal Multiplier (1 + ΔD) & Uncertainty Attenuation)
       Path 4:  Rule Boosts (Signature Severity Scales 1-5 & MITRE)
       Path 5:  Contextual Evidence (Forecast Early-Warning & Threat Intel)
       Path 6:  Historical Evidence (Recidivism Boost & Dynamic Trust)
       Path 7:  Graph Evidence (TGNN Corroboration & Attack Episode Risk)
       Path 8:  Asset / Trust / MITRE (Criticality Multipliers & Access Trust)
       Path 9:  Clipping (Sub-zero floors and upper saturation boundaries)
       Path 10: Complete Score Path (Full end-to-end multi-signal fusion)
"""

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from detection.risk_engine import (
        AdaptiveRiskEngine,
        RiskConfig,
        DecisionTrace,
    )

DEFAULT_EPSILON = 1e-6
DEFAULT_TOLERANCE = 1e-4


@dataclass
class EventFidelityEvaluation:
    event_id:             str
    path_name:            str
    engine_score:         float
    reconstructed_score:  float
    e_abs:                float          # |R_engine - R_reconstructed|
    e_rel:                float          # |R_engine - R_reconstructed| / max(|R_engine|, epsilon)
    tolerance:            float
    passed:               bool           # e_abs <= tolerance
    step_details:         Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FidelityMetrics:
    path_name:            str
    description:          str
    n_samples:            int
    mae:                  float
    median_error:         float
    p95_error:            float
    max_error:            float
    mean_relative_error:  float
    pass_rate:            float          # Fraction of samples where e_abs <= tolerance
    failure_rate:         float          # 1.0 - pass_rate
    tolerance:            float
    is_exact:             bool           # pass_rate == 1.0 under tolerance

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComprehensiveFidelityReport:
    timestamp:            float
    tolerance:            float
    epsilon:              float
    all_paths_passed:     bool
    path_results:         Dict[str, FidelityMetrics]
    sample_evaluations:   Dict[str, List[EventFidelityEvaluation]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "tolerance":        self.tolerance,
            "epsilon":          self.epsilon,
            "all_paths_passed": self.all_paths_passed,
            "path_results":     {k: v.to_dict() for k, v in self.path_results.items()},
            "summary_stats": {
                "total_paths": len(self.path_results),
                "passed_paths": sum(1 for v in self.path_results.values() if v.is_exact),
                "max_observed_error": max(v.max_error for v in self.path_results.values()) if self.path_results else 0.0,
                "overall_mae": float(np.mean([v.mae for v in self.path_results.values()])) if self.path_results else 0.0,
            }
        }


def compute_fidelity_metrics(
    evaluations: List[EventFidelityEvaluation],
    path_name: str,
    description: str,
    tolerance: float,
) -> FidelityMetrics:
    """Computes rigorous distributional error statistics across a collection of evaluations."""
    if not evaluations:
        return FidelityMetrics(
            path_name=path_name,
            description=description,
            n_samples=0,
            mae=0.0,
            median_error=0.0,
            p95_error=0.0,
            max_error=0.0,
            mean_relative_error=0.0,
            pass_rate=1.0,
            failure_rate=0.0,
            tolerance=tolerance,
            is_exact=True,
        )

    abs_errors = np.array([ev.e_abs for ev in evaluations], dtype=np.float64)
    rel_errors = np.array([ev.e_rel for ev in evaluations], dtype=np.float64)
    n = len(evaluations)

    mae = float(np.mean(abs_errors))
    median_err = float(np.median(abs_errors))
    p95_err = float(np.percentile(abs_errors, 95))
    max_err = float(np.max(abs_errors))
    mean_rel_err = float(np.mean(rel_errors))

    passed_count = sum(1 for ev in evaluations if ev.passed)
    pass_rate = float(passed_count / n)
    failure_rate = float(1.0 - pass_rate)
    is_exact = bool(passed_count == n)

    return FidelityMetrics(
        path_name=path_name,
        description=description,
        n_samples=n,
        mae=round(mae, 8),
        median_error=round(median_err, 8),
        p95_error=round(p95_err, 8),
        max_error=round(max_err, 8),
        mean_relative_error=round(mean_rel_err, 8),
        pass_rate=round(pass_rate, 4),
        failure_rate=round(failure_rate, 4),
        tolerance=tolerance,
        is_exact=is_exact,
    )


class ComputationalFidelityEvaluator:
    """
    Executes controlled computational sum-checks across 10 separate scoring paths
    to scientifically verify exact mathematical reconstructibility.
    """

    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        epsilon: float = DEFAULT_EPSILON,
        seed: int = 42,
    ):
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.seed = seed
        from detection.risk_engine import AdaptiveRiskEngine, RiskConfig, replay_decision_trace
        self.engine = AdaptiveRiskEngine()
        self._RiskConfig = RiskConfig
        self._replay_decision_trace = replay_decision_trace

    def _eval_single(
        self,
        event_id: str,
        path_name: str,
        engine_score: float,
        reconstructed_score: float,
        step_details: Optional[Dict[str, float]] = None,
    ) -> EventFidelityEvaluation:
        e_abs = abs(engine_score - reconstructed_score)
        e_rel = e_abs / max(abs(engine_score), self.epsilon)
        passed = (e_abs <= self.tolerance)

        return EventFidelityEvaluation(
            event_id=event_id,
            path_name=path_name,
            engine_score=round(engine_score, 4),
            reconstructed_score=round(reconstructed_score, 4),
            e_abs=round(e_abs, 8),
            e_rel=round(e_rel, 8),
            tolerance=self.tolerance,
            passed=passed,
            step_details=step_details or {},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Path 1: Base Scoring
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_base_scoring(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 1)
        cfg = self._RiskConfig(
            use_signature=True,
            use_ml=True,
            use_statistical=False,
            use_trust=False,
            use_history=False,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            s_val = rng.uniform(0.0, 1.0)
            a_val = rng.uniform(0.0, 1.0)
            sig = [type("Sig", (), {"severity": int(s_val * 5) or 1, "confidence": 0.95, "rule_name": f"r_{i}", "mitre_technique": "T1046"})()]
            ml = type("ML", (), {"ensemble_score": a_val, "confidence": 0.85})()

            res = self.engine.score_risk(f"base_host_{i}", sig, ml, None, override_config=cfg)
            engine_score = res.risk_score

            # Analytical reconstruction of base scoring path
            # S_sig = min(1.0, severity / 5.0), A_ml = a_val
            raw_sig = min(1.0, (int(s_val * 5) or 1) / 5.0)
            reconstructed = round(min(1.0, max(0.0, (cfg.w_sig * raw_sig) + (cfg.w_ml * a_val))), 4)

            evals.append(self._eval_single(
                event_id=f"BASE-{i}",
                path_name="base_scoring",
                engine_score=engine_score,
                reconstructed_score=reconstructed,
                step_details={"w_sig": cfg.w_sig, "S_sig": raw_sig, "w_ml": cfg.w_ml, "A_ml": a_val},
            ))

        metrics = compute_fidelity_metrics(evals, "base_scoring", "Signature + ML anomaly unmodulated linear combination", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 2: Adaptive Weighting
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_adaptive_weighting(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 2)
        cfg = self._RiskConfig(
            use_signature=True,
            use_ml=True,
            use_statistical=False,
            use_trust=False,
            use_history=False,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=True,
        )
        evals = []
        for i in range(n):
            sig_sev = rng.randint(1, 5)
            a_val = rng.uniform(0.0, 1.0)
            sig = [type("Sig", (), {"severity": sig_sev, "confidence": 0.95, "rule_name": f"r_{i}", "mitre_technique": "T1046"})()]
            ml = type("ML", (), {"ensemble_score": a_val, "confidence": 0.85})()

            res = self.engine.score_risk(f"adapt_host_{i}", sig, ml, None, override_config=cfg)
            trace = res.trace

            # Analytical reconstruction using adaptive quality weights
            q_sig = trace.adaptive_weights.get("q_sig", 1.0)
            q_ml = trace.adaptive_weights.get("q_ml", 1.0)
            raw_sig = min(1.0, sig_sev / 5.0)
            term_sig = cfg.w_sig * raw_sig * q_sig
            term_ml = cfg.w_ml * a_val * q_ml
            reconstructed = round(float(np.clip(term_sig + term_ml, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"ADAPT-{i}",
                path_name="adaptive_weighting",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"q_sig": q_sig, "q_ml": q_ml, "term_sig": term_sig, "term_ml": term_ml},
            ))

        metrics = compute_fidelity_metrics(evals, "adaptive_weighting", "Dynamic evidence quality weighting (Q_i)", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 3: Multipliers (Signal & Uncertainty Attenuation)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_multipliers(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 3)
        cfg = self._RiskConfig(
            use_signature=False,
            use_ml=True,
            use_statistical=True,
            use_trust=False,
            use_history=False,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=True,
            use_asset_crit=False,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            a_val = rng.uniform(0.1, 0.9)
            delta_d = rng.uniform(0.1, 2.5)
            ml = type("ML", (), {"ensemble_score": a_val, "confidence": 0.85})()
            stat = type("Stat", (), {"behavioral_drift": delta_d, "confidence": 0.80, "flags": [], "mitre_techniques": []})()

            res = self.engine.score_risk(f"mult_host_{i}", [], ml, stat, override_config=cfg)
            trace = res.trace

            # Analytical reconstruction: term_ml = w_ml * A_ml * (1 + delta_D), scaled by (1 - u_penalty)
            u_pen = trace.multiplicative_factors["uncertainty_penalty"]
            sig_mult = 1.0 + delta_d
            term_ml = cfg.w_ml * a_val * sig_mult
            reconstructed = round(float(np.clip(term_ml * (1.0 - u_pen), 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"MULT-{i}",
                path_name="multipliers",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"signal_multiplier": sig_mult, "u_penalty": u_pen, "term_ml": term_ml},
            ))

        metrics = compute_fidelity_metrics(evals, "multipliers", "Signal multiplier (1 + ΔD) and uncertainty attenuation (1 - U_pen)", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 4: Rule Boosts (Signature Severity Scales 1-5 & MITRE)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_rule_boosts(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 4)
        cfg = self._RiskConfig(
            use_signature=True,
            use_ml=False,
            use_statistical=False,
            use_trust=False,
            use_history=False,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            sev = rng.randint(1, 5)
            rule_name = f"RULE_CVE_{2024 + (i % 3)}_{1000 + i}"
            sig = [type("Sig", (), {"severity": sev, "confidence": 0.95, "rule_name": rule_name, "mitre_technique": "T1190"})()]

            res = self.engine.score_risk(f"rule_host_{i}", sig, None, None, override_config=cfg)
            expected_raw = min(1.0, sev / 5.0)
            reconstructed = round(float(np.clip(cfg.w_sig * expected_raw, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"RULE-{i}",
                path_name="rule_boosts",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"severity": sev, "raw_sig": expected_raw, "rule_contribution": reconstructed},
            ))

        metrics = compute_fidelity_metrics(evals, "rule_boosts", "Discrete signature rule severities and mapping", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 5: Contextual Evidence (Forecast Early Warning & Threat Intel)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_contextual_evidence(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 5)
        cfg = self._RiskConfig(
            use_signature=False,
            use_ml=False,
            use_statistical=False,
            use_trust=False,
            use_history=False,
            use_graph=False,
            use_forecast=True,
            use_ti=True,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            p_fore = rng.uniform(0.0, 1.0)
            ti_score = rng.uniform(0.0, 1.0)

            res = self.engine.score_risk(f"ctx_host_{i}", [], None, None, p_fore=p_fore, ti_score=ti_score, override_config=cfg)
            term_fore = cfg.w_fore * p_fore
            term_ti = cfg.w_ti * ti_score
            reconstructed = round(float(np.clip(term_fore + term_ti, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"CTX-{i}",
                path_name="contextual_evidence",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"p_fore": p_fore, "term_fore": term_fore, "ti_score": ti_score, "term_ti": term_ti},
            ))

        metrics = compute_fidelity_metrics(evals, "contextual_evidence", "Early warning prediction and threat intelligence feeds", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 6: Historical Evidence (Recidivism Boost & Dynamic Trust)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_historical_evidence(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 6)
        cfg = self._RiskConfig(
            use_signature=False,
            use_ml=False,
            use_statistical=False,
            use_trust=True,
            use_history=True,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            h_boost = rng.uniform(0.1, 0.8)
            trust_val = rng.uniform(0.0, 1.0)
            entity_key = f"hist_host_{i}"
            self.engine.set_trust(entity_key, trust_val)

            res = self.engine.score_risk(entity_key, [], None, None, h_boost=h_boost, override_config=cfg)
            term_hist = cfg.w_hist * h_boost
            trust_sub = cfg.w_trust * trust_val
            reconstructed = round(float(np.clip(term_hist - trust_sub, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"HIST-{i}",
                path_name="historical_evidence",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"h_boost": h_boost, "term_hist": term_hist, "trust_val": trust_val, "trust_sub": trust_sub},
            ))

        metrics = compute_fidelity_metrics(evals, "historical_evidence", "Historical recidivism boost and dynamic entity trust", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 7: Graph Evidence (TGNN Corroboration & Attack Episode Risk)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_graph_evidence(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 7)
        cfg = self._RiskConfig(
            use_signature=False,
            use_ml=False,
            use_statistical=False,
            use_trust=False,
            use_history=False,
            use_graph=True,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=False,
            use_evidence_quality=False,
            use_episode_reasoning=True,
        )
        evals = []
        for i in range(n):
            g_corr = rng.uniform(0.0, 1.0)
            r_ep = rng.uniform(0.0, 1.0)

            res = self.engine.score_risk(f"graph_host_{i}", [], None, None, g_corr=g_corr, r_ep=r_ep, override_config=cfg)
            term_graph = cfg.w_graph * g_corr
            term_ep = cfg.w_ep * r_ep
            reconstructed = round(float(np.clip(term_graph + term_ep, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"GRAPH-{i}",
                path_name="graph_evidence",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"g_corr": g_corr, "term_graph": term_graph, "r_ep": r_ep, "term_ep": term_ep},
            ))

        metrics = compute_fidelity_metrics(evals, "graph_evidence", "Temporal heterogeneous graph energy and multi-hop episode risk", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 8: Asset / Trust / MITRE
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_asset_trust_mitre(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 8)
        cfg = self._RiskConfig(
            use_signature=True,
            use_ml=False,
            use_statistical=False,
            use_trust=True,
            use_history=False,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=True,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            sev = rng.randint(1, 5)
            a_crit = rng.uniform(0.5, 2.0)
            trust_val = rng.uniform(0.1, 0.9)
            entity_key = f"asset_host_{i}"
            self.engine.set_trust(entity_key, trust_val)

            sig = [type("Sig", (), {"severity": sev, "confidence": 0.95, "rule_name": "r_mitre", "mitre_technique": "T1078"})()]
            res = self.engine.score_risk(entity_key, sig, None, None, a_crit=a_crit, override_config=cfg)

            raw_sig = min(1.0, sev / 5.0)
            term_sig = cfg.w_sig * raw_sig
            threat_scaled = term_sig * a_crit
            trust_sub = cfg.w_trust * trust_val
            reconstructed = round(float(np.clip(threat_scaled - trust_sub, 0.0, 1.0)), 4)

            evals.append(self._eval_single(
                event_id=f"ASSET-{i}",
                path_name="asset_trust_mitre",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"a_crit": a_crit, "threat_scaled": threat_scaled, "trust_sub": trust_sub},
            ))

        metrics = compute_fidelity_metrics(evals, "asset_trust_mitre", "Asset criticality scaling modulated by dynamic trust", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 9: Clipping Boundaries
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_clipping_boundaries(self, n: int = 100) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 9)
        cfg = self._RiskConfig(
            use_signature=True,
            use_ml=True,
            use_statistical=False,
            use_trust=True,
            use_history=True,
            use_graph=False,
            use_forecast=False,
            use_ti=False,
            use_uncertainty=False,
            use_asset_crit=True,
            use_evidence_quality=False,
        )
        evals = []
        for i in range(n):
            if i % 2 == 0:
                # Lower boundary test: zero threat, high trust -> raw_risk < 0.0, clipped to 0.0
                entity_key = f"clip_floor_{i}"
                self.engine.set_trust(entity_key, 1.0)
                res = self.engine.score_risk(entity_key, [], None, None, override_config=cfg)
                reconstructed = 0.0
            else:
                # Upper boundary test: extreme multi-threat, a_crit=2.0 -> raw_risk > 1.0, clipped to 1.0
                entity_key = f"clip_ceil_{i}"
                self.engine.set_trust(entity_key, 0.0)
                sig = [type("Sig", (), {"severity": 5, "confidence": 1.0, "rule_name": "r_mass", "mitre_technique": "T1486"})()]
                ml = type("ML", (), {"ensemble_score": 1.0, "confidence": 1.0})()
                res = self.engine.score_risk(entity_key, sig, ml, None, h_boost=1.0, a_crit=2.0, override_config=cfg)
                reconstructed = 1.0

            evals.append(self._eval_single(
                event_id=f"CLIP-{i}",
                path_name="clipping",
                engine_score=res.risk_score,
                reconstructed_score=reconstructed,
                step_details={"boundary": "floor" if i % 2 == 0 else "ceiling"},
            ))

        metrics = compute_fidelity_metrics(evals, "clipping", "Lower floor (0.0) and upper ceiling (1.0) saturation", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Path 10: Complete Score Path
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_complete_score_path(self, n: int = 200) -> Tuple[FidelityMetrics, List[EventFidelityEvaluation]]:
        rng = random.Random(self.seed + 10)
        evals = []
        for i in range(n):
            sev = rng.randint(1, 5)
            a_ml = rng.uniform(0.0, 1.0)
            delta_d = rng.uniform(0.0, 2.0)
            h_boost = rng.uniform(0.0, 0.5)
            g_corr = rng.uniform(0.0, 0.5)
            p_fore = rng.uniform(0.0, 0.4)
            ti_score = rng.uniform(0.0, 0.5)
            r_ep = rng.uniform(0.0, 0.4)
            a_crit = rng.uniform(0.8, 1.8)
            trust_val = rng.uniform(0.0, 0.8)

            entity_key = f"full_host_{i}"
            self.engine.set_trust(entity_key, trust_val)

            sig = [type("Sig", (), {"severity": sev, "confidence": 0.95, "rule_name": f"rule_{i}", "mitre_technique": "T1046"})()]
            ml = type("ML", (), {"ensemble_score": a_ml, "confidence": 0.85})()
            stat = type("Stat", (), {"behavioral_drift": delta_d, "confidence": 0.80, "flags": [], "mitre_techniques": []})()

            res = self.engine.score_risk(
                entity_key,
                sig,
                ml,
                stat,
                h_boost=h_boost,
                g_corr=g_corr,
                p_fore=p_fore,
                ti_score=ti_score,
                r_ep=r_ep,
                a_crit=a_crit,
            )

            # Analytical replay via DecisionTrace
            trace = res.trace
            replayed = self._replay_decision_trace(trace)

            evals.append(self._eval_single(
                event_id=trace.event_id,
                path_name="complete_score_path",
                engine_score=res.risk_score,
                reconstructed_score=replayed,
                step_details=trace.computational_steps,
            ))

        metrics = compute_fidelity_metrics(evals, "complete_score_path", "End-to-end full multi-signal risk controller fusion", self.tolerance)
        return metrics, evals

    # ─────────────────────────────────────────────────────────────────────────
    # Master Execution: All 10 Paths
    # ─────────────────────────────────────────────────────────────────────────
    def run_all_paths(self, n_per_path: int = 100) -> ComprehensiveFidelityReport:
        import time
        results: Dict[str, FidelityMetrics] = {}
        evaluations: Dict[str, List[EventFidelityEvaluation]] = {}

        path_runners = [
            ("base_scoring", self.evaluate_base_scoring),
            ("adaptive_weighting", self.evaluate_adaptive_weighting),
            ("multipliers", self.evaluate_multipliers),
            ("rule_boosts", self.evaluate_rule_boosts),
            ("contextual_evidence", self.evaluate_contextual_evidence),
            ("historical_evidence", self.evaluate_historical_evidence),
            ("graph_evidence", self.evaluate_graph_evidence),
            ("asset_trust_mitre", self.evaluate_asset_trust_mitre),
            ("clipping", self.evaluate_clipping_boundaries),
            ("complete_score_path", lambda: self.evaluate_complete_score_path(n=n_per_path * 2)),
        ]

        for path_name, runner in path_runners:
            if path_name == "complete_score_path":
                metrics, evals = runner()
            else:
                metrics, evals = runner(n=n_per_path)
            results[path_name] = metrics
            evaluations[path_name] = evals

        all_passed = all(m.is_exact for m in results.values())

        return ComprehensiveFidelityReport(
            timestamp=time.time(),
            tolerance=self.tolerance,
            epsilon=self.epsilon,
            all_paths_passed=all_passed,
            path_results=results,
            sample_evaluations=evaluations,
        )
