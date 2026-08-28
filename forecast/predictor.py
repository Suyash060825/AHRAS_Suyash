from __future__ import annotations
"""
AHRAS Temporal Risk Forecasting & Early Warning Module
-------------------------------------------------------
Implements per-indicator causal time-series risk forecasting using Holt's Linear
Exponential Smoothing (double exponential smoothing: level + trend).

Formula:
  level_t  = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
  trend_t  = beta  * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
  forecast(h) = level_t + h * trend_t

Key Capabilities:
  1. Multi-step risk trajectory prediction (horizon = 5 ticks/events)
  2. Early warning threshold-crossing alarm (will_breach_critical)
  3. Causal walk-forward MAE / RMSE accuracy estimation
  4. Lead-time measurement (time/events before true threshold crossing)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

log = logging.getLogger(__name__)

ALPHA = 0.50                      # Level smoothing factor
BETA  = 0.30                      # Trend smoothing factor
MIN_POINTS_FOR_FORECAST = 3       # Minimum observations
ESCALATION_THRESHOLD = 2.0        # Trend points per event to classify ESCALATING (on 0-100 scale)
DEFAULT_CRITICAL_THRESHOLD = 85.0 # Score bar for CRITICAL threat level


@dataclass
class ForecastResult:
    indicator: str
    data_points: int
    current_level: float
    trend: float                     # Points per event
    trend_label: str                 # ESCALATING | STABLE | DE-ESCALATING | INSUFFICIENT_DATA
    forecast_next: List[float]       # Predicted risk scores for next N events
    confidence: float                # 0.0 - 1.0 confidence score
    will_breach_critical: bool       # Whether forecast crosses threshold within horizon
    breach_in_events: Optional[int]  # Estimated steps until breach

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "data_points": self.data_points,
            "current_level": round(float(self.current_level), 2),
            "trend": round(float(self.trend), 2),
            "trend_label": self.trend_label,
            "forecast_next": [round(float(v), 2) for v in self.forecast_next],
            "confidence": round(float(self.confidence), 2),
            "will_breach_critical": self.will_breach_critical,
            "breach_in_events": self.breach_in_events,
        }


class AttackPredictor:
    """
    Stateless, high-throughput time-series predictor (< 1ms per inference).
    Evaluates historical risk series and forecasts future risk curves.
    """

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def predict(self, indicator: str, risk_history: List[float], critical_threshold: Optional[float] = None) -> ForecastResult:
        """
        risk_history: chronological list of past risk scores (0-100 or 0-1 normalized).
        """
        n = len(risk_history)
        if n < MIN_POINTS_FOR_FORECAST:
            return ForecastResult(
                indicator=indicator,
                data_points=n,
                current_level=float(risk_history[-1]) if risk_history else 0.0,
                trend=0.0,
                trend_label="INSUFFICIENT_DATA",
                forecast_next=[],
                confidence=0.0,
                will_breach_critical=False,
                breach_in_events=None,
            )

        max_val = max(risk_history) if risk_history else 0.0
        is_unit_scale = (max_val <= 1.0)

        if critical_threshold is None:
            crit_thresh = 0.85 if is_unit_scale else 85.0
        else:
            crit_thresh = critical_threshold

        level, trend = self._holt_fit(risk_history)

        forecast = []
        cap_val = 1.0 if is_unit_scale else 100.0
        for h in range(1, self.horizon + 1):
            val = max(0.0, min(cap_val, level + h * trend))
            forecast.append(val)

        thresh_esc = (ESCALATION_THRESHOLD / 100.0) if is_unit_scale else ESCALATION_THRESHOLD
        if trend > thresh_esc:
            label = "ESCALATING"
        elif trend < -thresh_esc:
            label = "DE-ESCALATING"
        else:
            label = "STABLE"

        variance = self._variance(risk_history)
        norm_var = 0.25 if is_unit_scale else 2500.0
        data_confidence = min(1.0, n / 20.0)
        stability_confidence = max(0.0, 1.0 - variance / norm_var)
        confidence = round(0.6 * data_confidence + 0.4 * stability_confidence, 3)

        breach_idx = None
        for i, v in enumerate(forecast):
            if v >= crit_thresh:
                breach_idx = i + 1
                break

        return ForecastResult(
            indicator=indicator,
            data_points=n,
            current_level=level,
            trend=trend,
            trend_label=label,
            forecast_next=forecast,
            confidence=confidence,
            will_breach_critical=breach_idx is not None,
            breach_in_events=breach_idx,
        )

    def predict_from_events(self, indicator: str, recent_events: List[dict]) -> ForecastResult:
        """Extracts risk_score from event dictionaries."""
        scores = []
        for e in recent_events:
            score = e.get("risk_score") or e.get("risk_score_100") or e.get("score")
            if score is not None:
                scores.append(float(score))
        return self.predict(indicator, scores)

    def _holt_fit(self, series: List[float]) -> Tuple[float, float]:
        level = float(series[0])
        trend = float(series[1] - series[0]) if len(series) > 1 else 0.0
        for y in series[1:]:
            prev_level = level
            level = ALPHA * float(y) + (1.0 - ALPHA) * (level + trend)
            trend = BETA * (level - prev_level) + (1.0 - BETA) * trend
        return level, trend

    def _variance(self, series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        return sum((x - mean) ** 2 for x in series) / len(series)

    def predict_fleet(self, indicator_histories: Dict[str, List[float]]) -> List[ForecastResult]:
        results = []
        for indicator, history in indicator_histories.items():
            try:
                results.append(self.predict(indicator, history))
            except Exception as e:
                log.warning(f"[FORECAST] Prediction failed for {indicator}: {e}")
        return results

    def top_escalating(self, indicator_histories: Dict[str, List[float]], n: int = 10) -> List[ForecastResult]:
        results = self.predict_fleet(indicator_histories)
        escalating = [r for r in results if r.trend_label == "ESCALATING"]
        escalating.sort(key=lambda r: r.trend, reverse=True)
        return escalating[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation & Lead-Time Metrics
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_errors(
    predictor: AttackPredictor,
    series: List[float],
    min_points: int = MIN_POINTS_FOR_FORECAST
) -> List[float]:
    """
    Causal one-step-ahead forecast error: e_t = y_hat_{t|t-1} - y_t.
    """
    errors: List[float] = []
    for t in range(min_points, len(series)):
        history = series[:t]
        actual = series[t]
        result = predictor.predict("__walk_forward__", history)
        if not result.forecast_next:
            continue
        one_step_ahead = result.forecast_next[0]
        errors.append(one_step_ahead - actual)
    return errors


def forecast_accuracy(
    predictor: AttackPredictor,
    series: List[float],
    min_points: int = MIN_POINTS_FOR_FORECAST
) -> Dict[str, Any]:
    """
    Computes walk-forward one-step-ahead MAE and RMSE.
    """
    errors = walk_forward_errors(predictor, series, min_points=min_points)
    if not errors:
        return {"n": 0, "mae": 0.0, "rmse": 0.0}
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return {"n": len(errors), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def threshold_crossing_lead_time(
    predictor: AttackPredictor,
    series: List[float],
    threshold: Optional[float] = None,
    min_points: int = MIN_POINTS_FOR_FORECAST
) -> Optional[int]:
    """
    Computes lead time: number of events BEFORE actual crossing that an alarm was raised.
    Returns (t_actual - t_pred) for earliest warning, or None if no warning / no crossing.
    """
    max_val = max(series) if series else 0.0
    is_unit_scale = (max_val <= 1.0)
    if threshold is None:
        thresh = 0.85 if is_unit_scale else 85.0
    else:
        thresh = threshold

    t_actual = next((i for i, v in enumerate(series) if v >= thresh), None)
    if t_actual is None or t_actual < min_points:
        return None

    for t_pred in range(min_points, t_actual):
        result = predictor.predict("__lead_time__", series[:t_pred], critical_threshold=thresh)
        if any(v >= thresh for v in result.forecast_next):
            return t_actual - t_pred
    return None
