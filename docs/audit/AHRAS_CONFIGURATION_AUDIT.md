# AHRAS Configuration Precedence & Security Audit
**Date:** September 30, 2026  
**Auditor:** Independent Technical Audit Team  

---

## 1. Configuration Hierarchy & Precedence

AHRAS resolves configuration using the following strict deterministic precedence (highest to lowest):
1. **Explicit API / Method Overrides** (e.g. `RiskConfig` parameters passed directly to `score_risk()`).
2. **System Environment Variables** (loaded via process environment and `.env` files).
3. **Hardened Default Settings** defined in `config/settings.py`.

---

## 2. Security-Critical Variables Audit

| Variable | Default Value | Acceptable Range / Options | Fail-Closed Policy |
|---|---|---|---|
| `AHRAS_ENV` | *(None - Required)* | `DEV`, `STAGING`, `PRODUCTION` | **REFUSES TO START** if unset. |
| `AHRAS_SECRET_KEY` | *(None - Required in PROD)* | Min 32 chars, high entropy | **REFUSES TO START** if weak or default in `PRODUCTION`. |
| `RESPONSE_MODE` | `STAGED_MITIGATION` | `STAGED_MITIGATION`, `AUTONOMOUS`, `OBSERVE_ONLY` | Fallback is safest staged mode. |
| `TRUSTED_PROXIES` | `127.0.0.1,::1,localhost` | Valid IP / CIDR list | Untrusted proxies cannot forge `X-Forwarded-For`. |
| `MAX_REQUEST_BYTES` | `10_485_760` (10MB) | `1MB` - `50MB` | Rejects payloads with HTTP 413. |
| `RATE_LIMIT_PER_MINUTE` | `120` | `10` - `10000` | Rejects excess traffic with HTTP 429. |

---

## 3. Subsystem Toggles (`RiskConfig`)

The core `RiskConfig` dataclass in `detection/risk_engine.py` provides independent toggles for every risk estimation component:
- `use_signature: bool`
- `use_ml: bool`
- `use_statistical: bool`
- `use_trust: bool`
- `use_history: bool`
- `use_graph: bool`
- `use_forecast: bool`
- `use_uncertainty: bool`
- `use_ti: bool`
- `use_deception: bool`
- `use_asset_crit: bool`
- `use_episode_reasoning: bool`
- `use_evidence_quality: bool`
- `use_selective_gate: bool`
- `use_dynamic_features: bool`

All toggles are verified for exact mathematical replay without drift.
