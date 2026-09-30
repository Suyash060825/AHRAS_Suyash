# AHRAS Exact Risk Engine Specification
**Date:** September 30, 2026  
**Auditor:** Mathematical & Core Architecture Team  
**Source of Truth:** [`detection/risk_engine.py`](file:///home/suyashpradhan/Downloads/AHRAS_Suyash-master/detection/risk_engine.py)  

---

## 1. Mathematical Formulation

The AHRAS Adaptive Risk Controller computes the composite threat score $R_t \in [0.0, 1.0]$ according to:

$$R_t = \text{Clip}_{[0.0, 1.0]} \left[ \Big( \text{Threat}_{\text{add}} \times \text{Mult}_{\text{crit}} \times \text{Mult}_{\text{unc}} \Big) - \text{Sub}_{\text{trust}} \right]$$

### A. Additive Threat Components ($\text{Threat}_{\text{add}}$)
$$\text{Threat}_{\text{add}} = \sum_{k} \text{Term}_k$$

Where:
1. **Signature Term:** $\text{Term}_{\text{sig}} = w_{\text{sig}} \cdot S_{\text{sig}} \cdot q_{\text{sig}}$
2. **ML Anomaly & Statistical Drift:** $\text{Term}_{\text{ml}} = w_{\text{ml}} \cdot A_{\text{ml}} \cdot (1 + \Delta D) \cdot q_{\text{ml}}$
3. **Historical Recidivism:** $\text{Term}_{\text{hist}} = w_{\text{hist}} \cdot H_{\text{boost}}$
4. **Graph Corroboration:** $\text{Term}_{\text{graph}} = w_{\text{graph}} \cdot G_{\text{corr}}$
5. **Proactive Forecast:** $\text{Term}_{\text{fore}} = w_{\text{fore}} \cdot P_{\text{fore}}$
6. **Threat Intelligence Prior:** $\text{Term}_{\text{ti}} = w_{\text{ti}} \cdot TI_{\text{score}}$
7. **Attack Episode Reasoning:** $\text{Term}_{\text{ep}} = w_{\text{ep}} \cdot R_{\text{ep}}$

### B. Multiplicative Modulation & Attenuation
- **Asset Criticality Multiplier:** $\text{Mult}_{\text{crit}} = A_{\text{crit}} \in [0.5, 2.0]$
- **Uncertainty Attenuation Multiplier:** $\text{Mult}_{\text{unc}} = (1 - U_{\text{penalty}})$ where $U_{\text{penalty}} = \text{Uncertainty} \times 0.30$

### C. Subtractive Trust Mitigation
- $\text{Sub}_{\text{trust}} = w_{\text{trust}} \cdot T_{\text{trust}}$

---

## 2. Default Hyperparameters (`RiskConfig`)
- $w_{\text{sig}} = 0.50$
- $w_{\text{ml}} = 0.30$
- $w_{\text{trust}} = 0.15$
- $w_{\text{hist}} = 0.10$
- $w_{\text{graph}} = 0.10$
- $w_{\text{fore}} = 0.05$
- $w_{\text{ti}} = 0.15$
- $w_{\text{ep}} = 0.10$

All operations are strictly bounded, zero-drift reconstructible via `DecisionTrace`, and preserve numerical float precision.
