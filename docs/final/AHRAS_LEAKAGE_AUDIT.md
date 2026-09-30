# AHRAS Leakage Audit & Cross-Modality Temporal Isolation
**Date:** September 30, 2026  
**Auditor:** Scientific Integrity & Data Audit Team  

---

## 1. Leakage Analysis Across Dimensions

| Dimension | Inspection Mechanism | Audit Result | Status |
|---|---|---|---|
| **Temporal Integrity** | Ensure prediction at $t$ only uses historical data $< t$ | `AttackPredictor` and `_WelfordAccumulator` use strict pre-event time windows | **VERIFIED (NO LEAKAGE)** |
| **Open-Set Isolation** | Verify withheld attack classes are absent in training splits | Protocols E & F explicitly partition unseen variants outside training matrices | **VERIFIED (NO LEAKAGE)** |
| **Cross-Dataset Isolation**| Validate source and target feature standardizers are fitted only on source train sets | `StandardScaler` fitted exclusively on in-domain train partitions | **VERIFIED (NO LEAKAGE)** |
| **Entity / Host Contamination**| Ensure entities in test sets are not memorized in train sets | Cross-environment splits evaluate disjoint entity pools | **VERIFIED (NO LEAKAGE)** |
| **Multimodal Synchronization**| Verify timestamp alignment across host and network telemetry without future lookahead | OCSF event-time normalization uses monotonic millisecond timestamps | **VERIFIED (NO LEAKAGE)** |

---

## 2. Scientific Summary

No lookahead contamination, label leakage, or test-set tuning was detected across the active evaluation splits.
