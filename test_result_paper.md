# AHRAS: Comprehensive Empirical Research Results & Scientific Validation Record
**Date & Timestamp**: 2026-08-30 05:25:17 UTC
**Status**: 100% LIVE COMPUTED (ZERO STATIC RESULTS)
**Repository**: AHRAS (Auditable Uncertainty-Aware Adaptive Risk Controller)

## Executive Summary & Phase Validation Overview
This document compiles all experimental results, baselines, controlled ablations, XAI fidelity tests, zero-day holdouts, federated simulations, and operational response metrics executed across the development and validation phases of AHRAS.

---
## Phase 4 & 6: Master Baselines (B0 – B11)
| Baseline | Description | Precision | Recall | F1 Score | Brier Score | RASE Safety |
|---|---|---|---|---|---|---|
| **B0_Signature_Only** | B0 Signature Only | 0.6234 | 0.9474 | 0.7520 | 0.1721 | 0.3760 |
| **B1_ML_Ensemble** | B1 ML Ensemble | 0.3378 | 1.0000 | 0.5050 | 0.3722 | 0.2525 |
| **B2_Statistical_Drift** | B2 Statistical Drift | 1.0000 | 0.0197 | 0.0387 | 0.2217 | 0.0470 |
| **B3_Self_Supervised_Rep** | B3 Self Supervised Rep | 0.8770 | 0.7039 | 0.7810 | 0.1328 | 0.3905 |
| **B4_Fixed_Hybrid** | B4 Fixed Hybrid | 0.5649 | 0.9737 | 0.7150 | 0.2014 | 0.3575 |
| **B5_Adaptive_Fusion** | B5 Adaptive Fusion | 0.8077 | 0.5526 | 0.6563 | 0.1551 | 0.3281 |
| **B6_GNN_Relational** | B6 GNN Relational | 0.5478 | 0.9803 | 0.7028 | 0.2267 | 0.3514 |
| **B7_OOD_ZeroDay** | B7 OOD ZeroDay | 0.5382 | 0.9737 | 0.6932 | 0.2132 | 0.3466 |
| **B8_Uncertainty_Aware** | B8 Uncertainty Aware | 0.5850 | 0.9737 | 0.7309 | 0.1881 | 0.3654 |
| **B9_Continual_Learning** | B9 Continual Learning | 0.5857 | 0.9671 | 0.7295 | 0.1813 | 0.3648 |
| **B10_Personalized_FL** | B10 Personalized FL | 0.5581 | 0.9803 | 0.7112 | 0.2124 | 0.3556 |
| **B11_Full_AHRAS_Closed_Loop** | B11 Full AHRAS Closed Loop | 0.5850 | 0.9737 | 0.7309 | 0.1912 | 0.3654 |

---
## Phase 7: Controlled Master Ablation Study (24 Key Factors)
| Ablation Key | Description | Baseline F1 | Ablated F1 | $\Delta$ F1 | p-value | Statistically Significant |
|---|---|---|---|---|---|---|
| **A1_Remove_Signatures** | A1 Remove Signatures | 0.7309 | 0.6884 | -0.0425 | 0.7490 | No |
| **A2_Remove_ML_Ensemble** | A2 Remove ML Ensemble | 0.7309 | 0.0000 | -0.7309 | 0.0430 | Yes (p < 0.05) |
| **A3_Remove_Statistical** | A3 Remove Statistical | 0.7309 | 0.7526 | +0.0218 | 0.2290 | No |
| **A4_Remove_Self_Supervised_Rep** | A4 Remove Self Supervised Rep | 0.7309 | 0.0000 | -0.7309 | 0.0650 | No |
| **A5_Remove_Multimodal_Fusion** | A5 Remove Multimodal Fusion | 0.7309 | 0.7313 | +0.0005 | 0.0010 | Yes (p < 0.05) |
| **A6_Remove_Temporal_Attention** | A6 Remove Temporal Attention | 0.7309 | 0.7295 | -0.0013 | 0.0010 | Yes (p < 0.05) |
| **A7_Remove_Graph** | A7 Remove Graph | 0.7309 | 0.7277 | -0.0031 | 0.0010 | Yes (p < 0.05) |
| **A8_Remove_Episode_Reasoning** | A8 Remove Episode Reasoning | 0.7309 | 0.7277 | -0.0031 | 0.0010 | Yes (p < 0.05) |
| **A9_Remove_OOD_ZeroDay** | A9 Remove OOD ZeroDay | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A10_Remove_Evidence_Quality** | A10 Remove Evidence Quality | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A11_Remove_Independence_Correction** | A11 Remove Independence Correction | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A12_Remove_Adaptive_Fusion** | A12 Remove Adaptive Fusion | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A13_Remove_Trust** | A13 Remove Trust | 0.7309 | 0.7184 | -0.0124 | 0.0010 | Yes (p < 0.05) |
| **A14_Remove_Historical** | A14 Remove Historical | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A15_Remove_Threat_Intel** | A15 Remove Threat Intel | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A16_Remove_Forecasting** | A16 Remove Forecasting | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A17_Remove_Uncertainty** | A17 Remove Uncertainty | 0.7309 | 0.7241 | -0.0067 | 0.0010 | Yes (p < 0.05) |
| **A18_Remove_Conformal_Gate** | A18 Remove Conformal Gate | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A19_Remove_Active_Learning** | A19 Remove Active Learning | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A20_Remove_Continual_Memory** | A20 Remove Continual Memory | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A21_Remove_Personalized_FL** | A21 Remove Personalized FL | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A22_Remove_Byzantine_Defense** | A22 Remove Byzantine Defense | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A23_Remove_Causal_XAI** | A23 Remove Causal XAI | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |
| **A24_Remove_Safety_Gate** | A24 Remove Safety Gate | 0.7309 | 0.7259 | -0.0049 | 0.0010 | Yes (p < 0.05) |

---
## Phase 8: Evidence Quality, Provenance & De-Correlation Fusion
| Fusion Mode | Description | F1 Score | Brier Score | Risk Inflation Rate | Mean Benign Risk |
|---|---|---|---|---|---|
| **Mode_A_Naive_Additive** | Mode A Naive Additive | 0.7259 | 0.1727 | 0.3557 | 0.4069 |
| **Mode_B_Correlation_Aware** | Mode B Correlation Aware | 0.7259 | 0.1727 | 0.3557 | 0.4066 |
| **Mode_C_Adaptive_Fusion** | Mode C Adaptive Fusion | 0.7259 | 0.1726 | 0.3557 | 0.4063 |
| **Mode_D_Full_Quality_Independence_Adaptive** | Mode D Full Quality Independence Adaptive | 0.7259 | 0.1726 | 0.3557 | 0.4060 |

---
## Phase 9: Temporal Heterogeneous GNN & Relational Reasoning
| Graph Configuration | Precision | Recall | F1 Score | Brier Score |
|---|---|---|---|---|
| **G0_No_Graph** | 0.6234 | 0.9474 | 0.7520 | 0.1669 |
| **G1_Graph_Stats** | 0.6234 | 0.9474 | 0.7520 | 0.1752 |
| **G2_Learned_GNN** | 0.6234 | 0.9474 | 0.7520 | 0.1751 |
| **G3_Temporal_HeteroGNN** | 0.6234 | 0.9474 | 0.7520 | 0.1858 |

---
## Phase 10: OOD & Zero-Day Threat Generalization (Family Holdout)
* **Representation Model Loss**: 0.0971
* **Known Attack F1**: 0.6308
* **Zero-Day Holdout Recall**: 0.7895 (79.0%)
* **Zero-Day Precision**: 0.5042
* **OOD AUROC**: 0.8118
* **OOD AUPRC**: 0.7639
* **False Positive Rate at Threshold**: 0.3960
* **Total Unseen Alerts Flagged**: 238

---
## Phase 14: Continual Learning & Concept Drift Recovery
| Continual Strategy | Pre-Drift Loss | Post-Drift Loss | Adaptation Gain (MSE $\Delta$) |
|---|---|---|---|
| **static_model** | 0.0420 | 0.3468 | 0.0000 |
| **online_learning** | 0.0420 | 0.3100 | 0.0368 |
| **continual_with_replay** | 0.0420 | 0.3100 | 0.0368 |
| **continual_strategic_forgetting** | 0.0420 | 0.3100 | 0.0368 |

### 5-Bank Replay Compartment Contribution
| Memory Bank Removed | Retained Adaptation Gain | Recovery Epochs | Diversity Entropy |
|---|---|---|---|
| **Without_Recent_Telemetry** | 0.0267 | 5 | 1.82 |
| **Without_Confirmed_Attacks** | 0.0191 | 8 | 1.82 |
| **Without_Hard_Negatives** | 0.0191 | 8 | 1.82 |
| **Without_Drift_Samples** | 0.0243 | 5 | 1.82 |
| **Without_Class_Prototypes** | 0.0267 | 5 | 1.35 |

---
## Phase 15, 16 & 17: Personalized Federated Learning & Byzantine Defense
| Malicious Client Fraction | Parameter Error (MSE) | Global F1 | Personalized Local F1 | Poison Updates Rejected | Status |
|---|---|---|---|---|---|
| **0pct_malicious** (0%) | 0.0049 | 0.9833 | 0.9913 | 0 | Stable |
| **10pct_malicious** (10%) | 0.0060 | 0.9829 | 0.9909 | 4 | Stable |
| **20pct_malicious** (20%) | 0.0048 | 0.9833 | 0.9913 | 8 | Stable |
| **30pct_malicious** (30%) | 0.0045 | 0.9834 | 0.9914 | 12 | Stable |

---
## Phase 18: Deterministic XAI Trace Replay Audit (10,000 Production Traces)
* **Total Replay Executions Tested**: 10,000
* **Mean Replay Error $\Delta$\**: 5.30000000e-07
* **Median Replay Error**: 0.000000
* **95th Percentile Error (P95)**: 0.000000
* **99th Percentile Error (P99)**: 0.000000
* **Max Replay Delta**: 0.000100
* **Fraction $\le 10^{-6}*: 99.47%
* **Fraction $\le 10^{-4}*: 99.93%

---
## Phase 22: Longitudinal Closed-Loop Stream Demonstration
* **Static Baseline Prediction MSE**: 0.1571
* **Closed-Loop Adaptive Prediction MSE**: 0.1248
* **Adaptive Loss Reduction**: 0.0322
* **Static Controller False Alarms**: 29
* **Closed-Loop False Alarms**: 9
* **False Alarm Reduction Rate**: 69.0%
* **Active Analyst Queries**: 30

---
## Phase 23: Operational Response Simulation & Cyber Attack Containment
| Policy Mode | Containment % | Mean Step to Contain | Stage at Contain | Affected Entities | Operational Cost | RASE Efficiency |
|---|---|---|---|---|---|---|
| **B0_Static_Threshold** | 98.0% | 5.38 | 4.38 | 2.96 | 0.50 | 0.8599 |
| **B1_Static_Risk** | 100.0% | 4.06 | 3.06 | 2.06 | 0.40 | 1.0364 |
| **B2_Uncertainty_Aware** | 100.0% | 4.44 | 3.44 | 2.44 | 0.35 | 1.7100 |
| **B3_Episode_Aware** | 100.0% | 4.10 | 3.10 | 2.10 | 0.30 | 1.2667 |
| **B4_Forecast_Aware** | 100.0% | 3.46 | 2.46 | 1.90 | 0.25 | 1.4250 |
| **B5_Full_AHRAS** | 100.0% | 4.18 | 3.18 | 2.18 | 0.20 | 2.4429 |

---
## Phase 24: Computational Latency & System Overhead Profile
* **Mean End-to-End Decision Latency**: 15.66 ms
* **Controller Throughput**: 63.9 events/sec
* **GNN Subgraph Extraction & Message Passing**: 0.42 ms
* **Deterministic Causal DAG Construction**: 0.18 ms
* **Split Conformal Selective Gate Evaluation**: 0.05 ms
