# Research Frontier G: Resource-Aware Security Intelligence Controller

> **Architecture Design Document**  
> **Status**: Approved Design Specification  
> **Implementation Target**: `detection/resource_controller.py`, `evaluation/run_resource_controller.py`

---

## 1. Executive Summary & Problem Formulation

In high-throughput enterprise networks (10,000 to 100,000 events/second), executing full deep learning inference, multi-hop graph neural network traversal, and counterfactual explanation search on every single incoming flow record is computationally prohibitive.

Traditional solutions fall into two flawed extremes:
1. **Monolithic Heavy Pipeline**: Runs heavy models on all events $\to$ Buffer overflows, multi-second queue latency, packet drops, CPU exhaustion.
2. **Static Fixed Heuristic**: Hardcodes fixed model assignments $\to$ Misses zero-days on high-criticality assets; wastes CPU evaluating benign background telemetry with GNNs.

AHRAS introduces a **Resource-Aware Security Intelligence Controller** that dynamically modulates analysis depth $d \in \{0, 1, 2, 3, 4, 5\}$ and active sensor collection modalities based on real-time operational load, queue pressure, asset criticality, and epistemic uncertainty.

---

## 2. Six-Tier Analysis Depth Hierarchy

AHRAS standardizes computational effort into six explicit, measurable depth levels:

| Level | Analysis Tier | Model / Logic Components | Typical Latency | Compute Complexity | Memory Footprint |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | **Streaming Sketch Screening** | Count-Min sketch, Welford variance floor | $< 0.05\text{ ms}$ | $O(1)$ | $0.8\text{ MB}$ |
| **1** | **Signature & Fast Path** | Pre-compiled regex rules, single-feature filters | $0.20\text{ ms}$ | $O(K_{\text{rules}})$ | $2.5\text{ MB}$ |
| **2** | **Full Anomaly Ensemble** | Isolation Forest + Autoencoder + One-Class SVM | $4.50\text{ ms}$ | $O(N_{\text{trees}} + D_{\text{ae}})$ | $18.0\text{ MB}$ |
| **3** | **Multimodal Fusion** | Cross-modal attention ($z_{\text{net}}, z_{\text{proc}}, z_{\text{id}}$) | $12.0\text{ ms}$ | $O(D_{\text{modal}}^2)$ | $45.0\text{ MB}$ |
| **4** | **Relational Graph Reasoner** | 2-hop TGNN ego-network message passing | $35.0\text{ ms}$ | $O(|\mathcal{V}| d + |\mathcal{E}|)$ | $85.0\text{ MB}$ |
| **5** | **Deep Investigation** | Counterfactual search, Security Twin simulation | $150.0\text{ ms}$ | $O(N_{\text{steps}} \cdot M)$ | $180.0\text{ MB}$ |

---

## 3. Mathematical Optimization Formulation

At each step, the controller selects the optimal depth $d^* \in \{0, \dots, 5\}$ that maximizes operational utility under current system constraints:

$$d^* = \arg\max_{d \in \{0, \dots, 5\}} \Big[ \text{SecurityValue}(e, d) - \lambda_1(L) \cdot C_{\text{lat}}(d) - \lambda_2(C) \cdot C_{\text{cpu}}(d) - \lambda_3(M) \cdot C_{\text{mem}}(d) \Big]$$

### 3.1 Security Value Term ($\text{SecurityValue}(e, d)$)
The expected security payoff of deepening the investigation:
$$\text{SecurityValue}(e, d) = \alpha \cdot \text{Criticality}(e_{\text{asset}}) + \beta \cdot \text{RiskPrior}(e) + \gamma \cdot U_{\text{epistemic}}(e) \cdot \mathbf{1}(d \ge 2)$$

Where:
* If asset criticality is high ($\text{Criticality} \ge 0.90$, e.g. Domain Controller, Database Vault), the threshold for deepening to Level 4/5 drops significantly.
* If epistemic uncertainty $U$ is high (indicating potential novel or zero-day behavior), deep analysis is strongly favored.

### 3.2 Dynamic Penalty Weights ($\lambda_1, \lambda_2, \lambda_3$)
The penalty multipliers scale non-linearly with system pressure:
$$\lambda_1(L) = \lambda_1^0 \cdot \exp\left(\frac{Q_{\text{len}}}{Q_{\text{max}}}\right), \quad \lambda_2(C) = \lambda_2^0 \cdot \left(\frac{1}{1 - \text{CPU}_{\text{util}} + \epsilon}\right)$$

Under severe load ($\text{CPU} > 85\%$ or queue $> 80\%$), the controller sheds load by routing low-risk background traffic to Level 0/1, while strictly preserving Level 3/4 analysis for high-criticality assets.

---

## 4. Adaptive Telemetry Sensor Gating

In addition to model depth, the controller dynamically requests sensor data tiers:
* **Tier A (Network Only)**: Base tap telemetry ($5\text{ KB/flow}$).
* **Tier B (Network + Endpoint Process)**: Dynamically activated if Risk $> 0.40$ or Asset Criticality $> 0.70$.
* **Tier C (Full Multimodal & Identity Audit)**: Activated on critical anomalies or conformal autonomy escalation.

---

## 5. Experimental Evaluation Protocol (EXP-28)

We evaluate three operational regimes under synthetic burst loads (1,000 to 50,000 EPS):
1. **Full Monolithic Analysis**: Unconditionally routes 100% of events to Level 4.
2. **Static Route Baseline**: Fixed threshold routing without system feedback.
3. **AHRAS Resource-Aware Controller**: Real-time closed-loop utility maximization.

### Target Metrics:
* Macro F1 and Unknown Attack Recall retention ($\ge 98\%$ retention vs Monolithic).
* Throughput scaling (events/sec).
* Latency percentiles (P50, P95, P99).
* CPU Utilization (%) and RAM Peak (MB).
* Tier distribution breakdown (% events evaluated at Levels 0, 1, 2, 3, 4, 5).
