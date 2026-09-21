# AHRAS Research Roadmap & Experimental Protocol

> **Platform**: Adaptive, Hybrid, Risk-Aware Security Intelligence and Defense Platform  
> **Scientific Integrity**: All empirical metrics originate exclusively from live computational runs. Zero fabricated figures or arbitrary decimal claims.

---

## 1. Research Questions & Hypotheses

### RQ1: Computational Fidelity & Deterministic Replay
* **Research Question**: Can an uncertainty-aware, multi-signal adaptive cyber risk controller maintain deterministic execution fidelity ($\Delta \le 10^{-6}$) across thousands of live operational traces?
* **Hypothesis**: By decoupling risk arithmetic into a DAG-structured `DecisionTrace` with closed-form partial derivatives, 100% of decisions can be reconstructed without loss of precision ($|\Delta_{\text{abs}}| \le 10^{-6}$).
* **Experiment**: Run 10,000 randomized and edge-case operational traces through `AdaptiveRiskEngine` and verify against `replay_decision_trace`.
* **Datasets**: Synthetic stress suite, CIC-IDS2017 held-out evaluation slice.
* **Metrics**: Maximum Absolute Error ($E_{\text{abs}}$), Mean Absolute Error (MAE), P95 Error, Tolerance Pass Rate ($\alpha = 10^{-4}$).
* **Expected Output**: Replay pass rate = 100%, Max Delta $\le 10^{-4}$.

---

### RQ2: Temporal & Cross-Dataset Generalization
* **Research Question**: How severe is the performance degradation when an ensemble IDS trained on earlier network flows or an enterprise environment is deployed on later flows or a different network topology?
* **Hypothesis**: Standard deep/ensemble classifiers degrade significantly ($> 30\%$ F1 drop) under cross-dataset shifts, whereas modular evidence normalization (OCSF standard) coupled with Welford-based statistical drift mitigation bounds degradation within $12\%$.
* **Experiment**: Train on CIC-IDS2017 (Wednesday morning/afternoon), evaluate on Thursday/Friday (temporal shift) and UNSW-NB15 (cross-domain shift).
* **Datasets**: CIC-IDS2017, UNSW-NB15.
* **Metrics**: Macro F1, Precision, Recall, FPR, PR-AUC, Out-of-Domain Degradation Ratio ($\Delta_{\text{OOD}}$).
* **Expected Output**: Honest degradation quantification documented in evaluation tables.

---

### RQ3: Open-Set Unknown Attack Detection
* **Research Question**: Can latent representation metric learning (Mahalanobis OOD distance) reliably identify completely unseen zero-day attack families without escalating benign false positives?
* **Hypothesis**: Projecting normalized multimodal evidence into a constrained latent space allows separation of known vs unknown distributions by setting nonconformity thresholds.
* **Experiment**: Hold out entire attack classes (e.g., Ransomware or PortScan) during training. Test on mixed benign, known, and unseen attack families.
* **Datasets**: CIC-IDS2017 multi-class partition, held-out attack slices.
* **Metrics**: Known-Class F1, Unknown-Family Recall, False Unknown Rate (FUR), AUROC/AUPRC.
* **Expected Output**: Unknown Family Recall $\ge 75\%$ with False Unknown Rate $\le 5\%$.

---

### RQ4: Continual Learning & Catastrophic Forgetting
* **Research Question**: Does a 5-compartment multi-memory replay buffer prevent catastrophic forgetting during abrupt behavioral concept drift compared to naive online retraining?
* **Hypothesis**: Isolating hard negatives, historical prototypes, and recent drift samples maintains backward transfer on previously seen attack families ($> 95\%$ retention) while adapting fusion weights to new attack baselines.
* **Experiment**: 500-step streaming simulation with injected baseline shift and new attack vectors. Compare Naive Streaming Update vs Multi-Memory Replay Buffer.
* **Datasets**: Continuous streaming synthetic telemetry + drifting CIC-IDS2017 sequences.
* **Metrics**: Adaptation Gain MSE, Backward Transfer Retention ($BWT$), Catastrophic Forgetting Ratio ($CFR$).
* **Expected Output**: $BWT \ge 0.95$, Adaptation Gain MSE $\le 0.02$.

---

### RQ5: Relational Multi-Hop Campaign Reasoning
* **Research Question**: Does temporal heterogeneous graph message passing (TGNN) and Noisy-OR attack path aggregation improve multi-hop lateral movement detection over isolated event-level detectors?
* **Hypothesis**: Attackers executing reconnaissance, credential dumping, and lateral movement across multiple nodes exhibit weak point-anomaly signals but high relational graph energy.
* **Experiment**: Simulate 2-to-5 hop lateral movement sequences across 50 enterprise hosts. Compare Single-Event Anomaly Baseline ($B_1$) vs TGNN Path Reasoner.
* **Datasets**: Graph-native simulated enterprise lateral movement logs.
* **Metrics**: Lateral Movement F1, Campaign Attribution Accuracy, Alert Volume Reduction ($\% \Delta$).
* **Expected Output**: Lateral Movement F1 increases from $\le 0.10$ (isolated events) to $\ge 0.88$ (TGNN path reasoning), with alert reduction $\ge 60\%$.

---

### RQ6: Safe Selective Autonomy & Conformal Gating
* **Research Question**: Can split conformal prediction nonconformity quantile thresholding statistically bound the operational cost of false autonomous containment in high-speed SOC environments?
* **Hypothesis**: Enforcing conformal selective gates guarantees that autonomous actions are only triggered when empirical nonconformity error is bounded below user-specified $\alpha$, routing uncertain events to human analyst queues.
* **Experiment**: 1,000 incident scenarios evaluating Conformal Selective Gate vs Uncalibrated Fixed-Threshold Heuristics under adversarial noise.
* **Datasets**: Multi-source telemetry with variable signal-to-noise ratio.
* **Metrics**: Risk-to-Action Safety Efficiency (RASE), False Intervention Rate, Abstention Rate.
* **Expected Output**: False intervention reduction $\ge 65\%$, RASE score improvement $\ge 35\%$.

---

## 2. Experimental Execution Matrix

| Experiment ID | Primary RQ | Script File | Target Artifact |
| :--- | :--- | :--- | :--- |
| **EXP-01** | RQ1 (Fidelity) | `evaluation/xai_fidelity_experiment.py` | `RESULTS_FINAL.json` (Table 12) |
| **EXP-02** | RQ2 (Generalization) | `evaluation/run_real_benchmarks.py` | `REAL_DATASET_VALIDATION_FINAL.json` |
| **EXP-03** | RQ3 (Open-Set) | `evaluation/adversarial_suite.py` | `CLAIMS_MANIFEST_FINAL.json` (CLM-04) |
| **EXP-04** | RQ4 (Continual) | `evaluation/research_experiments.py` | `CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json` |
| **EXP-05** | RQ5 (Graph) | `evaluation/research_experiments.py` | `GNN_GRAPH_NATIVE_RESULTS_FINAL.json` |
| **EXP-06** | RQ6 (Safety/RASE)| `evaluation/response_simulation.py` | `CLOSED_LOOP_FINAL.json` |
