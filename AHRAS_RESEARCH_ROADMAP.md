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

### RQ1b: Multi-Signal Marginal Utility & Proper Ablation (Phase 3)
* **Research Question**: What is the empirical marginal contribution of each detection mechanism, context modifier, and safety gate under leak-free temporal evaluation, and do contributions maintain statistical significance under Family-Wise Error Rate (FWER) control?
* **Hypothesis**: Disabling signature rules drops detection F1 by $> 50\%$, while removing conformal safety gating reduces operational safety efficiency (RASE) by $> 30\%$. All 18 controlled leave-one-out ablations maintain quantifiable effect sizes under 10,000 paired sample permutations and Holm-Bonferroni correction ($\alpha = 0.05$).
* **Experiment**: Evaluate 9 Canonical Progression Baselines (B1–B11) and 18 Controlled Leave-One-Out Ablations (A1–A18) across strict 3-way temporal splits (Train/Val/Test) with zero label leakage and validation-only threshold tuning.
* **Datasets**: Temporal benchmark stream, CIC-IDS2017 held-out slices.
* **Metrics**: Macro F1, Precision, Recall, Brier Score, Cohen's $d$, Empirical Permutation $p$-value, Holm-Bonferroni Adjusted $p$, RASE score.
* **Expected Output**: Verified in `PROPER_ABLATION_REPORT.json`; zero data leakage; canonical progression F1 progression from unimodal $B_1$ to closed-loop $B_{11}$.

---

### RQ1c: Controlled Adaptive Weight Learning (Phase 4)
* **Research Question**: Does online gradient adaptation with shadow validation improve risk calibration (Brier score and ECE) on held-out test data compared to static fixed weights, while preventing catastrophic drift?
* **Hypothesis**: Updating fusion weights on training feedback streams while locking the validation buffer reduces Brier score error by $\ge 10\%$ and Expected Calibration Error (ECE) by $\ge 50\%$, while shadow validation triggers freeze protection when drift exceeds 15%.
* **Experiment**: Train `AdaptiveWeightLearner` on train/feedback partitions only, validate on shadow holdout, and evaluate against fixed baseline weights on 100% untouched test data with 10,000 paired sample permutations.
* **Datasets**: Temporal benchmark stream, CIC-IDS2017 held-out slices.
* **Metrics**: Initial/Final weights, LR, update counts, Precision, Recall, F1, FPR, Brier score, ECE, Cohen's $d$, Permutation $p$-value.
* **Expected Output**: Verified in `ADAPTIVE_WEIGHT_EVALUATION_REPORT.json`; Brier reduction $> 10\%$, ECE reduction $> 50\%$, validation drift freeze verified.

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

### RQ4b: Historical Security Context & Recidivism Reasoning (Phase 6)
* **Research Question**: Does temporal historical security context and indicator recidivism tracking improve detection of stealthy multi-session persistent threats and repeat offenders over stateless single-event detectors, while controlling false positives through time-decayed memory?
* **Hypothesis**: Maintaining causal, time-decayed threat memory (recidivism boost modulated by 7-day, 30-day, and >30-day decay half-lives) elevates risk for chronic offenders while safely decaying dormant threats and suppressing false positives on benign recurring entities.
* **Experiment**: Simulate 60-day longitudinal enterprise security telemetry across 5 cohorts (Persistent, Dormant, Transient Attack, Benign Recurring, Benign Transient). Compare Stateless Baseline $H_0$ (`use_history=False`) vs Recidivism Engine $H_1$ (`use_history=True`) with 10,000 paired sample permutations.
* **Datasets**: 60-day longitudinal enterprise security telemetry stream (3,673 events).
* **Metrics**: Recidivist Threat Recall, Recidivist F1 Gain ($\% \Delta$), Detection Escalation Lead Time (sessions to Critical $R \ge 0.70$), Recency Decay Conformance ($1.0 \to 0.50 \to 0.25$), FPR Stability, Paired Permutation $p$-value, Cohen's $d$.
* **Verified Outcome**: Documented in `HISTORICAL_CONTEXT_REPORT.json`. Persistent recidivist threat recall elevates from $0.2838$ to $0.8598$ ($+109.19\%$ relative F1 gain, $p = 0.0001$, Cohen's $d = 0.3853$). Critical escalation lead time accelerates by $4.16$ sessions ($7.60 \to 3.44$ sessions). Recency decay exactly conforms to $1.0 \to 0.50 \to 0.25$, and benign FPR remains strictly stable at $0.0003$.

---

### RQ5: Relational Multi-Hop Campaign Reasoning
* **Research Question**: Does temporal heterogeneous graph message passing (TGNN) and Noisy-OR attack path aggregation improve multi-hop lateral movement detection over isolated event-level detectors?
* **Hypothesis**: Attackers executing reconnaissance, credential dumping, and lateral movement across multiple nodes exhibit weak point-anomaly signals but high relational graph energy.
* **Experiment**: Simulate 2-to-5 hop lateral movement sequences across 50 enterprise hosts. Compare Single-Event Anomaly Baseline ($B_1$) vs TGNN Path Reasoner.
* **Datasets**: Graph-native simulated enterprise lateral movement logs (30 campaigns, 3,500 background events across 4 tiers).
* **Metrics**: Lateral Movement F1, Campaign Attribution Accuracy, Alert Volume Reduction ($\% \Delta$).
* **Verified Outcome**: Lateral Movement F1 increases from $0.0200$ (isolated events) to $0.9565$ (TGNN path reasoning, gain of $+0.9365$, $p < 10^{-4}$), with alert volume reduction of $87.76\%$ ($95\%$ CI: $[85.35\%, 90.09\%]$) and false-positive reduction of $98.60\%$.

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
| **EXP-00** | RQ1b (Ablation) | `evaluation/run_proper_ablation.py` | `PROPER_ABLATION_REPORT.json` (Table 4) |
| **EXP-01** | RQ1 (Fidelity) | `evaluation/xai_fidelity_experiment.py` | `RESULTS_FINAL.json` (Table 12) |
| **EXP-04b**| RQ1c (Adaptive W) | `evaluation/run_adaptive_weight_evaluation.py` | `ADAPTIVE_WEIGHT_EVALUATION_REPORT.json` |
| **EXP-02** | RQ2 (Generalization) | `evaluation/run_real_benchmarks.py` | `REAL_DATASET_VALIDATION_FINAL.json` |
| **EXP-03** | RQ3 (Open-Set) | `evaluation/adversarial_suite.py` | `CLAIMS_MANIFEST_FINAL.json` (CLM-04) |
| **EXP-04** | RQ4 (Continual) | `evaluation/research_experiments.py` | `CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json` |
| **EXP-05** | RQ5 (Graph) | `evaluation/run_graph_correlation_evaluation.py` | `GRAPH_CORRELATION_REPORT.json` (Table 13) |
| **EXP-06-HIST** | RQ4b (History) | `evaluation/run_historical_context_evaluation.py` | `HISTORICAL_CONTEXT_REPORT.json` |
| **EXP-06** | RQ6 (Safety/RASE)| `evaluation/response_simulation.py` | `CLOSED_LOOP_FINAL.json` |
