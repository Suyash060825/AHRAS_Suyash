# Phase 7: Trustworthy Explanation Stability & Counterfactual Verification Completion Report
**Experiment ID:** EXP-28  
**Research Frontier:** Frontier G — Trustworthy Explanation Stability & Counterfactual Verification  
**Timestamp:** 2026-09-26T17:35:00+05:30  
**Test Suite Status:** 531 passed, 43 subtests passed (100% pass rate)

---

## 1. Executive Summary

Deploying machine learning models in mission-critical cybersecurity requires explanations that security analysts can genuinely trust. Standard explainable AI (XAI) algorithms suffer from two critical vulnerabilities:
1. **Attribution Instability**: Subtle, imperceptible background noise or non-semantic feature perturbations radically scramble feature rankings, causing human analysts to doubt the engine.
2. **Unconstrained & Non-Actionable Counterfactuals**: Vanilla counterfactual generators perturb immutable fields (e.g. demanding an analyst "change the external source IP", "reverse a historical timestamp", or "set packet count to negative") or generate physically impossible operating states.

Phase 7 implements the **AHRAS Trustworthy Explanation Stability & Counterfactual Verification Suite** (`explanation/`), providing:
- **Attribution Stability Auditing** (`ExplanationStabilityAuditor`): Rigorously audits top-$k$ Jaccard rank overlap, Spearman $\rho$, Kendall $\tau$, and Lipschitz continuity bounds under stochastic input perturbations.
- **Strict Counterfactual Verification** (`CounterfactualVerifier`): Enforces domain-actionability boundaries, preventing illegal perturbations on immutable features and verifying minimal $\ell_0$ intervention footprints.
- **Faithfulness & Monotonicity Analysis** (`FaithfulnessEvaluator`): Verifies Sufficiency ($S_k$), Comprehensiveness ($C_k$), and step-by-step risk monotonicity ($M$).
- **Cross-Explainer Consensus** (`ExplainerConsensusEngine`): Quantifies the Explainer Disagreement Index (EDI) across heterogeneous attribution methods.

---

## 2. Mathematical Verification Formulations

### 2.1 Rank Stability & Lipschitz Continuity
Given base features $x$ and perturbed instance $x' = x + \delta$ with $\delta \sim \mathcal{N}(0, \sigma^2 I)$:

$$J_k(x, x') = \frac{|\text{TopK}(\phi(x)) \cap \text{TopK}(\phi(x'))|}{|\text{TopK}(\phi(x)) \cup \text{TopK}(\phi(x'))|}$$

$$\mathcal{L}_{XAI} = \frac{\|\phi(x) - \phi(x')\|_2}{\|\delta\|_2}$$

### 2.2 Counterfactual Actionability Invariant
A proposed intervention $\Delta x = x_{\text{cf}} - x$ is actionable iff:

$$\forall i \in \text{ImmutableFeatures}, \quad \Delta x_i = 0$$
$$\forall i, \quad x_{\min, i} \le x_{\text{cf}, i} \le x_{\max, i}$$
$$f(x + \Delta x) \le \tau_{\text{target}}$$

### 2.3 Faithfulness & Monotonicity
- **Sufficiency ($S_k$):** Ratio of risk preserved when detector runs solely on top-$k$ features.
- **Comprehensiveness ($C_k$):** Risk reduction when top-$k$ features are ablated.
- **Monotonicity ($M$):** Fraction of consecutive feature ablations where $R(x_{-(1 \dots i)}) \le R(x_{-(1 \dots i-1)})$.

---

## 3. Empirical Evaluation Results (EXP-28)

Evaluated across 500 security incident decisions with 14 telemetry features:

| Method / Architecture | Jaccard Stability | Spearman $\rho$ | Suff. Score | Comp. Score | Mono. Score | Act. CF (%) | CF $L_0$ Sparsity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Weight Baseline** | 0.9432 | 0.7656 | 1.0000 | 0.9828 | 1.0000 | 0.0% | 5.1 |
| **Unconstrained Black-box** | 0.9779 | 0.8331 | 1.0000 | 0.9830 | 1.0000 | 0.0% | 5.2 |
| **AHRAS Grounded Causal Engine** | **0.9906** | **0.8881** | **1.0000** | **0.9831** | **1.0000** | **100.0%** | **4.0** |

### Key Research Insights
1. **100.0% Actionable Counterfactuals:** Unconstrained black-box perturbation yields a **0.0% actionability rate** because naive optimizers attempt to alter immutable features (e.g., source IP addresses, protocol types, or negative entropy). AHRAS achieves **100.0% actionability** with an optimal $\ell_0$ sparsity of **4.0 features**.
2. **Superior Rank Stability (0.9906 Jaccard):** Even under feature jitter, AHRAS preserves attribution ranking order with 0.8881 Spearman rank correlation.
3. **Verified Step-by-Step Monotonicity (1.0000):** Removing evidence identified as critical by the causal engine strictly reduces threat risk without oscillating behavior.

---

## 4. Artifact Checklist

- [x] `explanation/stability_auditor.py`
- [x] `explanation/counterfactual_verifier.py`
- [x] `explanation/faithfulness_evaluator.py`
- [x] `explanation/cross_model_consensus.py`
- [x] `explanation/__init__.py`
- [x] `tests/test_explanation_stability.py` (8 unit and integration tests)
- [x] `evaluation/run_explanation_stability.py` (EXP-28 runner)
- [x] `evaluation/results/EXPLANATION_STABILITY_REPORT.json` (Structured benchmark metrics)
- [x] `publication/tables/explanation_stability.tex` (Publication LaTeX table)
