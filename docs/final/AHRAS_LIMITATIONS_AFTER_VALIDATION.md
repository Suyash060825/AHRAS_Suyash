# AHRAS Limitations & Boundaries Post-Validation
**Date:** September 30, 2026  
**Auditor:** Independent Technical & Scientific Quality Team  

---

## 1. Overview of Known Operational Limitations

In adherence to the core principle of scientific truthfulness, the following explicit boundaries, failure conditions, and assumptions are formally documented:

### A. Telemetry Degradation Thresholds
- **Observation:** Under extreme packet loss ($> 30\%$) or sensor corruption, detection delay and uncertainty naturally elevate.
- **Behavior:** AHRAS attenuates autonomous action authority and downgrades mitigations to staged analyst approvals rather than making unsafe guesses.

### B. Heavy Ingress Traffic & Extreme Volumetric Bursts
- **Observation:** Under severe line-rate network flooding ($> 100,000\text{ EPS}$ per core), deep graph traversal must rely on streaming sketch screening (`streaming/sketch.py`) to prevent queue saturation.
- **Behavior:** OCSF streaming triage screens non-anomalous background traffic in $O(1)$ constant memory before deep neural graph propagation.

### C. Open-Set Unseen Attack Variants
- **Observation:** Completely novel zero-day attack families with zero known signatures or behavioral drift produce higher predictive entropy and wide conformal prediction sets.
- **Behavior:** AHRAS abstains from autonomous categorization and routes the ambiguous telemetry to Tier-2 SOC analysts.

### D. Dataset Availability Boundaries
- **Observation:** While local datasets (`CIC-IDS2017`, `UNSW-NB15`) are verified and benchmarked, 5 remote canonical datasets (`CSE-CIC-IDS2018`, `UGR'16`, `CTU-13`, `IoT-23`, `LANL Cyber1`) are designated **`BLOCKED`** in strict real mode until physical files are downloaded.
