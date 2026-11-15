# Test Assurance, Privacy/Security & Operations — Purchase Intent

**Current proof:** Documented code, MLflow artefacts and an evaluated test-dataset result. **Not evidenced:** marketing UAT, true incremental lift, production consent enforcement, live monitoring or commercial approval.

## Verification matrix
| Gate | Tests / expected evidence | Status |
|---|---|---|
| Input contract | Missing/invalid values, unseen categories, late event, duplication | Test design; execution not reverified here |
| Model | Leakage audit, repeated training, independent holdout and threshold lock | Reported metrics; reproducibility on current HEAD not independently verified |
| Calibration/robustness | Confidence calibration, time holdout, cohort stability, intervention capacity | Proposed |
| API | Auth bypass denial, rate/batch limits, error contract and restart recovery | Reference service; deployed execution open |
| Privacy | Consent suppression, opt-out, retention expiry, pseudonymised logs | Proposed business controls |
| Business UAT | Score explanations, audience preview, budget caps, cancellation | Not executed/approved |
| Pilot | Randomized audience treatment/control; economic uplift and complaints | Not evidenced |
| Release | Security scan, rollback + version pin, monitoring dashboards | Proposed deployment gates |

## Risk and control register
| Risk | Consequence | Proposed response |
|---|---|---|
| Outcome leakage or incorrect event horizon | Inflated metrics | Time-aware split and retrospective feature audit |
| Aggressive outreach | Complaints/brand harm | Consent filter, frequency cap, human approval |
| Poorly calibrated score | Inefficient capacity allocation | Calibration and conversion-rate tracking |
| Cohort disparities | Uneven targeting / discrimination | Subgroup performance and proxy review |
| Personal data exposure | Privacy and regulatory consequences | Least privilege, TLS, encrypted storage, log redaction, deletion schedule |
| Drift/failing scoring service | Stale recommendations | Drift alerts, kill switch, model/version rollback |

## Operating responsibilities (to assign)
Business sponsor sets KPI and stop-loss; marketing owner controls eligibility and creative; data owner provides approved event lineage; model owner monitors precision/recall/calibration; security/privacy owner reviews consent and access; platform owner manages releases and incidents. No person has been formally assigned in this documentation.

**Proposed incidents:** inaccurate scores, consent policy failure, integration outage, data-quality drop, model integrity mismatch. Stop activation → revoke problematic version → restore existing campaign rules → triage using run IDs → roll back image and model → record incident and revalidate.

**Monitoring:** API availability/latency/5xx; feature drift; score distribution; delayed measured conversion; suppression/opt-out and experiment uplift. Existing [runbook](../operational-readiness/service-transition-and-operations-runbook.md) includes reference thresholds; they are not achieved SLOs.

**Pilot exit:** validated data contract, model risk review, consent policy, test log, security assessment, controlled impact experiment, ownership register, rollback rehearsal and approvals. All remain separate evidence requirements.
