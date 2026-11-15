# Business Requirements & Acceptance Traceability — Purchase Intent

**Status:** Proposed stakeholder requirements, not a signed marketing/CRM implementation brief.

## Value hypothesis
A customer-session score may help marketing teams **prioritise** relevant outreach. The model predicts session purchase propensity; it does not show that a campaign influenced purchasing.

**Potential users:** marketing analyst, CRM lead, ecommerce product owner, privacy lead, operations owner and ML engineer. No named sponsor or signatory is evidenced.

**Illustrative As-Is:** undifferentiated session metrics and campaigns. **Reference To-Be:** quality-checked events → classification probability → consent and campaign eligibility → human-approved audience selection → controlled experiment → feedback/ROI review.

**Candidate business KPIs (unmeasured):** incremental conversions in treatment vs control, contact cost, opt-out/complaints, intervention precision, false-positive targeting burden, uplift by cohort and incremental margin after campaign spend.

## Requirements traceability
| ID | Priority | Requirement | Repository anchor | Acceptance evidence required |
|---|---|---|---|---|
| BR-01 | Must | Define a purchase event and observation horizon unambiguously | Original notebook and feature-processing description | Approved event dictionary + timestamps |
| BR-02 | Must | Restrict targeting to consented, eligible customers | **Business control not evidenced as implemented** | Privacy/legal and marketing signoff |
| FR-01 | Must | Preprocess numerical/categorical session fields consistently | `src/preprocess.py` | Missing values, unexpected categories and schema tests |
| FR-02 | Must | Compare LR, Random Forest and LightGBM without test leakage | `src/train_model.py` | Reproducible validation and held-out test report |
| FR-03 | Must | Store operating threshold with chosen model/version | Model bundle / inference logic | Changed-threshold and restart tests |
| FR-04 | Should | Provide authenticated scoring and bounded requests | `app.py` reference | Positive/negative API contract tests |
| FR-05 | Should | Record features, model version and threshold for review without revealing PII | Monitoring/MLflow references | Privacy-reviewed inference log sample |
| NFR-01 | Must | Define rollback, model owner, alert thresholds and uptime targets | Operations guide | Signed operation and incident playbook |

**Decision point:** a ROC-AUC result on a fixed dataset is not a targeting-business-case acceptance condition. A randomized controlled experiment and ROI analysis are necessary for an incremental-value claim.

## Open approvals
Agree the event cutoff, data lawful basis/consent, campaign eligibility and capacity, acceptable opt-out rate, true incremental-value threshold, cohort fairness criteria, retention, and stop-loss. Assign decision rights to marketing, product, data/privacy, model-risk and operations owners; none are signed here.
