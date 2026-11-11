# Business Requirements & Solution Design — Online Shopping Purchase Intent ML

A project-level **business requirements and solution-design record**, not a stakeholder-approved BRD or completed UAT.

## Business objective and workflow

**Objective:** Analyse session signals to estimate purchase intent and support consideration of more targeted outreach. A predicted purchase is not a realized order or proven campaign uplift.

**Potential stakeholders:** marketing analyst, CRM team, e-commerce product owner, business analyst and AI engineer.

**Possible As-Is → reference To-Be:** Undifferentiated session reporting → standardized customer-event features → model comparison → validation-only threshold selection → API propensity scoring → suggested campaign workflow with ongoing KPI review.

## Requirements Traceability Matrix

| ID | Requirement | Evidence | Suggested acceptance check |
|---|---|---|---|
| BR-01 | Address class imbalance | Evaluation reports | Review ROC-AUC, recall and precision instead of accuracy alone |
| FR-01 | Standardize input data | `src/preprocess.py` | Confirm numeric/categorical fields are consistently transformed |
| FR-02 | Compare candidate approaches without test leakage | `src/train_model.py` | Verify validation-based model selection |
| FR-03 | Version the prediction threshold with the model | Model bundle, `src/inference.py` | Confirm inference uses the stored threshold |
| FR-04 | Provide structured scores through an API | `app.py` | Test accepted/rejected request payloads |
| NFR-01 | Preserve reproducible evaluation evidence | MLflow and `outputs/` | Inspect test metrics, experiment parameters and data-quality outputs |

These are **proposed acceptance checks**, not customer UAT.

## Digital transformation and KPI design

A business pilot should define customer consent, conversion-event definitions, targeting constraints, A/B test design, incremental conversion, campaign cost and operational owners. Model AUC alone does not establish commercial value.

See [AI readiness](../ai-enablement/readiness-and-governance.md).
