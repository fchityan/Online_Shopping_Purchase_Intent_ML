# Solution Architecture & Session-Data Contract — Purchase Intent

**Status:** Reference ML/API design; commerce integration and real campaign activation are not evidenced.

```text
Shopping session events (approved source)
  → schema validation & feature engineering
  → stratified train / validation / held-out test
  → Logistic Regression / Random Forest / LightGBM
  → validation model + threshold selection
  → model bundle (preprocessing, threshold, version)
  → FastAPI score → consent / eligibility / budget gate (proposed)
  → controlled marketing treatment and outcome capture (proposed)
  → MLflow and drift/performance review
```

## Business data-contract fields
| Domain | Required semantic rule | Risk to resolve |
|---|---|---|
| Session ID/time | Unique session and prediction cutoff | Avoid cross-session leakage, duplicates and late events |
| Engagement features | Activity observed **before** outcome | Exclude post-purchase/marketing-treatment fields |
| Purchase target | Realized conversion label at defined horizon | Missing/late conversion attribution |
| Segment attributes | Minimal lawful features to assess effects | Sensitive/proxy features, consent and fairness |
| Output | Probability/rank, label threshold and model version | Score calibration, interpretation and suppression rules |
| Operational audit | Campaign eligibility, treatment version and outcome | Pseudonymisation, retention and audit access |

The actual code-defined schema is authoritative; this is a **proposed business contract**, not an approved production interface specification.

## Design decisions
- **ADR-PI-01:** Compare models by ranking and precision/recall, not accuracy alone, because purchase events are imbalanced.
- **ADR-PI-02:** Select model and threshold on validation; reserve the test set for independent reporting.
- **ADR-PI-03:** Keep threshold, schema and model together in a versioned bundle.
- **ADR-PI-04:** Do not send automated interventions based solely on a score; apply consent and product constraints.
- **ADR-PI-05:** Use randomized pilot controls to establish business effect rather than treating AUC as uplift.

**Recorded model evidence:** LightGBM locked-test ROC-AUC **0.9253**; reported precision/recall change from **0.7986/0.6126** at threshold 0.50 to **0.6900/0.6702** at 0.35. These metrics describe the evaluated dataset, not real campaign economics. See [model governance](../ai-enablement/ai-readiness-and-model-governance-assessment.md).

Integration requires an event producer contract, identity and consent controls, scoring SLA, campaign-platform API schema, idempotent activation, holdout assignment, response logging and fallback to existing business rules.
