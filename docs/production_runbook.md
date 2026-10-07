# Production Runbook

This project now implements production-style application and MLOps controls. It should only be called a live production system after it is deployed against real commerce traffic with real downstream decisions and operational ownership.

## Release flow

1. Run the training pipeline against the approved source dataset.
2. Review validation model selection, threshold tuning, locked-test metrics, data-quality alerts, drift output, and business-impact output.
3. Review `outputs/deployment_manifest.json`.
4. Promote the immutable `outputs/model_bundle.joblib` and record its SHA-256.
5. Store API credentials and the expected model fingerprint in the secret manager.
6. Deploy behind TLS ingress/API gateway.
7. Verify `/live`, `/ready`, `/metrics`, and a known-good prediction request.

## Runtime controls

- `/live` and `/health`: process liveness.
- `/ready`: model readiness.
- `/metrics`: Prometheus request and latency metrics plus model info.
- `/metadata`: authenticated model/threshold/schema/fingerprint metadata.
- `/predict`: authenticated bounded-batch inference.
- Every response gets `X-Request-ID`.
- Structured JSON logs omit raw prediction payloads.
- `MODEL_SHA256` can pin the exact promoted artifact.
- `ENABLE_DOCS=false` removes Swagger/OpenAPI endpoints in production.

## Suggested SLO targets

Targets only; measure them in the deployed environment before making claims:

- 99.9% monthly availability.
- p95 single-request service latency below 150 ms under normal load.
- 5xx rate below 0.5% over 15 minutes.
- zero traffic to replicas that fail model readiness.

## Monitoring

Application metrics should be joined with business/outcome monitoring: conversion rate by score band, calibration, precision/recall at the active threshold, delayed labels, drift, schema rejections, and targeting lift. Alert thresholds should be based on observed production distributions rather than notebook/test values.

## Rollback

Keep immutable image and model versions. Roll back both together when a release causes errors or material outcome degradation. Never replace a promoted artifact in place.

## Real-world integrations still required

A live production environment still needs centralized logs/traces, cloud IAM, managed secrets, registry controls, ingress rate limiting/WAF, image/dependency scanning, alert routing, delayed-label ingestion, canary rollout, outcome feedback, and an accountable on-call owner.
