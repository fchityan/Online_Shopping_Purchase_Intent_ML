# Security Policy

Never commit API keys, cloud credentials, customer/session data, or private production datasets.

Use the deployment secret manager for `API_KEY` and `MODEL_SHA256`, terminate TLS at the ingress/API gateway, keep `REQUIRE_API_KEY=true`, mount models read-only, run non-root, drop Linux capabilities, and apply gateway rate/request-size limits.

Production release pipelines should add organization-standard dependency scanning, image vulnerability scanning/signing, and secret scanning. Rotate any credential before removing it from Git history.
