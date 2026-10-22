# ADR 0001 - SeaweedFS as the local S3-compatible object store

Status: accepted (2026-10-22)

## Context
The spec names MinIO as an example S3-compatible store. On 2026-10-22 no MinIO community image could be pulled
(`minio/minio` and `quay.io/minio/minio` tags returned "manifest unknown"; MinIO stopped publishing community
images in 2025).

## Decision
Use `chrislusf/seaweedfs:4.01` in single-node mode with its S3 gateway on 8333. Credentials are rendered from
`.env` at container start (`infra/seaweedfs/entrypoint.sh`); nothing secret is committed. The application only
uses boto3 with path-style addressing, so any S3-compatible service can replace it by changing
`S3_ENDPOINT_URL`.

## Consequences
Health checks use the SeaweedFS master status endpoint. DVC uses the same store (bucket `dvc`) as its remote;
MLflow proxies artifacts into bucket `mlflow-artifacts`.
