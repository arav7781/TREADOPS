# ADR 0004 - Store sanitized images only; one ingest path for training and serving

Status: accepted (2026-10-22)

## Decision
The API validates the upload (size, format, pixel count before full decode), applies EXIF orientation, and
re-encodes it without metadata (JPEG q95, 4:4:4). Only this sanitized image is stored; raw bytes (which may
carry GPS or device data) are hashed for identity and discarded. The `prepare` stage runs exactly the same
`ingest()` on archive originals, so canonical training bytes and serving bytes come from one code path.

## Consequences
The stage trace starts at "ingest" with the API's measured timing. Unit tests assert that serving and training
produce byte-identical canonical images and tensors for the same input.
