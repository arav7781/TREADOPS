# Security policy

TreadOps is an independent engineering prototype maintained on a best-effort basis. Security reports are welcome
and are handled privately.

## Supported versions

Only the latest commit on `main` is supported. There are no maintained release branches.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub: **Security → Report a vulnerability**
(<https://github.com/arav7781/TREADOPS/security/advisories/new>). Only the maintainers can read these reports.

Please include:

- the affected component (web/BFF, API, worker, copilot, ML pipeline, infrastructure) and commit SHA;
- steps to reproduce, with a proof of concept if you have one;
- the impact you expect (for example data exposure, privilege escalation, a guardrail bypass);
- whether the issue is already public.

We will acknowledge the report, agree a fix and a disclosure date with you, and credit you in the advisory unless
you prefer otherwise.

## Scope

In scope: the code in this repository, including the BFF, the `/v1` API, job handlers, the copilot's guardrails and
tools, the container images and the CI workflows.

Out of scope:

- deployments run by third parties;
- the seeded demo accounts in a local `.env` (they exist for local use only);
- denial of service against the local development stack;
- vulnerabilities in dependencies that do not affect how TreadOps uses them (please report those upstream).

## How the system protects itself

These are the controls as implemented; see `services/api/treadops_api/security.py`,
`apps/web/src/lib/server/bff.ts` and `docs/adr/`.

| Area | Control |
|---|---|
| Passwords | Salted scrypt (n=2^14, r=8, p=1), compared in constant time |
| Sessions | HS256 JWT with required `iss`, `aud`, `exp`, `iat`, `sub` and `sid`, backed by a server-side session row that can be revoked; default lifetime 480 minutes (`SESSION_TTL_MINUTES`) |
| Browser | The token never reaches page JavaScript: the Next.js BFF keeps it in an `HttpOnly`, `SameSite=Lax` cookie (`Secure` when `SESSION_COOKIE_SECURE=true`) and rejects cross-origin mutations |
| Authorisation | Roles (`viewer`, `inspector`, `ml_engineer`, `admin`) are enforced by the API on every route, not only hidden in the UI |
| Media | Images are served through HMAC-SHA256 signed URLs that expire after 600 s (`MEDIA_URL_TTL_SECONDS`); storage keys are never sent to the browser |
| Uploads | 8 MiB limit (`MAX_UPLOAD_BYTES`), decode and type validation, EXIF/GPS metadata stripped before storage, at most 8 uploads in flight (`MAX_INFLIGHT_UPLOADS`) with `429` + `Retry-After` beyond that |
| Releases | Promotion only through the API: gates passed, bundle file hashes verified, warm-up on golden images, expected-generation compare-and-swap, audited |
| Copilot | Input guardrails (prompt injection, off-topic, PII redaction) and output guardrails (no safety certification, triage consistency, numeric grounding against tool outputs, system-prompt leak) |
| Secrets | `make env` generates secrets locally into `.env`, which is git-ignored; CI generates throw-away secrets per run |
| Audit | Promotions, rollbacks, reviews and alert transitions are recorded with actor, reason and correlation ID |
| Cameras | Stream URLs (which may embed camera credentials) are encrypted at rest (Fernet, key derived from `JWT_SECRET`) and only ever returned masked (`rtsp://user:•••@host`); only `rtsp`, `rtsps`, `http` and `https` are accepted (`file://` only when `CAMERA_ALLOW_FILE_URLS=true` for tests); adding, changing and removing cameras needs the ML-engineer role and is audited |

## Known limitations

- **No login throttling or lockout.** Put a rate-limiting reverse proxy in front of any shared deployment.
- **Local stack is plain HTTP.** Terminate TLS in front of the web app and set `SESSION_COOKIE_SECURE=true`.
- **Demo accounts share one password** (`DEMO_PASSWORD` in `.env`). Change or remove them before sharing a
  deployment.
- `JWT_SECRET` signs sessions and media URLs and derives the camera-URL encryption key; rotating it signs everyone
  out, invalidates open media links and means camera stream URLs must be entered again.
- A camera URL makes the connector (and the "test connection" call) open a network connection to that host. That
  is the point of the feature, but it also means an ML engineer can make the server reach internal addresses;
  restrict that role accordingly and run the connector on a network segment that can only see the cameras.
- LLM provider credentials (`LLM_API_KEY`) live in `.env`; prompts sent to an external provider leave your network.
  PII is redacted first, but review your provider's data policy.
- With `VISION_CHECK_MODEL` set, the inference worker sends the canonical tyre image (metadata stripped, JPEG, long
  side ≤ 768 px) of every tyre YOLO would auto-clear to that provider. Leave it empty to keep images on-premises;
  auto-clears then rely on YOLO alone.
