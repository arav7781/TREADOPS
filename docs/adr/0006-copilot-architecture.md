# ADR 0006 - Guarded multimodal copilot on LangGraph with in-house LLMOps tracing

Status: accepted (2026-10-22)

## Decision
- The copilot is a LangGraph state machine (`guard_input -> plan <-> tools -> guard_output -> suggest`) that
  talks to any OpenAI-compatible endpoint through the project's `LLMClient` (tool calling, usage incl. cached
  tokens).
- Classification is never done by the LLM. `classify_tyre_image` submits a real inspection job, so the YOLO
  bundle pinned by the active deployment decides and full lineage is stored; the chat UI shows the same live
  stage images as the inspect page.
- Guardrails are deterministic and unit-tested: prompt-injection and off-topic blocks, PII redaction before the
  LLM, certification-claim rewriting, grounding of every figure against tool outputs, replacement of answers
  that contradict the YOLO triage, system-prompt leak blocking, and a mandatory inspector disclaimer.
- Without `LLM_BASE_URL`/`LLM_MODEL`, or when an LLM call fails, a deterministic tool-grounded planner answers
  and the trace is labelled `deterministic_fallback`.
- Traces, spans, token usage, cost, guardrail events and feedback live in PostgreSQL next to the business data;
  prompts are versioned files whose SHA-256 is recorded per turn; an offline golden-set evaluation logs to MLflow.

## Consequences
No external tracing SaaS is required. Cost is reported only for models listed in `configs/copilot.yaml`.
