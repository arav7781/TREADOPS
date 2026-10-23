## What and why

<!-- What does this change do, and why is it needed? Link the issue if there is one. -->

## How it was verified

<!-- Commands you ran and what they showed. Paste measured numbers, not estimates. -->

## Checklist

- [ ] `make lint` and `make typecheck` pass
- [ ] Tests added or updated; the relevant suites pass (`make test-unit test-ml test-integration`, web `npm test`, `make test-e2e`)
- [ ] API schema changed → `make contracts` run and the regenerated files committed
- [ ] Copilot prompt/tools/guardrails changed → prompt version bumped and `make eval-copilot` passes
- [ ] DVC pipeline module or config changed → affected stages rerun and `dvc.lock` committed
- [ ] Lock file changed → `make licenses` run
- [ ] Every number in docs or UI comes from a stored run (`docs/evidence/`)
- [ ] No secrets, `.env` contents or personal data in the diff
- [ ] Screenshot attached for UI changes
