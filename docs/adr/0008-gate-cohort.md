# ADR 0008 - Gate cohort and locked-test discipline

Status: accepted (2026-10-22)

Temperature is fitted on the calibration role, the binary threshold is selected on the policy role, and gates
are evaluated on the outer-validation pool (calibration + policy), labelled "development evidence". The locked
test runs once per frozen candidate for final reporting; every access is appended to
`docs/evidence/test_access_log.jsonl` with purpose and checkpoint hash. Replay and copilot evaluation images come
from the policy role, never the test role.
