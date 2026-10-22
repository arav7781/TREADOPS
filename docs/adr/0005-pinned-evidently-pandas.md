# ADR 0005 - pandas < 3 for the pinned Evidently adapter

Status: accepted (2026-10-22)

Evidently 0.7.23 imports `pandas.core.dtypes.common.infer_dtype_from_object`, which pandas 3.0 removed; the
adapter failed with ImportError under pandas 3.1.0. pandas is constrained to `>=2.2,<3` (resolved 2.3.3). The
monitoring adapter refuses to run against any Evidently version other than the pinned one and records an
explicit `skipped`/`error` status instead of failing the window.
