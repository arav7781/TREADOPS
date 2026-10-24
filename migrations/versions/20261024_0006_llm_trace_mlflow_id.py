"""llm trace mlflow id: the MLflow Tracing trace that mirrors each LLMOps trace

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-24 21:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0006'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('llm_traces', sa.Column('mlflow_trace_id', sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column('llm_traces', 'mlflow_trace_id')
