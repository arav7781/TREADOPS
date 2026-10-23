"""inspection rejection: images the subject gate refused (not a tyre), with no prediction

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-23 22:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('inspections', sa.Column('rejection', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('inspections', 'rejection')
