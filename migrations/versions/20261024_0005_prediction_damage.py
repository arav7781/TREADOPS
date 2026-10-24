"""prediction damage: boxes from the damage locator and whether they escalated an auto-clear

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-24 17:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('predictions', sa.Column('damage', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('predictions', 'damage')
