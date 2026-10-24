"""camera roi: the inspection zone of a line camera and whether its live view is annotated

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-24 23:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0007'
down_revision: str | None = '0006'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('cameras', sa.Column('roi', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('cameras', sa.Column('overlay', sa.Boolean(), server_default=sa.text('true'), nullable=False))


def downgrade() -> None:
    op.drop_column('cameras', 'overlay')
    op.drop_column('cameras', 'roi')
