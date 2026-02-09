"""Add location field to resume_metadata table

Revision ID: 007
Revises: 006
Create Date: 2025-02-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'resume_metadata',
        sa.Column('location', sa.String(255), nullable=True, comment='Candidate location (city, state, country)')
    )


def downgrade() -> None:
    op.drop_column('resume_metadata', 'location')
