"""Add resume_text field to resume_metadata table

Revision ID: 004
Revises: 003
Create Date: 2025-01-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add resume_text column after status
    # Using Text type which maps to LONGTEXT in MySQL (up to 4GB)
    op.add_column(
        'resume_metadata',
        sa.Column('resume_text', sa.Text(), nullable=True, comment='Full extracted resume text')
    )


def downgrade() -> None:
    # Remove resume_text column
    op.drop_column('resume_metadata', 'resume_text')

