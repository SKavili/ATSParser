"""Add designation field to resume_metadata table

Revision ID: 002
Revises: 001
Create Date: 2025-12-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add designation column after jobrole
    op.add_column(
        'resume_metadata',
        sa.Column('designation', mysql.VARCHAR(length=255), nullable=True, comment='Current or most recent job title')
    )


def downgrade() -> None:
    # Remove designation column
    op.drop_column('resume_metadata', 'designation')

