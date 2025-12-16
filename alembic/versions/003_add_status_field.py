"""Add status field to resume_metadata table

Revision ID: 003
Revises: 002
Create Date: 2025-12-15 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add status column after skillset
    op.add_column(
        'resume_metadata',
        sa.Column('status', mysql.VARCHAR(length=50), nullable=True, server_default='pending', comment='Processing status: pending, processing, completed, failed:reason')
    )


def downgrade() -> None:
    # Remove status column
    op.drop_column('resume_metadata', 'status')



