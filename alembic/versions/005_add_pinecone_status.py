"""Add pinecone_status field to resume_metadata table

Revision ID: 005
Revises: 004
Create Date: 2025-01-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add pinecone_status column after resume_text
    # Using Integer: 0 = not indexed, 1 = indexed
    op.add_column(
        'resume_metadata',
        sa.Column('pinecone_status', sa.Integer(), nullable=True, server_default='0', comment='Pinecone indexing status: 0 = not indexed, 1 = indexed')
    )


def downgrade() -> None:
    # Remove pinecone_status column
    op.drop_column('resume_metadata', 'pinecone_status')

