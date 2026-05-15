"""Add ats_pinecone_status_all to resume_metadata

Revision ID: 008
Revises: 007
Create Date: 2026-02-16

"""
from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resume_metadata",
        sa.Column(
            "ats_pinecone_status_all",
            sa.Integer(),
            nullable=True,
            server_default="0",
            comment="Pinecone ats index: 0 = not indexed, 1 = indexed",
        ),
    )


def downgrade() -> None:
    op.drop_column("resume_metadata", "ats_pinecone_status_all")
