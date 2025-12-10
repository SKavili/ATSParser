"""Initial resume_metadata table

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'resume_metadata',
        sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('candidatename', mysql.VARCHAR(length=255), nullable=True),
        sa.Column('jobrole', mysql.VARCHAR(length=255), nullable=True),
        sa.Column('experience', mysql.VARCHAR(length=100), nullable=True),
        sa.Column('domain', mysql.VARCHAR(length=255), nullable=True),
        sa.Column('mobile', mysql.VARCHAR(length=50), nullable=True),
        sa.Column('email', mysql.VARCHAR(length=255), nullable=True),
        sa.Column('education', mysql.TEXT(), nullable=True),
        sa.Column('filename', mysql.VARCHAR(length=512), nullable=False),
        sa.Column('skillset', mysql.TEXT(), nullable=True),
        sa.Column('created_at', mysql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', mysql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('resume_metadata')

