"""Add AI search tables (ai_search_queries and ai_search_results)

Revision ID: 006
Revises: 005
Create Date: 2025-01-20 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ai_search_queries table
    op.create_table(
        'ai_search_queries',
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.current_timestamp(), nullable=True),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create ai_search_results table
    op.create_table(
        'ai_search_results',
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column('search_query_id', sa.BigInteger(), nullable=False),
        sa.Column('results_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.current_timestamp(), nullable=True),
        sa.ForeignKeyConstraint(['search_query_id'], ['ai_search_queries.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index on search_query_id for faster lookups
    op.create_index(
        'ix_ai_search_results_search_query_id',
        'ai_search_results',
        ['search_query_id']
    )


def downgrade() -> None:
    # Drop tables in reverse order (due to foreign key)
    op.drop_index('ix_ai_search_results_search_query_id', table_name='ai_search_results')
    op.drop_table('ai_search_results')
    op.drop_table('ai_search_queries')
