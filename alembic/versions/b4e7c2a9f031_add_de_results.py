"""add de_results

Revision ID: b4e7c2a9f031
Revises: e93ca3a97846
Create Date: 2026-07-04 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4e7c2a9f031"
down_revision: Union[str, None] = "e93ca3a97846"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "de_results",
        sa.Column("gene_symbol", sa.Text(), nullable=False),
        sa.Column("log_fc", sa.Float(), nullable=False),
        sa.Column("ave_expr", sa.Float(), nullable=False),
        sa.Column("t_stat", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("adj_p_val", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("gene_symbol"),
    )
    op.create_index("ix_de_results_adj_p_val", "de_results", ["adj_p_val"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_de_results_adj_p_val", table_name="de_results")
    op.drop_table("de_results")
