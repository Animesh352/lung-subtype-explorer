"""add top_genes

Revision ID: c5d6e7f8a9b0
Revises: b4e7c2a9f031
Create Date: 2026-07-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "b4e7c2a9f031"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "top_genes",
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("gene_symbol", sa.Text(), nullable=False),
        sa.Column("mean_shap", sa.Float(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("rank"),
    )


def downgrade() -> None:
    op.drop_table("top_genes")
