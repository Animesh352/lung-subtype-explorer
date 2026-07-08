"""add gene_annotations

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-07-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d6e7f8a9b0c1"
down_revision: Union[str, None] = "c5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gene_annotations",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("found", sa.Boolean(), nullable=False),
        sa.Column("entrez_id", sa.Integer(), nullable=True),
        sa.Column("ensembl_id", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("gene_type", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("uniprot_id", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )
    op.create_index("ix_gene_annotations_entrez_id", "gene_annotations", ["entrez_id"])


def downgrade() -> None:
    op.drop_index("ix_gene_annotations_entrez_id", table_name="gene_annotations")
    op.drop_table("gene_annotations")
