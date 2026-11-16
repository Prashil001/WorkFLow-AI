"""scope document_chunks by corpus and record source urls

Revision ID: e7b2c4a91f53
Revises: c3f1a9d27e10
Create Date: 2026-11-16 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7b2c4a91f53'
down_revision: Union[str, Sequence[str], None] = 'c3f1a9d27e10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rows written before corpora existed all came from the pricing mock data.
    op.add_column(
        "document_chunks",
        sa.Column(
            "corpus",
            sa.String(length=64),
            nullable=False,
            server_default="pricing-mock",
        ),
    )
    op.alter_column("document_chunks", "corpus", server_default=None)
    op.add_column(
        "document_chunks",
        sa.Column("source_url", sa.Text(), nullable=True),
    )
    op.drop_constraint(
        "uq_document_chunks_source_chunk",
        "document_chunks",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_document_chunks_corpus_source_chunk",
        "document_chunks",
        ["corpus", "source_id", "chunk_index"],
    )


def downgrade() -> None:
    connection = op.get_bind()
    other_corpora = connection.execute(sa.text(
        "SELECT COUNT(*) FROM document_chunks WHERE corpus <> 'pricing-mock'"
    )).scalar_one()
    if other_corpora > 0:
        raise RuntimeError(
            "Cannot downgrade while chunks from corpora other than "
            "pricing-mock exist; delete them first"
        )

    op.drop_constraint(
        "uq_document_chunks_corpus_source_chunk",
        "document_chunks",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_document_chunks_source_chunk",
        "document_chunks",
        ["source_id", "chunk_index"],
    )
    op.drop_column("document_chunks", "source_url")
    op.drop_column("document_chunks", "corpus")
