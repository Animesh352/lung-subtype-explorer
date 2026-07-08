from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Sample(Base):
    __tablename__ = "samples"

    sample_id: Mapped[str] = mapped_column(Text, primary_key=True)
    cohort: Mapped[str] = mapped_column(Text, nullable=False)
    sample_type: Mapped[str] = mapped_column(Text, nullable=False)
    subtype: Mapped[str | None] = mapped_column(Text, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)


class Gene(Base):
    __tablename__ = "genes"

    gene_id: Mapped[str] = mapped_column(Text, primary_key=True)
    gene_symbol: Mapped[str] = mapped_column(Text, nullable=False, index=True)


class Expression(Base):
    __tablename__ = "expression"

    sample_id: Mapped[str] = mapped_column(Text, ForeignKey("samples.sample_id"), primary_key=True)
    gene_id: Mapped[str] = mapped_column(Text, ForeignKey("genes.gene_id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_expression_gene_sample", "gene_id", "sample_id"),)


class DEResult(Base):
    __tablename__ = "de_results"

    gene_symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    log_fc: Mapped[float] = mapped_column(Float, nullable=False)
    ave_expr: Mapped[float] = mapped_column(Float, nullable=False)
    t_stat: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    adj_p_val: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_de_results_adj_p_val", "adj_p_val"),)


class TopGene(Base):
    __tablename__ = "top_genes"

    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    gene_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    mean_shap: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)


class GeneAnnotation(Base):
    __tablename__ = "gene_annotations"

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    found: Mapped[bool] = mapped_column(Boolean, nullable=False)
    entrez_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ensembl_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    gene_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    uniprot_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
