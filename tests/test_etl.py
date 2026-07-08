"""Tests for pure ETL functions: barcode parsing and variance filtering."""

import numpy as np
import pandas as pd

from pipeline.etl import build_sample_meta, filter_low_variance, parse_sample_type

# ---------------------------------------------------------------------------
# parse_sample_type
# ---------------------------------------------------------------------------


class TestParseSampleType:
    def test_primary_tumor(self):
        assert parse_sample_type("TCGA-05-4244-01A-01R-1107-07") == "primary_tumor"

    def test_primary_tumor_short_form(self):
        # Column headers in HiSeqV2 use 15-char barcodes ending at the type code
        assert parse_sample_type("TCGA-05-4244-01") == "primary_tumor"

    def test_solid_tissue_normal(self):
        assert parse_sample_type("TCGA-05-4244-11A-01R-1107-07") == "solid_tissue_normal"

    def test_solid_tissue_normal_short(self):
        assert parse_sample_type("TCGA-05-4244-11") == "solid_tissue_normal"

    def test_recurrent_tumor(self):
        assert parse_sample_type("TCGA-50-5066-02") == "recurrent_tumor"

    def test_metastatic(self):
        assert parse_sample_type("TCGA-05-4244-06") == "metastatic"

    def test_additional_metastatic(self):
        assert parse_sample_type("TCGA-D3-A1QA-07") == "additional_metastatic"

    def test_blood_derived_normal(self):
        assert parse_sample_type("TCGA-05-4244-10") == "blood_derived_normal"

    def test_unknown_code_returns_other_prefix(self):
        result = parse_sample_type("TCGA-05-4244-99")
        assert result.startswith("other_")
        assert "99" in result

    def test_malformed_barcode_too_few_segments(self):
        assert parse_sample_type("TCGA-05-4244") == "unknown"

    def test_empty_string(self):
        assert parse_sample_type("") == "unknown"

    def test_positions_14_15_are_code(self):
        # Verify the extraction matches positions 14-15 (1-indexed) in the full string
        barcode = "TCGA-05-4244-01A"
        # Position 14 (1-indexed) == index 13 (0-indexed) == '0'
        # Position 15 (1-indexed) == index 14 (0-indexed) == '1'
        assert barcode[13:15] == "01"
        assert parse_sample_type(barcode) == "primary_tumor"


# ---------------------------------------------------------------------------
# filter_low_variance
# ---------------------------------------------------------------------------


def _make_expr(data: dict[str, list[float]]) -> pd.DataFrame:
    samples = [f"TCGA-XX-000{i}-01" for i in range(len(next(iter(data.values()))))]
    return pd.DataFrame(data, index=samples)


class TestFilterLowVariance:
    def test_zero_variance_gene_dropped(self):
        df = _make_expr(
            {
                "FLAT": [5.0, 5.0, 5.0, 5.0, 5.0],
                "VARIABLE": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = filter_low_variance(df, percentile=10.0)
        assert "VARIABLE" in result.columns
        # FLAT has zero variance; even at p10 it sits below any positive-variance gene
        assert "FLAT" not in result.columns or result["FLAT"].var() > 0

    def test_high_variance_genes_retained(self):
        df = _make_expr(
            {
                "HIGH1": [1.0, 5.0, 3.0, 8.0, 2.0],
                "HIGH2": [9.0, 1.0, 7.0, 2.0, 6.0],
                "LOW": [1.0, 1.0, 1.01, 1.0, 1.0],
            }
        )
        result = filter_low_variance(df, percentile=50.0)
        assert "HIGH1" in result.columns
        assert "HIGH2" in result.columns

    def test_bottom_percentile_fraction_dropped(self):
        np.random.seed(42)
        n_genes = 100
        n_samples = 20
        # Give each gene a variance proportional to its index so we know exactly
        # which ones should be dropped at p10
        data = {f"G{i:03d}": np.random.randn(n_samples) * (i + 1) for i in range(n_genes)}
        df = _make_expr(data)

        result = filter_low_variance(df, percentile=10.0)
        # At least 90 % should remain (a few ties at the threshold may vary by 1)
        assert len(result.columns) >= n_genes * 0.89

    def test_all_genes_retained_at_zero_percentile(self):
        df = _make_expr(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [4.0, 5.0, 6.0],
                "C": [7.0, 8.0, 9.0],
            }
        )
        result = filter_low_variance(df, percentile=0.0)
        assert set(result.columns) == {"A", "B", "C"}

    def test_output_shape_rows_unchanged(self):
        df = _make_expr(
            {
                "G1": [1.0, 2.0, 3.0, 4.0],
                "G2": [4.0, 3.0, 2.0, 1.0],
                "G3": [0.0, 0.0, 0.0, 0.0],
            }
        )
        result = filter_low_variance(df, percentile=50.0)
        assert len(result) == len(df)

    def test_index_preserved(self):
        df = _make_expr({"GENE_A": [1.0, 2.0, 3.0], "GENE_B": [10.0, 0.0, 5.0]})
        result = filter_low_variance(df, percentile=0.0)
        assert list(result.index) == list(df.index)

    def test_filter_logs_dropped_count(self, caplog):
        import logging

        df = _make_expr(
            {
                "CONST": [1.0, 1.0, 1.0, 1.0],
                "VAR": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with caplog.at_level(logging.INFO, logger="pipeline.etl"):
            filter_low_variance(df, percentile=50.0)

        assert any("dropping" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# build_sample_meta
# ---------------------------------------------------------------------------


def _make_phenotype(samples: list[str], disease: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"_primary_disease": [disease] * len(samples)},
        index=pd.Index(samples, name="sample_id"),
    )


class TestBuildSampleMeta:
    def test_cohort_labels_correct(self):
        luad = ["TCGA-AA-0001-01", "TCGA-AA-0002-01"]
        lusc = ["TCGA-BB-0001-01"]
        pheno = _make_phenotype(luad + lusc, "lung adenocarcinoma")
        meta = build_sample_meta(luad, lusc, pheno)

        assert list(meta.loc[luad, "cohort"]) == ["LUAD", "LUAD"]
        assert meta.loc["TCGA-BB-0001-01", "cohort"] == "LUSC"

    def test_sample_type_derived_from_barcode(self):
        luad = ["TCGA-AA-0001-01", "TCGA-AA-0002-11"]
        pheno = _make_phenotype(luad, "lung adenocarcinoma")
        meta = build_sample_meta(luad, [], pheno)

        assert meta.loc["TCGA-AA-0001-01", "sample_type"] == "primary_tumor"
        assert meta.loc["TCGA-AA-0002-11", "sample_type"] == "solid_tissue_normal"

    def test_subtype_populated_from_phenotype(self):
        luad = ["TCGA-AA-0001-01"]
        pheno = _make_phenotype(luad, "lung adenocarcinoma")
        meta = build_sample_meta(luad, [], pheno)

        assert meta.loc["TCGA-AA-0001-01", "subtype"] == "lung adenocarcinoma"

    def test_age_gender_stage_null_without_clinical(self):
        luad = ["TCGA-AA-0001-01"]
        pheno = _make_phenotype(luad, "lung adenocarcinoma")
        meta = build_sample_meta(luad, [], pheno)

        for col in ("age", "gender", "stage"):
            assert pd.isna(meta.loc["TCGA-AA-0001-01", col])

    def test_age_populated_from_cohort_clinical(self):
        luad = ["TCGA-AA-0001-01"]
        pheno = _make_phenotype(luad, "lung adenocarcinoma")
        clinical = pd.DataFrame(
            {"age_at_initial_pathologic_diagnosis": [67]},
            index=pd.Index(luad, name="sample_id"),
        )
        meta = build_sample_meta(luad, [], pheno, luad_clinical=clinical)

        assert meta.loc["TCGA-AA-0001-01", "age"] == 67

    def test_index_is_sample_id(self):
        luad = ["TCGA-AA-0001-01", "TCGA-AA-0002-01"]
        pheno = _make_phenotype(luad, "lung adenocarcinoma")
        meta = build_sample_meta(luad, [], pheno)

        assert meta.index.name == "sample_id"
        assert set(meta.index) == set(luad)
