#!/usr/bin/env Rscript
# Differential expression: LUAD vs LUSC using limma
#
# Values are log2(normalized_count + 1) -- continuous; use lmFit + eBayes.
# DESeq2 / edgeR require raw integer counts and must NOT be used here.
#
# Usage:
#   Rscript differential_expression.R <expr.parquet> <meta.parquet> <output.csv>
#
# Inputs
#   expr.parquet -- samples x genes float32 matrix (ETL output)
#   meta.parquet -- sample metadata with 'cohort' column (LUAD / LUSC)
#
# Output
#   output.csv   -- topTable results sorted by adj.P.Val

suppressPackageStartupMessages({
  library(arrow)
  library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript differential_expression.R <expr.parquet> <meta.parquet> <output.csv>")
}

expr_path   <- args[1]
meta_path   <- args[2]
output_path <- args[3]

# ---------------------------------------------------------------------------
# 1. Load expression (samples x genes) and transpose to genes x samples
# ---------------------------------------------------------------------------

message("Reading expression matrix: ", expr_path)
expr_df    <- arrow::read_parquet(expr_path)
sample_ids <- expr_df[["sample_id"]]
gene_names <- setdiff(names(expr_df), "sample_id")
expr_mat   <- t(as.matrix(expr_df[, gene_names]))
colnames(expr_mat) <- sample_ids
rownames(expr_mat) <- gene_names
storage.mode(expr_mat) <- "double"
message(sprintf("  %d genes x %d samples", nrow(expr_mat), ncol(expr_mat)))

# ---------------------------------------------------------------------------
# 2. Load sample metadata and align with expression column order
# ---------------------------------------------------------------------------

message("Reading sample metadata: ", meta_path)
meta_df <- arrow::read_parquet(meta_path)
idx     <- match(sample_ids, meta_df[["sample_id"]])
if (any(is.na(idx))) {
  stop("Some expression sample IDs not found in metadata.")
}
meta_df <- meta_df[idx, ]
stopifnot(identical(meta_df[["sample_id"]], sample_ids))
message(sprintf("  %d samples (%d LUAD, %d LUSC)",
  nrow(meta_df),
  sum(meta_df[["cohort"]] == "LUAD"),
  sum(meta_df[["cohort"]] == "LUSC")))

# ---------------------------------------------------------------------------
# 3. Design matrix: ~0 + cohort gives explicit LUAD and LUSC columns
# ---------------------------------------------------------------------------

cohort <- factor(meta_df[["cohort"]])
design <- model.matrix(~0 + cohort)
colnames(design) <- levels(cohort)   # "LUAD", "LUSC"

# ---------------------------------------------------------------------------
# 4. Fit: lmFit + LUAD - LUSC contrast + eBayes(trend=TRUE, robust=TRUE)
# ---------------------------------------------------------------------------

message("Fitting limma model...")
fit         <- lmFit(expr_mat, design)
cont_matrix <- makeContrasts(LUADvsLUSC = LUAD - LUSC, levels = design)
fit2        <- contrasts.fit(fit, cont_matrix)
fit2        <- eBayes(fit2, trend = TRUE, robust = TRUE)

# ---------------------------------------------------------------------------
# 5. Extract all genes sorted by adj.P.Val
# ---------------------------------------------------------------------------

tt <- topTable(fit2, coef = "LUADvsLUSC", number = Inf, sort.by = "p")

results <- data.frame(
  gene_symbol = rownames(tt),
  logFC       = tt$logFC,
  AveExpr     = tt$AveExpr,
  t           = tt$t,
  P.Value     = tt$P.Value,
  adj.P.Val   = tt$adj.P.Val,
  stringsAsFactors = FALSE
)

n_sig <- sum(results$adj.P.Val < 0.05)
message(sprintf("Significant genes (adj.P.Val < 0.05): %d / %d", n_sig, nrow(results)))

# ---------------------------------------------------------------------------
# 6. Write output
# ---------------------------------------------------------------------------

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(results, output_path, row.names = FALSE)
message("DE results written to: ", output_path)
