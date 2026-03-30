from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("data/gse269951/candidate_expression")
OUT_DIR = BASE_DIR / "screening"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDE_PATH = BASE_DIR / "dmel_candidate_expression_wide.tsv"
LONG_PATH = BASE_DIR / "dmel_candidate_expression_long.tsv"
SAMPLE_SHEET_PATH = BASE_DIR / "dmel_sample_sheet.tsv"


# ============================================================
# Plot style
# ============================================================

sns.set_theme(style="white", context="notebook")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Load data
# ============================================================

def load_inputs():
    wide = pd.read_csv(WIDE_PATH, sep="\t", dtype=str)
    long = pd.read_csv(LONG_PATH, sep="\t", dtype=str)
    sample_sheet = pd.read_csv(SAMPLE_SHEET_PATH, sep="\t", dtype=str)

    # numeric expression
    long["vst_expression"] = pd.to_numeric(long["vst_expression"], errors="coerce")
    sample_sheet["stage_sort_key"] = pd.to_numeric(sample_sheet["stage_sort_key"], errors="coerce")

    return wide, long, sample_sheet


# ============================================================
# Build stage-level matrix
# ============================================================

def build_stage_means(long_df: pd.DataFrame, sample_sheet: pd.DataFrame):
    stage_order = (
        sample_sheet[["exact_stage_label", "stage_sort_key"]]
        .drop_duplicates()
        .sort_values("stage_sort_key")
    )
    ordered_stages = stage_order["exact_stage_label"].tolist()

    gene_info = (
        long_df[["FlyBase_FBgn", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]]
        .drop_duplicates(subset=["FlyBase_FBgn"])
        .copy()
    )

    stage_means = (
        long_df.groupby(
            ["FlyBase_FBgn", "exact_stage_label"],
            dropna=False
        )["vst_expression"]
        .mean()
        .reset_index()
    )

    stage_wide = stage_means.pivot(
        index="FlyBase_FBgn",
        columns="exact_stage_label",
        values="vst_expression"
    ).reset_index()

    stage_wide.columns.name = None

    stage_cols_present = [c for c in ordered_stages if c in stage_wide.columns]
    stage_wide = stage_wide[["FlyBase_FBgn"] + stage_cols_present].copy()

    stage_wide = gene_info.merge(stage_wide, on="FlyBase_FBgn", how="inner")

    meta_cols = ["FlyBase_FBgn", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]
    stage_wide = stage_wide[meta_cols + stage_cols_present].copy()

    return gene_info, stage_means, stage_wide, stage_cols_present


# ============================================================
# Z-score per gene
# ============================================================

def row_zscore(df: pd.DataFrame, value_cols):
    x = df[value_cols].astype(float).copy()
    means = x.mean(axis=1)
    stds = x.std(axis=1, ddof=0).replace(0, np.nan)
    z = x.sub(means, axis=0).div(stds, axis=0).fillna(0.0)
    return z


# ============================================================
# Stage summary
# ============================================================

def save_stage_summary(stage_wide: pd.DataFrame, stage_cols):
    expr = stage_wide[stage_cols].astype(float)

    summary = pd.DataFrame({
        "stage": stage_cols,
        "mean_expression_across_candidates": expr.mean(axis=0).values,
        "median_expression_across_candidates": expr.median(axis=0).values,
        "sd_expression_across_candidates": expr.std(axis=0, ddof=0).values,
    })

    summary.to_csv(OUT_DIR / "candidate_stage_summary.tsv", sep="\t", index=False)
    return summary


# ============================================================
# Heatmap
# ============================================================

def make_heatmap(stage_wide: pd.DataFrame, stage_cols):
    z = row_zscore(stage_wide, stage_cols)

    row_labels = stage_wide["DB_Object_Symbol"].fillna(stage_wide["FlyBase_FBgn"]).tolist()

    heatmap_df = z.copy()
    heatmap_df.index = row_labels
    heatmap_df.columns = stage_cols

    # cluster genes, keep stages in biological order
    g = sns.clustermap(
        heatmap_df,
        row_cluster=True,
        col_cluster=False,
        cmap="vlag",
        center=0,
        xticklabels=True,
        yticklabels=False,
        figsize=(10, 12),
        cbar_kws={"label": "Gene-wise z-score"},
        dendrogram_ratio=(0.14, 0.05),
    )

    g.ax_heatmap.set_title("Candidate splicing-factor expression across developmental stages", pad=12)
    g.ax_heatmap.set_xlabel("Stage")
    g.ax_heatmap.set_ylabel("Genes")

    plt.savefig(OUT_DIR / "candidate_stage_heatmap.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "candidate_stage_heatmap.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Gene trajectory clustering
# ============================================================

def cluster_genes(stage_wide: pd.DataFrame, stage_cols, n_clusters=6):
    z = row_zscore(stage_wide, stage_cols)

    # hierarchical clustering on stage-averaged gene profiles
    distances = pdist(z.values, metric="euclidean")
    Z = linkage(distances, method="average")
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

    cluster_table = stage_wide[["FlyBase_FBgn", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]].copy()
    cluster_table["cluster_id"] = cluster_ids

    z_with_clusters = z.copy()
    z_with_clusters["cluster_id"] = cluster_ids
    z_with_clusters["FlyBase_FBgn"] = stage_wide["FlyBase_FBgn"].values
    z_with_clusters["DB_Object_Symbol"] = stage_wide["DB_Object_Symbol"].values

    cluster_table.to_csv(OUT_DIR / "candidate_cluster_assignments.tsv", sep="\t", index=False)
    return cluster_table, z_with_clusters


# ============================================================
# Cluster mean profiles
# ============================================================

def plot_cluster_profiles(z_with_clusters: pd.DataFrame, stage_cols):
    profile_table = (
        z_with_clusters.groupby("cluster_id")[stage_cols]
        .mean()
        .reset_index()
        .sort_values("cluster_id")
    )

    profile_table.to_csv(OUT_DIR / "candidate_cluster_mean_profiles.tsv", sep="\t", index=False)

    long_profiles = profile_table.melt(
        id_vars="cluster_id",
        value_vars=stage_cols,
        var_name="stage",
        value_name="mean_zscore"
    )

    plt.figure(figsize=(10, 5.5))
    sns.lineplot(
        data=long_profiles,
        x="stage",
        y="mean_zscore",
        hue="cluster_id",
        marker="o",
        linewidth=2
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Mean gene-wise z-score")
    plt.title("Mean developmental expression profiles of candidate clusters")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "candidate_cluster_profiles.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "candidate_cluster_profiles.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Optional: top dynamic candidates
# ============================================================

def save_top_dynamic_candidates(stage_wide: pd.DataFrame, stage_cols, top_n=40):
    expr = stage_wide[stage_cols].astype(float)
    stage_range = expr.max(axis=1) - expr.min(axis=1)

    dynamic = stage_wide[["FlyBase_FBgn", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]].copy()
    dynamic["expression_range"] = stage_range.values
    dynamic = dynamic.sort_values("expression_range", ascending=False)

    dynamic.to_csv(OUT_DIR / "candidate_dynamic_ranking.tsv", sep="\t", index=False)
    dynamic.head(top_n).to_csv(OUT_DIR / "candidate_top_dynamic_genes.tsv", sep="\t", index=False)




########################

def plot_stage_expression_distributions(long_df: pd.DataFrame, sample_sheet: pd.DataFrame):
    stage_order = (
        sample_sheet[["exact_stage_label", "stage_sort_key"]]
        .drop_duplicates()
        .sort_values("stage_sort_key")
    )
    ordered_stages = stage_order["exact_stage_label"].tolist()

    plot_df = long_df.copy()
    plot_df["exact_stage_label"] = pd.Categorical(
        plot_df["exact_stage_label"],
        categories=ordered_stages,
        ordered=True
    )

    plt.figure(figsize=(11, 5.5))
    sns.boxplot(
        data=plot_df,
        x="exact_stage_label",
        y="vst_expression",
        color="#AFC3DD",
        fliersize=1.5
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("VST expression")
    plt.title("Distribution of candidate-gene expression across developmental stages")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "candidate_stage_expression_distributions.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "candidate_stage_expression_distributions.pdf", bbox_inches="tight")
    plt.close()


#############

def save_top_pupal_candidates(stage_wide: pd.DataFrame, stage_cols, top_n=40):
    pupal_cols = [c for c in stage_cols if c.startswith("pupae_")]
    tmp = stage_wide.copy()
    tmp["mean_pupal_expression"] = tmp[pupal_cols].astype(float).mean(axis=1)
    tmp = tmp.sort_values("mean_pupal_expression", ascending=False)

    tmp.to_csv(OUT_DIR / "candidate_ranked_by_mean_pupal_expression.tsv", sep="\t", index=False)
    tmp.head(top_n).to_csv(OUT_DIR / "candidate_top_pupal_genes.tsv", sep="\t", index=False)

##########

def plot_top_pupal_heatmap(stage_wide: pd.DataFrame, stage_cols, top_n=40):
    pupal_cols = [c for c in stage_cols if c.startswith("pupae_")]

    tmp = stage_wide.copy()
    tmp["mean_pupal_expression"] = tmp[pupal_cols].astype(float).mean(axis=1)
    tmp = tmp.sort_values("mean_pupal_expression", ascending=False).head(top_n).copy()

    plot_df = tmp[["DB_Object_Symbol"] + stage_cols].copy()
    plot_df = plot_df.set_index("DB_Object_Symbol")
    plot_df = plot_df.astype(float)

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        plot_df,
        cmap="mako",
        cbar_kws={"label": "Mean VST expression"}
    )
    plt.xlabel("Stage")
    plt.ylabel("Top pupal-expressed candidate genes")
    plt.title("Top candidate splicing factors ranked by mean pupal expression")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_pupal_expression_heatmap.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "top_pupal_expression_heatmap.pdf", bbox_inches="tight")
    plt.close()

#########

def plot_top_dynamic_gene_lines(stage_wide: pd.DataFrame, stage_cols, top_n=12):
    expr = stage_wide[stage_cols].astype(float)
    stage_range = expr.max(axis=1) - expr.min(axis=1)

    tmp = stage_wide.copy()
    tmp["expression_range"] = stage_range.values
    tmp = tmp.sort_values("expression_range", ascending=False).head(top_n).copy()

    long_plot = tmp.melt(
        id_vars=["FlyBase_FBgn", "DB_Object_Symbol"],
        value_vars=stage_cols,
        var_name="stage",
        value_name="expression"
    )
    long_plot["expression"] = pd.to_numeric(long_plot["expression"], errors="coerce")

    plt.figure(figsize=(10.5, 5.8))
    sns.lineplot(
        data=long_plot,
        x="stage",
        y="expression",
        hue="DB_Object_Symbol",
        marker="o",
        linewidth=2
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Mean VST expression")
    plt.title("Top dynamically expressed candidate splicing factors")
    plt.legend(title="Gene", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_dynamic_gene_lines.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "top_dynamic_gene_lines.pdf", bbox_inches="tight")
    plt.close()

###########

def plot_go_bin_stage_summary(stage_wide: pd.DataFrame, stage_cols):
    long_plot = stage_wide.melt(
        id_vars=["FlyBase_FBgn", "DB_Object_Symbol", "go_bin"],
        value_vars=stage_cols,
        var_name="stage",
        value_name="expression"
    )
    long_plot["expression"] = pd.to_numeric(long_plot["expression"], errors="coerce")

    summary = (
        long_plot.groupby(["go_bin", "stage"], dropna=False)["expression"]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "go_bin_stage_mean_expression.tsv", sep="\t", index=False)

    plt.figure(figsize=(10, 5.2))
    sns.lineplot(
        data=summary,
        x="stage",
        y="expression",
        hue="go_bin",
        marker="o",
        linewidth=2.2
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Stage")
    plt.ylabel("Mean VST expression")
    plt.title("Mean candidate expression across stages by GO-derived category")
    plt.legend(title="GO bin", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "go_bin_stage_summary.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "go_bin_stage_summary.pdf", bbox_inches="tight")
    plt.close()

############

def plot_cluster_sizes(cluster_table: pd.DataFrame):
    counts = (
        cluster_table["cluster_id"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["cluster_id", "n_genes"]
    counts.to_csv(OUT_DIR / "candidate_cluster_sizes.tsv", sep="\t", index=False)

    plt.figure(figsize=(6.5, 4.5))
    sns.barplot(data=counts, x="cluster_id", y="n_genes", color="#4C78A8")
    plt.xlabel("Cluster")
    plt.ylabel("Number of genes")
    plt.title("Sizes of candidate expression clusters")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "candidate_cluster_sizes.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / "candidate_cluster_sizes.pdf", bbox_inches="tight")
    plt.close()



# ============================================================
# Main
# ============================================================

def main():
    wide, long_df, sample_sheet = load_inputs()
    gene_info, stage_means, stage_wide, stage_cols = build_stage_means(long_df, sample_sheet)

    stage_means.to_csv(OUT_DIR / "candidate_stage_means_long.tsv", sep="\t", index=False)
    stage_wide.to_csv(OUT_DIR / "candidate_stage_means_wide.tsv", sep="\t", index=False)

    save_stage_summary(stage_wide, stage_cols)
    make_heatmap(stage_wide, stage_cols)

    cluster_table, z_with_clusters = cluster_genes(stage_wide, stage_cols, n_clusters=6)
    plot_cluster_profiles(z_with_clusters, stage_cols)

    save_top_dynamic_candidates(stage_wide, stage_cols, top_n=40)

    plot_stage_expression_distributions(long_df, sample_sheet)
    save_top_pupal_candidates(stage_wide, stage_cols, top_n=40)
    plot_top_pupal_heatmap(stage_wide, stage_cols, top_n=40)
    plot_top_dynamic_gene_lines(stage_wide, stage_cols, top_n=12)
    plot_go_bin_stage_summary(stage_wide, stage_cols)
    plot_cluster_sizes(cluster_table)

    report_lines = [
        f"Candidate genes in stage-level table: {len(stage_wide)}",
        f"Number of developmental stages used: {len(stage_cols)}",
        f"Stages: {', '.join(stage_cols)}",
        "Outputs:",
        "- candidate_stage_means_long.tsv",
        "- candidate_stage_means_wide.tsv",
        "- candidate_stage_summary.tsv",
        "- candidate_stage_heatmap.png/.pdf",
        "- candidate_cluster_assignments.tsv",
        "- candidate_cluster_mean_profiles.tsv",
        "- candidate_cluster_profiles.png/.pdf",
        "- candidate_dynamic_ranking.tsv",
        "- candidate_top_dynamic_genes.tsv",
        "- candidate_stage_expression_distributions.png/.pdf",
        "- candidate_ranked_by_mean_pupal_expression.tsv",
        "- candidate_top_pupal_genes.tsv",
        "- top_pupal_expression_heatmap.png/.pdf",
        "- top_dynamic_gene_lines.png/.pdf",
        "- go_bin_stage_mean_expression.tsv",
        "- go_bin_stage_summary.png/.pdf",
        "- candidate_cluster_sizes.tsv",
        "- candidate_cluster_sizes.png/.pdf",
    ]
    (OUT_DIR / "screening_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("[DONE]")
    print(f"Saved outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()