from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

GO_GENES = DATA_DIR / "go_unique_genes.tsv"
EXP_GENES = DATA_DIR / "go_unique_genes_experimental.tsv"
MAPPING = DATA_DIR / "go_genes_fbgn_fbtr_fbpp.tsv"


# ============================================================
# Visual style
# ============================================================

BLUE = "#4C78A8"
BLUE_LIGHT = "#AFC3DD"
BLUE_DARK = "#2B5C8A"
SLATE = "#2F3440"
MID = "#69707A"
GRID = "#D9DEE7"
BG = "white"

PANEL_COLORS = {
    "BP": "#4C78A8",
    "CC": "#72B7B2",
    "Experimental": "#59A14F",
    "Protein": "#E15759",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": MID,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helpers
# ============================================================

def style_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(colors=SLATE)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    ax.title.set_color(SLATE)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.06, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=SLATE,
        va="bottom",
        ha="left"
    )


# ============================================================
# Load data
# ============================================================

def load_inputs():
    go_genes = pd.read_csv(GO_GENES, sep="\t", dtype=str)
    exp_genes = pd.read_csv(EXP_GENES, sep="\t", dtype=str)
    mapping = pd.read_csv(MAPPING, sep="\t", dtype=str)
    return go_genes, exp_genes, mapping


def build_gene_table(go_genes, exp_genes, mapping):
    genes = (
        go_genes[["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name"]]
        .drop_duplicates()
        .rename(columns={"DB_Object_ID": "FBgn"})
        .copy()
    )

    bp_ids = set(go_genes.loc[go_genes["go_bin"] == "BP_function", "DB_Object_ID"])
    cc_ids = set(go_genes.loc[go_genes["go_bin"] == "CC_component", "DB_Object_ID"])
    exp_ids = set(exp_genes["DB_Object_ID"])

    protein_ids = set(
        mapping.loc[
            mapping["FlyBase_FBpp"].notna() & (mapping["FlyBase_FBpp"] != ""),
            "FlyBase_FBgn"
        ]
    )

    transcript_counts = (
        mapping.loc[
            mapping["FlyBase_FBtr"].notna() & (mapping["FlyBase_FBtr"] != "")
        ]
        .groupby("FlyBase_FBgn")["FlyBase_FBtr"]
        .nunique()
        .to_dict()
    )

    genes["BP_support"] = genes["FBgn"].isin(bp_ids)
    genes["CC_support"] = genes["FBgn"].isin(cc_ids)
    genes["Experimental_support"] = genes["FBgn"].isin(exp_ids)
    genes["Has_protein_mapping"] = genes["FBgn"].isin(protein_ids)
    genes["n_transcripts"] = genes["FBgn"].map(transcript_counts).fillna(0).astype(int)

    return genes


# ============================================================
# Derived summaries
# ============================================================

def get_counts_series(genes):
    return pd.Series({
        "GO-derived unique genes": genes["FBgn"].nunique(),
        "BP-supported genes": int(genes["BP_support"].sum()),
        "CC-supported genes": int(genes["CC_support"].sum()),
        "Experimental genes": int(genes["Experimental_support"].sum()),
        "Genes with transcript mapping": int((genes["n_transcripts"] > 0).sum()),
        "Genes with protein mapping": int(genes["Has_protein_mapping"].sum()),
    }).sort_values()


def get_intersection_counts(genes):
    memberships = []

    for _, row in genes.iterrows():
        sets = []
        if row["BP_support"]:
            sets.append("BP")
        if row["CC_support"]:
            sets.append("CC")
        if row["Experimental_support"]:
            sets.append("Experimental")
        if row["Has_protein_mapping"]:
            sets.append("Protein")
        if sets:
            memberships.append(tuple(sets))

    counts = pd.Series(memberships).value_counts().sort_values(ascending=False)
    return counts


def get_top_isoform_table(genes, n=20):
    mapped = genes.loc[genes["n_transcripts"] > 0].copy()
    top = (
        mapped[["DB_Object_Symbol", "FBgn", "n_transcripts"]]
        .sort_values(["n_transcripts", "DB_Object_Symbol"], ascending=[False, True])
        .head(n)
        .sort_values(["n_transcripts", "DB_Object_Symbol"], ascending=[True, True])
    )
    return mapped, top


# ============================================================
# Panel drawing functions
# ============================================================

def draw_counts_panel(ax, genes):
    counts = get_counts_series(genes)

    bars = ax.barh(
        counts.index,
        counts.values,
        color=BLUE,
        edgecolor="none",
        height=0.72
    )

    style_axes(ax, grid_axis="x")
    ax.set_xlabel("Number of genes")
    ax.set_title("Summary of the GO-derived candidate set", pad=10)

    xmax = counts.max()
    ax.set_xlim(0, xmax * 1.14)

    for bar, value in zip(bars, counts.values):
        ax.text(
            value + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            color=SLATE,
            fontsize=9
        )


def draw_intersections_panel(ax, genes):
    intersection_counts = get_intersection_counts(genes)

    labels = [
        fill(" ∩ ".join(x), width=18)
        for x in intersection_counts.index
    ]

    colors = []
    for combo in intersection_counts.index:
        if "Experimental" in combo and "BP" in combo:
            colors.append(PANEL_COLORS["Experimental"])
        elif "BP" in combo:
            colors.append(PANEL_COLORS["BP"])
        elif "CC" in combo:
            colors.append(PANEL_COLORS["CC"])
        elif "Protein" in combo:
            colors.append(PANEL_COLORS["Protein"])
        else:
            colors.append(BLUE_LIGHT)

    bars = ax.bar(
        range(len(intersection_counts)),
        intersection_counts.values,
        color=colors,
        edgecolor="none",
        width=0.72
    )

    style_axes(ax, grid_axis="y")
    ax.set_xticks(range(len(intersection_counts)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of genes")
    ax.set_title("Overlap structure of the candidate set", pad=10)

    ymax = intersection_counts.max()
    ax.set_ylim(0, ymax * 1.18)

    for bar, value in zip(bars, intersection_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.02,
            f"{int(value)}",
            ha="center",
            va="bottom",
            color=SLATE,
            fontsize=8
        )


def draw_top_isoforms_panel(ax, genes):
    _, top = get_top_isoform_table(genes, n=20)

    ax.hlines(
        y=top["DB_Object_Symbol"],
        xmin=0,
        xmax=top["n_transcripts"],
        color=BLUE_LIGHT,
        linewidth=2.2
    )
    ax.scatter(
        top["n_transcripts"],
        top["DB_Object_Symbol"],
        s=44,
        color=BLUE_DARK,
        zorder=3
    )

    style_axes(ax, grid_axis="x")
    ax.set_xlabel("Number of mapped transcripts")
    ax.set_title("Top genes by transcript isoform count", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def draw_distribution_panel(ax, genes):
    mapped, _ = get_top_isoform_table(genes, n=20)

    max_n = int(mapped["n_transcripts"].max())
    bins = np.arange(0.5, max_n + 1.5, 1)

    ax.hist(
        mapped["n_transcripts"],
        bins=bins,
        color=BLUE_LIGHT,
        edgecolor="white",
        linewidth=1.0,
        rwidth=0.9
    )

    median_val = mapped["n_transcripts"].median()
    ax.axvline(median_val, color=SLATE, linestyle="--", linewidth=1.2)

    style_axes(ax, grid_axis="y")
    ax.set_xlabel("Number of mapped transcripts per gene")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of transcript counts across genes", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ymax = ax.get_ylim()[1]
    ax.text(
        median_val + 0.15,
        ymax * 0.93,
        f"median = {median_val:.0f}",
        color=SLATE,
        fontsize=9
    )


# ============================================================
# Output functions
# ============================================================

def save_individual_panels(genes):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    draw_counts_panel(ax, genes)
    save_figure(fig, "panel_A_counts")

    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    draw_intersections_panel(ax, genes)
    save_figure(fig, "panel_B_intersections")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    draw_top_isoforms_panel(ax, genes)
    save_figure(fig, "panel_C_top_isoforms")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    draw_distribution_panel(ax, genes)
    save_figure(fig, "panel_D_isoform_distribution")


def save_composite_figure(genes):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    draw_counts_panel(ax1, genes)
    draw_intersections_panel(ax2, genes)
    draw_top_isoforms_panel(ax3, genes)
    draw_distribution_panel(ax4, genes)

    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")
    add_panel_label(ax3, "C")
    add_panel_label(ax4, "D")

    fig.suptitle(
        "GO-derived splicing-factor candidate set in Drosophila melanogaster",
        fontsize=15,
        fontweight="bold",
        color=SLATE,
        y=0.98
    )

    fig.savefig(FIG_DIR / "figure_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_overview.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    go_genes, exp_genes, mapping = load_inputs()
    genes = build_gene_table(go_genes, exp_genes, mapping)

    save_individual_panels(genes)
    save_composite_figure(genes)

    genes.to_csv(FIG_DIR / "gene_visualization_table.tsv", sep="\t", index=False)

    print("[DONE]")
    print(f"Saved figure table: {FIG_DIR / 'gene_visualization_table.tsv'}")
    print(f"Saved composite figure: {FIG_DIR / 'figure_overview.png'}")
    print(f"Saved all panels in: {FIG_DIR}")


if __name__ == "__main__":
    main()












# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt


# DATA_DIR = Path("data")
# FIG_DIR = Path("figures")
# FIG_DIR.mkdir(exist_ok=True)

# GO_GENES = DATA_DIR / "go_unique_genes.tsv"
# EXP_GENES = DATA_DIR / "go_unique_genes_experimental.tsv"
# MAPPING = DATA_DIR / "go_genes_fbgn_fbtr_fbpp.tsv"


# def load_inputs():
#     go_genes = pd.read_csv(GO_GENES, sep="\t", dtype=str)
#     exp_genes = pd.read_csv(EXP_GENES, sep="\t", dtype=str)
#     mapping = pd.read_csv(MAPPING, sep="\t", dtype=str)
#     return go_genes, exp_genes, mapping


# def build_gene_table(go_genes, exp_genes, mapping):
#     genes = (
#         go_genes[["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name"]]
#         .drop_duplicates()
#         .rename(columns={"DB_Object_ID": "FBgn"})
#         .copy()
#     )

#     bp_ids = set(go_genes.loc[go_genes["go_bin"] == "BP_function", "DB_Object_ID"])
#     cc_ids = set(go_genes.loc[go_genes["go_bin"] == "CC_component", "DB_Object_ID"])
#     exp_ids = set(exp_genes["DB_Object_ID"])

#     protein_ids = set(
#         mapping.loc[mapping["FlyBase_FBpp"].notna() & (mapping["FlyBase_FBpp"] != ""), "FlyBase_FBgn"]
#     )

#     transcript_counts = (
#         mapping.loc[mapping["FlyBase_FBtr"].notna() & (mapping["FlyBase_FBtr"] != "")]
#         .groupby("FlyBase_FBgn")["FlyBase_FBtr"]
#         .nunique()
#         .to_dict()
#     )

#     genes["BP_support"] = genes["FBgn"].isin(bp_ids)
#     genes["CC_support"] = genes["FBgn"].isin(cc_ids)
#     genes["Experimental_support"] = genes["FBgn"].isin(exp_ids)
#     genes["Has_protein_mapping"] = genes["FBgn"].isin(protein_ids)
#     genes["n_transcripts"] = genes["FBgn"].map(transcript_counts).fillna(0).astype(int)

#     return genes


# def plot_counts_panel(genes):
#     counts = pd.Series({
#         "GO-derived unique genes": genes["FBgn"].nunique(),
#         "BP-supported genes": int(genes["BP_support"].sum()),
#         "CC-supported genes": int(genes["CC_support"].sum()),
#         "Experimental genes": int(genes["Experimental_support"].sum()),
#         "Genes with transcript mapping": int((genes["n_transcripts"] > 0).sum()),
#         "Genes with protein mapping": int(genes["Has_protein_mapping"].sum()),
#     })

#     fig, ax = plt.subplots(figsize=(8, 4.8))
#     counts = counts.sort_values()
#     ax.barh(counts.index, counts.values)
#     ax.set_xlabel("Number of genes")
#     ax.set_title("Summary of GO-derived splicing-factor candidate set")

#     for i, v in enumerate(counts.values):
#         ax.text(v + max(counts.values) * 0.01, i, str(v), va="center")

#     fig.tight_layout()
#     fig.savefig(FIG_DIR / "panel_A_counts.png", dpi=300, bbox_inches="tight")
#     fig.savefig(FIG_DIR / "panel_A_counts.pdf", bbox_inches="tight")
#     plt.close(fig)


# def plot_upset_panel(genes):
#     memberships = []

#     for _, row in genes.iterrows():
#         sets = []
#         if row["BP_support"]:
#             sets.append("BP")
#         if row["CC_support"]:
#             sets.append("CC")
#         if row["Experimental_support"]:
#             sets.append("Experimental")
#         if row["Has_protein_mapping"]:
#             sets.append("Protein")
#         memberships.append(tuple(sets) if sets else ("None",))

#     intersection_counts = pd.Series(memberships).value_counts()

#     labels = [
#         " ∩ ".join(x) if x != ("None",) else "None"
#         for x in intersection_counts.index
#     ]

#     fig, ax = plt.subplots(figsize=(10, 5.5))
#     ax.bar(range(len(intersection_counts)), intersection_counts.values)
#     ax.set_xticks(range(len(intersection_counts)))
#     ax.set_xticklabels(labels, rotation=45, ha="right")
#     ax.set_ylabel("Number of genes")
#     ax.set_title("Overlap structure of the candidate set")

#     fig.tight_layout()
#     fig.savefig(FIG_DIR / "panel_B_intersections.png", dpi=300, bbox_inches="tight")
#     fig.savefig(FIG_DIR / "panel_B_intersections.pdf", bbox_inches="tight")
#     plt.close(fig)


# def plot_isoform_panels(genes):
#     mapped = genes.loc[genes["n_transcripts"] > 0].copy()

#     top = (
#         mapped[["DB_Object_Symbol", "FBgn", "n_transcripts"]]
#         .sort_values(["n_transcripts", "DB_Object_Symbol"], ascending=[False, True])
#         .head(20)
#         .sort_values("n_transcripts")
#     )

#     fig1, ax1 = plt.subplots(figsize=(8, 6))
#     ax1.barh(top["DB_Object_Symbol"], top["n_transcripts"])
#     ax1.set_xlabel("Number of mapped transcripts")
#     ax1.set_title("Top genes by transcript isoform count")
#     fig1.tight_layout()
#     fig1.savefig(FIG_DIR / "panel_C_top_isoforms.png", dpi=300, bbox_inches="tight")
#     fig1.savefig(FIG_DIR / "panel_C_top_isoforms.pdf", bbox_inches="tight")
#     plt.close(fig1)

#     fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
#     bins = range(1, int(mapped["n_transcripts"].max()) + 2)
#     ax2.hist(mapped["n_transcripts"], bins=bins, align="left", rwidth=0.85)
#     ax2.set_xlabel("Number of mapped transcripts per gene")
#     ax2.set_ylabel("Number of genes")
#     ax2.set_title("Distribution of transcript counts across genes")
#     fig2.tight_layout()
#     fig2.savefig(FIG_DIR / "panel_D_isoform_distribution.png", dpi=300, bbox_inches="tight")
#     fig2.savefig(FIG_DIR / "panel_D_isoform_distribution.pdf", bbox_inches="tight")
#     plt.close(fig2)


# def main():
#     go_genes, exp_genes, mapping = load_inputs()
#     genes = build_gene_table(go_genes, exp_genes, mapping)

#     plot_counts_panel(genes)
#     plot_upset_panel(genes)
#     plot_isoform_panels(genes)

#     genes.to_csv(FIG_DIR / "gene_visualization_table.tsv", sep="\t", index=False)

#     print("[DONE]")
#     print(f"Saved figure table: {FIG_DIR / 'gene_visualization_table.tsv'}")
#     print(f"Saved figures in: {FIG_DIR}")


# if __name__ == "__main__":
#     main()
