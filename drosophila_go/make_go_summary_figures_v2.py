from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator


DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

GO_GENES = DATA_DIR / "go_unique_genes.tsv"
EXP_GENES = DATA_DIR / "go_unique_genes_experimental.tsv"
MAPPING = DATA_DIR / "go_genes_fbgn_fbtr_fbpp.tsv"


COLORS = {
    "ink": "#243447",
    "muted": "#6A7585",
    "grid": "#D9E0EA",
    "panel_bg": "#F7F9FC",
    "bp": "#3566A8",
    "cc": "#2F8F88",
    "experimental": "#C47B29",
    "protein": "#A14E67",
    "transcript": "#53657D",
    "neutral": "#A8B3C2",
    "neutral_dark": "#607081",
    "bp_only": "#4A78B7",
    "bp_cc": "#658F88",
    "cc_only": "#6AAEA8",
}

SET_ORDER = ["BP", "CC", "Experimental", "Protein"]
CLASS_ORDER = ["BP only", "BP + CC", "CC only"]
CLASS_COLORS = {
    "BP only": COLORS["bp_only"],
    "BP + CC": COLORS["bp_cc"],
    "CC only": COLORS["cc_only"],
}
SET_COLORS = {
    "BP": COLORS["bp"],
    "CC": COLORS["cc"],
    "Experimental": COLORS["experimental"],
    "Protein": COLORS["protein"],
}


def configure_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": COLORS["grid"],
        "axes.linewidth": 0.9,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.9,
        "grid.alpha": 0.85,
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def mix_with_white(color, amount=0.65):
    rgb = np.array(to_rgb(color))
    white = np.ones(3)
    mixed = rgb * (1 - amount) + white * amount
    return tuple(mixed)


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["ink"])
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis)


def add_panel_label(ax, label, x=-0.14):
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def save_figure(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_inputs():
    go_genes = pd.read_csv(GO_GENES, sep="\t", dtype=str).fillna("")
    exp_genes = pd.read_csv(EXP_GENES, sep="\t", dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING, sep="\t", dtype=str).fillna("")
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
            mapping["FlyBase_FBpp"].ne(""),
            "FlyBase_FBgn",
        ]
    )

    transcript_counts = (
        mapping.loc[mapping["FlyBase_FBtr"].ne("")]
        .groupby("FlyBase_FBgn")["FlyBase_FBtr"]
        .nunique()
    )

    genes["BP_support"] = genes["FBgn"].isin(bp_ids)
    genes["CC_support"] = genes["FBgn"].isin(cc_ids)
    genes["Experimental_support"] = genes["FBgn"].isin(exp_ids)
    genes["Has_protein_mapping"] = genes["FBgn"].isin(protein_ids)
    genes["n_transcripts"] = genes["FBgn"].map(transcript_counts).fillna(0).astype(int)
    genes["Has_transcript_mapping"] = genes["n_transcripts"] > 0

    genes["go_class"] = np.select(
        [
            genes["BP_support"] & genes["CC_support"],
            genes["BP_support"],
            genes["CC_support"],
        ],
        [
            "BP + CC",
            "BP only",
            "CC only",
        ],
        default="Unassigned",
    )

    return genes


def build_summary_counts(go_genes, genes):
    total_genes = genes["FBgn"].nunique()

    rows = [
        {
            "label": "Gene-bin rows",
            "count": int(len(go_genes)),
            "annotation": f"{len(go_genes) / total_genes:.2f} rows per gene",
            "color": COLORS["neutral_dark"],
        },
        {
            "label": "Unique genes (FBgn)",
            "count": int(total_genes),
            "annotation": "deduplicated across BP and CC",
            "color": COLORS["ink"],
        },
        {
            "label": "BP-supported genes",
            "count": int(genes["BP_support"].sum()),
            "annotation": f"{genes['BP_support'].mean():.1%} of unique genes",
            "color": COLORS["bp"],
        },
        {
            "label": "CC-supported genes",
            "count": int(genes["CC_support"].sum()),
            "annotation": f"{genes['CC_support'].mean():.1%} of unique genes",
            "color": COLORS["cc"],
        },
        {
            "label": "Experimental subset",
            "count": int(genes["Experimental_support"].sum()),
            "annotation": f"{genes['Experimental_support'].mean():.1%} of unique genes",
            "color": COLORS["experimental"],
        },
        {
            "label": "Transcript-mapped genes",
            "count": int(genes["Has_transcript_mapping"].sum()),
            "annotation": f"{genes['Has_transcript_mapping'].mean():.1%} of unique genes",
            "color": COLORS["transcript"],
        },
        {
            "label": "Protein-mapped genes",
            "count": int(genes["Has_protein_mapping"].sum()),
            "annotation": f"{genes['Has_protein_mapping'].mean():.1%} of unique genes",
            "color": COLORS["protein"],
        },
    ]

    return pd.DataFrame(rows)


def build_overlap_table(genes):
    records = []

    for row in genes.itertuples(index=False):
        active_sets = [
            set_name
            for set_name, attr in [
                ("BP", "BP_support"),
                ("CC", "CC_support"),
                ("Experimental", "Experimental_support"),
                ("Protein", "Has_protein_mapping"),
            ]
            if getattr(row, attr)
        ]
        combo_key = tuple(active_sets)
        records.append(combo_key)

    counts = pd.Series(records).value_counts().sort_values(ascending=False)
    overlap = pd.DataFrame({
        "combo": counts.index,
        "count": counts.values,
    })

    for set_name in SET_ORDER:
        overlap[set_name] = overlap["combo"].apply(lambda combo: set_name in combo)

    overlap["n_sets"] = overlap["combo"].apply(len)
    overlap["share"] = overlap["count"] / len(genes)
    overlap["pattern"] = overlap["combo"].apply(lambda combo: " + ".join(combo))
    overlap = overlap.sort_values(
        ["count", "n_sets", "pattern"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return overlap


def build_top_gene_table(genes, n=15):
    top = (
        genes.loc[genes["Has_transcript_mapping"]]
        .sort_values(
            ["n_transcripts", "DB_Object_Symbol", "FBgn"],
            ascending=[False, True, True],
        )
        .head(n)
        .sort_values(
            ["n_transcripts", "DB_Object_Symbol", "FBgn"],
            ascending=[True, True, True],
        )
        .copy()
    )
    return top


def draw_counts_panel(ax, counts_df):
    y_positions = np.arange(len(counts_df))
    max_count = counts_df["count"].max()

    for y, row in zip(y_positions, counts_df.itertuples(index=False)):
        line_color = mix_with_white(row.color, amount=0.68)
        ax.hlines(y, 0, row.count, color=line_color, linewidth=5, zorder=1, capstyle="round")
        ax.scatter(
            row.count,
            y,
            s=110,
            color=row.color,
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        ax.text(
            row.count + max_count * 0.03,
            y,
            f"{row.count:,}  |  {row.annotation}",
            va="center",
            ha="left",
            fontsize=9.2,
            color=COLORS["muted"],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(counts_df["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, max_count * 1.36)
    ax.set_xlabel("Number of genes / rows")
    ax.set_title("Candidate-set summary", loc="left", pad=10)
    ax.xaxis.set_major_locator(MultipleLocator(100))

    style_axis(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", pad=10)
    ax.axvline(0, color=COLORS["grid"], linewidth=1.0, zorder=0)

    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() == "Unique genes (FBgn)":
            tick_label.set_fontweight("bold")

    ax.text(
        0,
        -0.18,
        "The source table contains 506 gene-bin rows because the same FBgn can be counted once in "
        "BP_function and once in CC_component; deduplication yields 309 unique genes.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["muted"],
        wrap=True,
    )


def draw_upset_panel(fig, outer_spec, overlap_df):
    inner = outer_spec.subgridspec(2, 1, height_ratios=[3.4, 1.6], hspace=0.05)
    ax_bar = fig.add_subplot(inner[0, 0])
    ax_matrix = fig.add_subplot(inner[1, 0], sharex=ax_bar)

    x = np.arange(len(overlap_df))

    bars = ax_bar.bar(
        x,
        overlap_df["count"],
        width=0.72,
        color=COLORS["ink"],
        alpha=0.92,
        zorder=3,
    )

    style_axis(ax_bar, grid_axis="y")
    ax_bar.grid(axis="x", visible=False)
    ax_bar.set_ylabel("Genes")
    ax_bar.set_title("Observed overlap structure", loc="left", pad=10)
    ax_bar.set_xlim(-0.6, len(overlap_df) - 0.4)
    ax_bar.set_xticks([])
    ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True))

    y_max = overlap_df["count"].max()
    ax_bar.set_ylim(0, y_max * 1.2)

    for bar, row in zip(bars, overlap_df.itertuples(index=False)):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            row.count + y_max * 0.03,
            f"{row.count}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["ink"],
        )

    ax_bar.text(
        1.0,
        1.03,
        f"{len(overlap_df)} observed intersection patterns",
        transform=ax_bar.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
    )

    for row_index, set_name in enumerate(SET_ORDER):
        if row_index % 2 == 0:
            ax_matrix.axhspan(
                row_index - 0.5,
                row_index + 0.5,
                color=COLORS["panel_bg"],
                zorder=0,
            )

    inactive_color = mix_with_white(COLORS["neutral"], amount=0.2)
    connector_color = mix_with_white(COLORS["ink"], amount=0.25)

    for column_index, row in enumerate(overlap_df.itertuples(index=False)):
        active_rows = []
        for row_index, set_name in enumerate(SET_ORDER):
            ax_matrix.scatter(
                column_index,
                row_index,
                s=42,
                color=inactive_color,
                zorder=2,
            )
            if getattr(row, set_name):
                active_rows.append((row_index, set_name))

        if active_rows:
            row_positions = [item[0] for item in active_rows]
            if len(row_positions) > 1:
                ax_matrix.plot(
                    [column_index, column_index],
                    [min(row_positions), max(row_positions)],
                    color=connector_color,
                    linewidth=1.7,
                    zorder=1,
                )
            for row_index, set_name in active_rows:
                ax_matrix.scatter(
                    column_index,
                    row_index,
                    s=88,
                    color=SET_COLORS[set_name],
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=3,
                )

    ax_matrix.set_yticks(np.arange(len(SET_ORDER)))
    ax_matrix.set_yticklabels(SET_ORDER)
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([""] * len(x))
    ax_matrix.invert_yaxis()
    ax_matrix.set_xlabel("Intersection patterns sorted by size")
    ax_matrix.grid(False)

    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    for tick_label in ax_matrix.get_yticklabels():
        tick_label.set_color(SET_COLORS[tick_label.get_text()])
        tick_label.set_fontweight("bold")

    ax_matrix.tick_params(axis="y", length=0)
    ax_matrix.tick_params(axis="x", length=0)
    ax_matrix.set_xlim(-0.6, len(overlap_df) - 0.4)

    return ax_bar, ax_matrix


def draw_isoform_distribution_panel(ax, genes):
    plot_df = genes.loc[genes["go_class"].isin(CLASS_ORDER), ["go_class", "n_transcripts"]].copy()

    sns.violinplot(
        data=plot_df,
        y="go_class",
        x="n_transcripts",
        hue="go_class",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette=CLASS_COLORS,
        dodge=False,
        cut=0,
        inner=None,
        linewidth=0,
        bw_adjust=0.9,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        y="go_class",
        x="n_transcripts",
        order=CLASS_ORDER,
        color=COLORS["ink"],
        alpha=0.28,
        jitter=0.22,
        size=3.3,
        ax=ax,
        zorder=2,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    summary = (
        plot_df.groupby("go_class")["n_transcripts"]
        .agg(
            n="size",
            q1=lambda s: s.quantile(0.25),
            median="median",
            q3=lambda s: s.quantile(0.75),
        )
        .reindex(CLASS_ORDER)
    )

    x_max = max(int(plot_df["n_transcripts"].max()), 1)

    for y, class_name in enumerate(CLASS_ORDER):
        stats = summary.loc[class_name]
        ax.plot(
            [stats["q1"], stats["q3"]],
            [y, y],
            color="white",
            linewidth=5,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            stats["median"],
            y,
            s=72,
            color="white",
            edgecolor=COLORS["ink"],
            linewidth=1.1,
            zorder=5,
        )
        ax.text(
            x_max + 0.8,
            y,
            f"n={int(stats['n'])}, median={stats['median']:.0f}",
            ha="left",
            va="center",
            fontsize=9,
            color=COLORS["muted"],
        )

    style_axis(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_ylabel("")
    ax.set_xlabel("Mapped transcripts per gene")
    ax.set_title("Isoform counts by GO-bin class", loc="left", pad=10)
    ax.set_xlim(-0.2, x_max + 5.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for tick_label in ax.get_yticklabels():
        label = tick_label.get_text()
        tick_label.set_color(CLASS_COLORS.get(label, COLORS["ink"]))
        tick_label.set_fontweight("bold")

    ax.text(
        0,
        -0.18,
        "Classes are mutually exclusive: BP only, CC only, or genes present in both GO bins.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["muted"],
    )


def draw_top_genes_panel(ax, top_df):
    y_positions = np.arange(len(top_df))

    for y, row in zip(y_positions, top_df.itertuples(index=False)):
        line_color = mix_with_white(CLASS_COLORS[row.go_class], amount=0.7)
        ax.hlines(y, 0, row.n_transcripts, color=line_color, linewidth=4.5, zorder=1)
        ax.scatter(
            row.n_transcripts,
            y,
            s=95,
            color=CLASS_COLORS[row.go_class],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row.n_transcripts + 0.45,
            y,
            f"{row.n_transcripts}",
            ha="left",
            va="center",
            fontsize=9,
            color=COLORS["muted"],
        )

    style_axis(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_df["DB_Object_Symbol"])
    ax.set_xlabel("Mapped transcripts")
    ax.set_title("Top genes by transcript count", loc="left", pad=10)
    ax.set_xlim(0, top_df["n_transcripts"].max() + 4.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontstyle("italic")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_COLORS[label],
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=8.5,
            label=label,
        )
        for label in CLASS_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        title="GO-bin class",
        frameon=False,
        loc="lower right",
    )


def save_individual_panels(counts_df, overlap_df, genes, top_df):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    draw_counts_panel(ax, counts_df)
    save_figure(fig, "panel_A_counts_v2")

    fig = plt.figure(figsize=(10.0, 5.6))
    outer = fig.add_gridspec(1, 1)
    draw_upset_panel(fig, outer[0, 0], overlap_df)
    save_figure(fig, "panel_B_overlap_upset_v2")

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    draw_isoform_distribution_panel(ax, genes)
    save_figure(fig, "panel_C_isoform_distribution_v2")

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    draw_top_genes_panel(ax, top_df)
    save_figure(fig, "panel_D_top_isoform_genes_v2")


def save_composite_figure(counts_df, overlap_df, genes, top_df):
    fig = plt.figure(figsize=(16.4, 10.6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.25, 1.25],
        height_ratios=[1.0, 1.0],
        wspace=0.62,
        hspace=0.56,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])
    ax_b_bar, _ = draw_upset_panel(fig, gs[0, 1:], overlap_df)

    draw_counts_panel(ax_a, counts_df)
    draw_isoform_distribution_panel(ax_c, genes)
    draw_top_genes_panel(ax_d, top_df)

    add_panel_label(ax_a, "A")
    add_panel_label(ax_b_bar, "B", x=-0.04)
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D", x=-0.04)

    fig.suptitle(
        "GO-derived Drosophila splicing-factor candidate set",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.955,
        "FlyBase GO rows (n=506) collapse to 309 unique FBgn genes after removing BP/CC duplication; "
        "experimental support and transcript/protein mapping are shown as downstream annotations.",
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["muted"],
    )
    fig.text(
        0.02,
        0.02,
        "Inputs: data/go_unique_genes.tsv, data/go_unique_genes_experimental.tsv, "
        "and data/go_genes_fbgn_fbtr_fbpp.tsv.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
    )

    save_figure(fig, "figure_overview_v2")


def save_support_tables(genes, counts_df, overlap_df, top_df):
    genes.to_csv(FIG_DIR / "gene_visualization_table_v2.tsv", sep="\t", index=False)
    counts_df.to_csv(FIG_DIR / "summary_counts_v2.tsv", sep="\t", index=False)
    overlap_df.to_csv(FIG_DIR / "intersection_counts_v2.tsv", sep="\t", index=False)
    top_df.to_csv(FIG_DIR / "top_isoform_genes_v2.tsv", sep="\t", index=False)


def main():
    configure_style()
    go_genes, exp_genes, mapping = load_inputs()
    genes = build_gene_table(go_genes, exp_genes, mapping)

    counts_df = build_summary_counts(go_genes, genes)
    overlap_df = build_overlap_table(genes)
    top_df = build_top_gene_table(genes, n=15)

    save_individual_panels(counts_df, overlap_df, genes, top_df)
    save_composite_figure(counts_df, overlap_df, genes, top_df)
    save_support_tables(genes, counts_df, overlap_df, top_df)

    print("[DONE]")
    print(f"Saved overview figure: {FIG_DIR / 'figure_overview_v2.png'}")
    print(f"Saved individual panels in: {FIG_DIR}")
    print(f"Saved support tables in: {FIG_DIR}")


if __name__ == "__main__":
    main()
