from pathlib import Path
import re
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("data/gse269951")
GEO_DIR = BASE_DIR / "geo"
META_DIR = BASE_DIR / "metadata"
OUT_DIR = BASE_DIR / "candidate_expression"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_TABLE = Path("data/go_genes_fbgn_fbtr_fbpp.tsv")
METADATA_TABLE = META_DIR / "GSE269951_metadata.tsv"
DMEL_VST_PATH = GEO_DIR / "GSE269951_d_melanogaster_vst_matrix.csv.gz"


# ============================================================
# Candidate table: collapse to one row per FBgn
# ============================================================

def build_gene_annotation_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)

    grouped = (
        df.groupby("FlyBase_FBgn", dropna=False)
        .agg({
            "DB_Object_Symbol": "first",
            "DB_Object_Name": "first",
            "go_bin": lambda x: ";".join(sorted(set(v for v in x.dropna() if v))),
        })
        .reset_index()
    )

    return grouped


# ============================================================
# Metadata: keep only Dmel and create stage ordering
# ============================================================

def extract_first_int(text: str):
    if not isinstance(text, str):
        return None
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def stage_sort_key(row) -> int:
    stage = str(row.get("developmental_stage", "")).lower()
    timing = str(row.get("timing", "")).lower()

    if stage == "embryo":
        n = extract_first_int(timing)
        return 10 + (n if n is not None else 0)

    if stage == "larvae":
        if "instar" in timing:
            n = extract_first_int(timing)
            return 100 + (n if n is not None else 0)
        return 100

    if stage == "pupae":
        if timing.startswith("p"):
            n = extract_first_int(timing)
            return 200 + (n if n is not None else 0)
        n = extract_first_int(timing)
        return 300 + (n if n is not None else 0)

    if stage == "imago":
        n = extract_first_int(timing)
        return 400 + (n if n is not None else 0)

    return 999


def prepare_dmel_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", dtype=str)
    meta = meta[meta["species_short"] == "Dmel"].copy()

    meta["stage_sort_key"] = meta.apply(stage_sort_key, axis=1)
    meta = meta.sort_values(["stage_sort_key", "replicate", "gsm_id"]).reset_index(drop=True)

    return meta


# ============================================================
# Match matrix sample columns to metadata
# ============================================================

def normalize_stage_for_matrix(stage: str) -> str:
    stage = str(stage).strip().lower()

    if stage == "embryo":
        return "embryo"
    if stage == "larvae":
        return "larvae"
    if stage == "pupae":
        return "pupae"
    if stage == "imago":
        return "adults"

    return stage


def normalize_timing_for_matrix(timing: str) -> str:
    timing = str(timing).strip()

    timing_map = {
        "1instar": "1st instar",
        "2instar": "2nd instar",
        "3instar": "3rd instar",
        "5d": "5d",
        "10h": "10h",
        "24h": "24h",
        "48h": "48h",
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "P4",
        "P5": "P5",
    }

    return timing_map.get(timing, timing)


def extract_replicate_number(rep: str) -> str:
    rep = str(rep).strip()
    m = re.search(r"(\d+)$", rep)
    return m.group(1) if m else rep

def attach_matrix_columns(meta: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    sample_cols = list(matrix.columns[1:])
    meta = meta.copy()

    def build_matrix_label(row):
        stage = normalize_stage_for_matrix(row["developmental_stage"])
        timing = normalize_timing_for_matrix(row["timing"])

        # special case: larval instars in the matrix all end with "1"
        if stage == "larvae":
            rep = "1"
        else:
            rep = extract_replicate_number(row["replicate"])

        return f"{stage} {timing} {rep}"

    meta["matrix_column"] = meta.apply(build_matrix_label, axis=1)

    if set(sample_cols) != set(meta["matrix_column"]):
        pd.DataFrame({"matrix_column": sample_cols}).to_csv(
            OUT_DIR / "dmel_matrix_columns.tsv", sep="\t", index=False
        )
        meta.to_csv(
            OUT_DIR / "dmel_metadata_with_expected_columns.tsv", sep="\t", index=False
        )

        raise ValueError(
            "Constructed metadata sample labels still do not match matrix columns. "
            "Inspect dmel_matrix_columns.tsv and dmel_metadata_with_expected_columns.tsv"
        )

    return meta


# ============================================================
# Build candidate-only expression outputs
# ============================================================

def build_outputs():
    annot = build_gene_annotation_table(CANDIDATE_TABLE)
    meta = prepare_dmel_metadata(METADATA_TABLE)

    matrix = pd.read_csv(DMEL_VST_PATH, compression="gzip", dtype=str)
    matrix = matrix.rename(columns={matrix.columns[0]: "FlyBase_FBgn"})

    meta = attach_matrix_columns(meta, matrix)

    candidate_ids = set(annot["FlyBase_FBgn"])
    matched = matrix[matrix["FlyBase_FBgn"].isin(candidate_ids)].copy()
    missing = annot[~annot["FlyBase_FBgn"].isin(set(matched["FlyBase_FBgn"]))].copy()

    ordered_sample_cols = list(meta["matrix_column"])
    matched = matched[["FlyBase_FBgn"] + ordered_sample_cols].copy()

    wide = annot.merge(matched, on="FlyBase_FBgn", how="inner")
    wide.to_csv(OUT_DIR / "dmel_candidate_expression_wide.tsv", sep="\t", index=False)

    missing.to_csv(OUT_DIR / "dmel_candidate_missing_from_vst.tsv", sep="\t", index=False)

    meta.to_csv(OUT_DIR / "dmel_sample_sheet.tsv", sep="\t", index=False)

    long = wide.melt(
        id_vars=["FlyBase_FBgn", "DB_Object_Symbol", "DB_Object_Name", "go_bin"],
        value_vars=ordered_sample_cols,
        var_name="matrix_column",
        value_name="vst_expression"
    )

    long = long.merge(
        meta[
            [
                "matrix_column",
                "gsm_id",
                "title",
                "replicate",
                "developmental_stage",
                "timing",
                "coarse_stage",
                "exact_stage_label",
                "stage_sort_key",
            ]
        ],
        on="matrix_column",
        how="left"
    )

    long.to_csv(OUT_DIR / "dmel_candidate_expression_long.tsv", sep="\t", index=False)

    report_lines = [
        f"Unique candidate FBgn in input table: {annot['FlyBase_FBgn'].nunique()}",
        f"Matched candidate genes in Dmel VST matrix: {wide['FlyBase_FBgn'].nunique()}",
        f"Missing candidate genes from Dmel VST matrix: {missing['FlyBase_FBgn'].nunique()}",
        f"Dmel samples in metadata: {len(meta)}",
    ]
    (OUT_DIR / "dmel_candidate_expression_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("[DONE]")
    print(f"Saved: {OUT_DIR / 'dmel_sample_sheet.tsv'}")
    print(f"Saved: {OUT_DIR / 'dmel_candidate_expression_wide.tsv'}")
    print(f"Saved: {OUT_DIR / 'dmel_candidate_expression_long.tsv'}")
    print(f"Saved: {OUT_DIR / 'dmel_candidate_missing_from_vst.tsv'}")
    print(f"Saved: {OUT_DIR / 'dmel_candidate_expression_report.txt'}")


if __name__ == "__main__":
    build_outputs()