from pathlib import Path
import gzip
import re
import requests
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("data/gse269951")
GEO_DIR = BASE_DIR / "geo"
META_DIR = BASE_DIR / "metadata"
REPORT_DIR = BASE_DIR / "reports"

for p in [GEO_DIR, META_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Input candidate table from previous step
# ============================================================

CANDIDATE_TABLE = Path("data/go_genes_fbgn_fbtr_fbpp.tsv")


# ============================================================
# GEO files needed for current step
# ============================================================

SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269951/soft/GSE269951_family.soft.gz"
DMEL_VST_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269951/suppl/GSE269951_d_melanogaster_vst_matrix.csv.gz"

SOFT_PATH = GEO_DIR / "GSE269951_family.soft.gz"
DMEL_VST_PATH = GEO_DIR / "GSE269951_d_melanogaster_vst_matrix.csv.gz"


# ============================================================
# Download helper
# ============================================================

def download_file(url: str, out_path: Path):
    if out_path.exists():
        print(f"[OK] Exists: {out_path}")
        return

    print(f"[DOWNLOAD] {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[SAVED] {out_path}")


# ============================================================
# Parse GEO SOFT file into metadata table
# ============================================================

def parse_soft_metadata(soft_path: Path) -> pd.DataFrame:
    samples = []
    current = None
    characteristics = []

    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("^SAMPLE = "):
                if current is not None:
                    current["characteristics_raw"] = characteristics.copy()
                    samples.append(current)

                gsm = line.split("=", 1)[1].strip()
                current = {"gsm_id": gsm}
                characteristics = []

            elif current is not None:
                if line.startswith("!Sample_title = "):
                    current["title"] = line.split("=", 1)[1].strip()

                elif line.startswith("!Sample_source_name_ch1 = "):
                    current["source_name"] = line.split("=", 1)[1].strip()

                elif line.startswith("!Sample_organism_ch1 = "):
                    current["organism"] = line.split("=", 1)[1].strip()

                elif line.startswith("!Sample_characteristics_ch1 = "):
                    value = line.split("=", 1)[1].strip()
                    characteristics.append(value)

                elif line.startswith("!Sample_relation = "):
                    value = line.split("=", 1)[1].strip()
                    if "SRA:" in value:
                        current["sra_relation"] = value

        if current is not None:
            current["characteristics_raw"] = characteristics.copy()
            samples.append(current)

    rows = []
    for s in samples:
        row = {
            "gsm_id": s.get("gsm_id", ""),
            "title": s.get("title", ""),
            "source_name": s.get("source_name", ""),
            "organism": s.get("organism", ""),
            "sra_relation": s.get("sra_relation", ""),
        }

        for item in s.get("characteristics_raw", []):
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    df["species_short"] = df["organism"].map({
        "Drosophila melanogaster": "Dmel",
        "Drosophila virilis": "Dvir",
    }).fillna("")

    stage_map = {
        "embryo": "embryo",
        "larvae": "larva",
        "pupae": "pupa",
        "imago": "adult",
    }

    if "developmental_stage" in df.columns:
        df["coarse_stage"] = df["developmental_stage"].map(stage_map).fillna(df["developmental_stage"])
    else:
        df["coarse_stage"] = ""

    timing = df["timing"] if "timing" in df.columns else pd.Series([""] * len(df))
    developmental_stage = df["developmental_stage"] if "developmental_stage" in df.columns else pd.Series([""] * len(df))

    df["exact_stage_label"] = [
        f"{stage}_{time}".strip("_")
        for stage, time in zip(developmental_stage.fillna(""), timing.fillna(""))
    ]

    def extract_srx(x: str) -> str:
        if not isinstance(x, str):
            return ""
        m = re.search(r"(SRX\d+)", x)
        return m.group(1) if m else ""

    df["srx_accession"] = df["sra_relation"].apply(extract_srx)

    preferred = [
        "gsm_id",
        "title",
        "organism",
        "species_short",
        "source_name",
        "tissue",
        "gender",
        "replicate",
        "developmental_stage",
        "timing",
        "coarse_stage",
        "exact_stage_label",
        "srx_accession",
        "sra_relation",
    ]

    existing = [c for c in preferred if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    df = df[existing + other]

    return df


# ============================================================
# Coverage check using FBgn only
# ============================================================

def coverage_check_fbgn_only(matrix_path: Path, candidates_path: Path):
    matrix = pd.read_csv(matrix_path, compression="gzip", dtype=str)
    candidates = pd.read_csv(candidates_path, sep="\t", dtype=str)

    matrix = matrix.copy()
    matrix_id_col = matrix.columns[0]
    matrix[matrix_id_col] = matrix[matrix_id_col].astype(str)

    candidate_fbgn = set(candidates["FlyBase_FBgn"].dropna().astype(str))

    matched = matrix[matrix[matrix_id_col].isin(candidate_fbgn)].copy()

    report_lines = [
        f"Matrix rows: {len(matrix)}",
        f"Unique candidate FBgn: {len(candidate_fbgn)}",
        f"Matched matrix rows by FBgn: {len(matched)}",
    ]

    report_path = REPORT_DIR / "dmel_vst_candidate_coverage_fbgn.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    matched.to_csv(REPORT_DIR / "dmel_vst_candidate_matches_by_fbgn.csv", index=False)

    print(f"[REPORT] {report_path}")
    print(f"[SAVED] {REPORT_DIR / 'dmel_vst_candidate_matches_by_fbgn.csv'}")


# ============================================================
# Main
# ============================================================

def main():
    download_file(SOFT_URL, SOFT_PATH)
    download_file(DMEL_VST_URL, DMEL_VST_PATH)

    metadata = parse_soft_metadata(SOFT_PATH)
    metadata.to_csv(META_DIR / "GSE269951_metadata.tsv", sep="\t", index=False)
    print(f"[SAVED] {META_DIR / 'GSE269951_metadata.tsv'}")

    coverage_check_fbgn_only(DMEL_VST_PATH, CANDIDATE_TABLE)

    print("[DONE]")


if __name__ == "__main__":
    main()