from pathlib import Path
import requests
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Use FlyBase current release bulk GO file
GAF_URL = "https://s3ftp.flybase.org/releases/current/precomputed_files/go/gene_association.fb.gz"
GAF_PATH = DATA_DIR / "gene_association.fb.gz"

# GO terms from our agreed list
GO_TERMS = {
    "BP_function": [
        "GO:0000398",  # mRNA splicing, via spliceosome
        "GO:0000380",  # alternative mRNA splicing, via spliceosome
        "GO:0048024",  # regulation of mRNA splicing, via spliceosome
        "GO:0000381",  # regulation of alternative mRNA splicing, via spliceosome
        "GO:0008380",  # RNA splicing
        "GO:0043484",  # regulation of RNA splicing
    ],
    "CC_component": [
        "GO:0005681",  # spliceosomal complex
        "GO:0097525",  # spliceosomal snRNP complex
        "GO:0005685",  # U1 snRNP
        "GO:0005686",  # U2 snRNP
        "GO:0071001",  # U4/U6 snRNP
        "GO:0005682",  # U5 snRNP
        "GO:0097526",  # spliceosomal tri-snRNP complex
        "GO:0046540",  # U4/U6 x U5 tri-snRNP complex
        "GO:0005684",  # U2-type spliceosomal complex
        "GO:0071011",  # precatalytic spliceosome
        "GO:0071013",  # catalytic step 2 spliceosome
    ],
}

# Experimental evidence bin
EXPERIMENTAL_EVIDENCE = {
    "EXP", "IDA", "IMP", "IGI", "IPI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP"
}

GAF_COLUMNS = [
    "DB",
    "DB_Object_ID",
    "DB_Object_Symbol",
    "Qualifier",
    "GO_ID",
    "DB_Reference",
    "Evidence",
    "With_or_From",
    "Aspect",
    "DB_Object_Name",
    "DB_Object_Synonym",
    "DB_Object_Type",
    "Taxon",
    "Date",
    "Assigned_By",
    "Annotation_Extension",
    "Gene_Product_Form_ID",
]


# -----------------------------
# Helpers
# -----------------------------
def download_file(url: str, out_path: Path):
    if out_path.exists():
        print(f"[OK] File already exists: {out_path}")
        return

    print(f"[DOWNLOAD] {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[SAVED] {out_path}")


def load_gaf(path: Path) -> pd.DataFrame:
    print(f"[LOAD] {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        comment="!",
        header=None,
        dtype=str,
        low_memory=False,
        compression="gzip",
    )

    # Handle either 16 or 17 columns safely
    df.columns = GAF_COLUMNS[:df.shape[1]]

    for col in df.columns:
        df[col] = df[col].fillna("")

    return df


def main():
    download_file(GAF_URL, GAF_PATH)
    df = load_gaf(GAF_PATH)

    # map GO term -> bin
    go_to_bin = {}
    for bin_name, go_list in GO_TERMS.items():
        for go_id in go_list:
            go_to_bin[go_id] = bin_name

    selected_go_ids = set(go_to_bin.keys())

    # Filter by exact GO IDs
    matches = df[df["GO_ID"].isin(selected_go_ids)].copy()

    # Drop negated annotations
    matches = matches[~matches["Qualifier"].str.contains("NOT", na=False)].copy()

    # Add bins
    matches["go_bin"] = matches["GO_ID"].map(go_to_bin)
    matches["evidence_bin"] = matches["Evidence"].apply(
        lambda x: "experimental" if x in EXPERIMENTAL_EVIDENCE else "non_experimental_or_assertion"
    )

    # Keep useful columns
    keep_cols = [
        "DB_Object_ID",
        "DB_Object_Symbol",
        "DB_Object_Name",
        "GO_ID",
        "Aspect",
        "Qualifier",
        "Evidence",
        "DB_Reference",
        "go_bin",
        "evidence_bin",
    ]
    matches = matches[keep_cols].sort_values(
        ["go_bin", "GO_ID", "DB_Object_Symbol", "Evidence"]
    )

    # Save raw annotation-level matches
    raw_path = DATA_DIR / "go_matches_raw.tsv"
    matches.to_csv(raw_path, sep="\t", index=False)

    # Unique genes per bin
    unique_genes = (
        matches[["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]]
        .drop_duplicates()
        .sort_values(["go_bin", "DB_Object_Symbol"])
    )
    unique_path = DATA_DIR / "go_unique_genes.tsv"
    unique_genes.to_csv(unique_path, sep="\t", index=False)

    # Unique genes with only experimental support among matched rows
    exp_genes = (
        matches[matches["evidence_bin"] == "experimental"][
            ["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]
        ]
        .drop_duplicates()
        .sort_values(["go_bin", "DB_Object_Symbol"])
    )
    exp_path = DATA_DIR / "go_unique_genes_experimental.tsv"
    exp_genes.to_csv(exp_path, sep="\t", index=False)

    # Summary counts
    summary = (
        matches.groupby(["go_bin", "GO_ID", "evidence_bin"])["DB_Object_ID"]
        .nunique()
        .reset_index(name="n_unique_genes")
        .sort_values(["go_bin", "GO_ID", "evidence_bin"])
    )
    summary_path = DATA_DIR / "go_summary_by_term.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    # Separate BP and CC outputs
    for bin_name in GO_TERMS:
        subset = matches[matches["go_bin"] == bin_name].copy()
        subset.to_csv(DATA_DIR / f"{bin_name}_raw.tsv", sep="\t", index=False)

        subset_unique = (
            subset[["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name"]]
            .drop_duplicates()
            .sort_values("DB_Object_Symbol")
        )
        subset_unique.to_csv(DATA_DIR / f"{bin_name}_unique_genes.tsv", sep="\t", index=False)

    print("\nDone.")
    print(f"Raw matches:              {raw_path}")
    print(f"Unique genes:             {unique_path}")
    print(f"Experimental genes:       {exp_path}")
    print(f"Summary by term:          {summary_path}")
    print(f"BP unique genes:          {DATA_DIR / 'BP_function_unique_genes.tsv'}")
    print(f"CC unique genes:          {DATA_DIR / 'CC_component_unique_genes.tsv'}")
    print(f"\nAnnotation-level rows matched: {len(matches)}")
    print(f"Unique genes matched:          {unique_genes['DB_Object_ID'].nunique()}")


if __name__ == "__main__":
    main()