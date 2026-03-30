from pathlib import Path
import requests
import pandas as pd

# input: the gene list you already created in the previous step
INPUT_GENES = Path("data/go_unique_genes.tsv")

# output folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# current FlyBase mapping file
MAPPING_URL = "https://s3ftp.flybase.org/releases/FB2026_01/precomputed_files/genes/fbgn_fbtr_fbpp_fb_2026_01.tsv.gz"
MAPPING_PATH = DATA_DIR / "fbgn_fbtr_fbpp_fb_2026_01.tsv.gz"

# output: only mappings for your GO-derived genes
OUTPUT_PATH = DATA_DIR / "go_genes_fbgn_fbtr_fbpp.tsv"


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


def load_gene_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df[["DB_Object_ID", "DB_Object_Symbol", "DB_Object_Name", "go_bin"]].copy()
    df = df.rename(columns={"DB_Object_ID": "FlyBase_FBgn"})
    return df


def load_mapping_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", header=None, dtype=str, compression="gzip")
    df = df.iloc[:, :3].copy()  # Select only the first three columns
    df.columns = ["FlyBase_FBgn", "FlyBase_FBtr", "FlyBase_FBpp"]
    return df


def main():
    download_file(MAPPING_URL, MAPPING_PATH)

    genes = load_gene_list(INPUT_GENES)
    mapping = load_mapping_table(MAPPING_PATH)

    result = genes.merge(mapping, on="FlyBase_FBgn", how="left")
    result = result.sort_values(["go_bin", "DB_Object_Symbol", "FlyBase_FBtr"])

    result.to_csv(OUTPUT_PATH, sep="\t", index=False)

    print("[DONE]")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Genes in input: {genes['FlyBase_FBgn'].nunique()}")
    print(f"Genes with at least one mapping: {result['FlyBase_FBtr'].notna().groupby(result['FlyBase_FBgn']).any().sum()}")
    print(f"Transcript rows saved: {len(result)}")


if __name__ == "__main__":
    main()