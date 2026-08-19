"""
Download and prepare the IMC (Dataset 2) data into data/imc/.
Run from within HiGINE/ (one level above scripts/).

Downloads the raw archive into data/imc/raw/, then derives cell_data.csv and
clinical_data.csv from it. Fold splits are built from imc_5parts.csv, a static
patient-to-part assignment stored alongside this script. Only the "LUAD_D*"
(Discovery) keys are used; "LUAD_V*" (Validation) keys have no clinical data.
"""

import logging
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import center_of_mass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_DIR = Path("data/imc")
RAW_DATA_DIR = DATA_DIR / "raw"

ZENODO_URL = "https://zenodo.org/record/7760826/files/LungData.zip"
ZIP_PATH = RAW_DATA_DIR / "LungData.zip"
LUNGDATA_DIR = RAW_DATA_DIR / "LungData"
SEGM_DIR = LUNGDATA_DIR / "LUAD_IMC_Segmentation"
CELLTYPE_DIR = LUNGDATA_DIR / "LUAD_IMC_CellType"
CLINICAL_XLSX = LUNGDATA_DIR / "LUAD Clinical Data.xlsx"

CELL_CSV_PATH = DATA_DIR / "cell_data.csv"
CLINICAL_CSV_PATH = DATA_DIR / "clinical_data.csv"
FOLDS_DIR = DATA_DIR / "folds"

DISCOVERY_PREFIX = "LUAD_D"

# static patient -> part assignment
PARTS_CSV_PATH = Path(__file__).parent / "imc_5parts.csv"
TOTAL_PARTS = 5
TEST_PARTS = 1
VAL_PARTS = 1


def report_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    if block_num % 200 == 0 or downloaded >= total_size:
        print(f"  {pct:3d}% ({downloaded / 1e6:.0f} / {total_size / 1e6:.0f} MB)", end="\r")


def download_and_extract() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if LUNGDATA_DIR.exists():
        logger.info(f"already extracted, skipping: {LUNGDATA_DIR}")
        return

    if not ZIP_PATH.exists():
        logger.info(f"Downloading {ZENODO_URL}\n  -> {ZIP_PATH}")
        urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH, reporthook=report_progress)
        print()
    else:
        logger.info(f"zip already downloaded, skipping download: {ZIP_PATH}")

    logger.info(f"Extracting {ZIP_PATH} -> {RAW_DATA_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as f:
        f.extractall(RAW_DATA_DIR)


def key_to_id(key: str) -> int:
    """'LUAD_D001' -> 1001."""
    return 1000 + int(key.replace(DISCOVERY_PREFIX, ""))


def discovery_keys() -> list[str]:
    return sorted(p.stem for p in CELLTYPE_DIR.glob(f"{DISCOVERY_PREFIX}*.mat"))


def build_clinical_csv() -> pd.DataFrame:
    """Derive clinical_data.csv from the raw clinical Excel file."""
    if CLINICAL_CSV_PATH.exists():
        logger.info(f"clinical CSV already exists, skipping: {CLINICAL_CSV_PATH}")
        return pd.read_csv(CLINICAL_CSV_PATH)

    logger.info(f"Building clinical CSV from {CLINICAL_XLSX}")
    df = pd.read_excel(CLINICAL_XLSX)

    df["ID"] = df["Key"].apply(key_to_id)
    df["OS_M"] = df["Survival or loss to follow-up (years)"] * 12
    df["OS_EVENT"] = df["Death (No: 0, Yes: 1)"].astype(bool)

    # exclude patients with <= 3 months follow-up
    df = df[df["OS_M"] > 3].copy()

    # short- vs long-term survival: died within 3 years -> 0, else 1
    df["label"] = np.nan
    df.loc[(df["OS_M"] <= 36) & df["OS_EVENT"], "label"] = 0
    df.loc[df["OS_M"] > 36, "label"] = 1
    df = df.dropna(subset=["label"])

    # exclude patients missing a stage
    df = df.dropna(subset=["Stage (I-II: 0, III-IV:1)"])

    out = pd.DataFrame({
        "ID": df["ID"],
        "Stage": df["Stage (I-II: 0, III-IV:1)"].astype(int),
        "Follow-up (days)": (df["Survival or loss to follow-up (years)"] * 365).round().astype(int),
        "Dead/Alive": df["OS_EVENT"].map({True: "Dead", False: "Alive"}),
        "label": df["label"].astype(int),
        "censored": (~df["OS_EVENT"]).astype(int),
    })
    out = out.sort_values("ID").reset_index(drop=True)
    out.to_csv(CLINICAL_CSV_PATH, index=False)
    logger.info(f"wrote {len(out)} patients -> {CLINICAL_CSV_PATH}")
    return out


def process_one_key(key: str, all_cell_labels: list[str]) -> pd.DataFrame:
    """Build the per-cell feature rows for one patient/spot."""
    seg = loadmat(SEGM_DIR / key / "nuclei_multiscale.mat")
    cell_id_mask = seg["nucleiOccupancyIndexed"]
    n_cells = int(cell_id_mask.max())

    coords = np.array(center_of_mass(cell_id_mask != 0, cell_id_mask, list(range(1, n_cells + 1))))

    ct = loadmat(CELLTYPE_DIR / f"{key}.mat")
    cell_types = [np.nan if len(x) == 0 else x[0] for x in ct["cellTypes"].squeeze()]

    one_hot = pd.get_dummies(pd.Series(cell_types)).reindex(columns=all_cell_labels, fill_value=False)
    one_hot = one_hot.astype(float)

    df = one_hot
    df["Cell X Position"] = coords[:, 0]
    df["Cell Y Position"] = coords[:, 1]

    # drop label indices with no pixels in the mask (NaN centroid), renumber
    df = df.dropna(subset=["Cell X Position", "Cell Y Position"]).reset_index(drop=True)
    df.insert(0, "Cell ID", range(1, len(df) + 1))
    df["ID"] = f"Lung # {key_to_id(key)}"
    return df


def build_cell_csv() -> None:
    if CELL_CSV_PATH.exists():
        logger.info(f"cell-level CSV already exists, skipping: {CELL_CSV_PATH}")
        return

    keys = discovery_keys()
    logger.info(f"Building cell-level CSV from {len(keys)} Discovery patients")

    # cell-type label vocabulary is shared across all patients/spots
    ref = loadmat(CELLTYPE_DIR / f"{keys[0]}.mat")
    all_cell_labels = sorted(x[0] for x in ref["allLabels"].squeeze())

    first = True
    for i, key in enumerate(keys):
        try:
            df = process_one_key(key, all_cell_labels)
        except Exception:
            logger.exception(f"failed to process {key}, skipping")
            continue
        df.to_csv(CELL_CSV_PATH, mode="w" if first else "a", header=first, index=False)
        first = False
        if (i + 1) % 50 == 0 or i + 1 == len(keys):
            logger.info(f"  {i + 1}/{len(keys)} patients processed")

    logger.info(f"wrote cell-level data -> {CELL_CSV_PATH}")


def build_folds() -> None:
    """Rotate imc_5parts.csv into data/imc/folds/split_N_{train,val,test}.csv."""
    if FOLDS_DIR.exists():
        logger.info(f"folds already exist, skipping: {FOLDS_DIR}")
        return

    logger.info(f"Building folds from {PARTS_CSV_PATH}")
    parts_df = pd.read_csv(PARTS_CSV_PATH)
    parts = {i: parts_df.loc[parts_df["part"] == i, "ID"].tolist() for i in range(TOTAL_PARTS)}

    def offset(i: int, fold: int) -> int:
        return (i + fold) % TOTAL_PARTS

    FOLDS_DIR.mkdir(parents=True)
    for fold in range(TOTAL_PARTS):
        test_ids = [id_ for j in range(TEST_PARTS) for id_ in parts[offset(j, fold)]]
        val_ids = [id_ for j in range(VAL_PARTS) for id_ in parts[offset(j + TEST_PARTS, fold)]]
        train_parts = TOTAL_PARTS - TEST_PARTS - VAL_PARTS
        train_ids = [id_ for j in range(train_parts) for id_ in parts[offset(j + TEST_PARTS + VAL_PARTS, fold)]]

        for split_name, ids in (("test", test_ids), ("val", val_ids), ("train", train_ids)):
            pd.DataFrame({"ID": ids}).to_csv(FOLDS_DIR / f"split_{fold}_{split_name}.csv", index=False)

    logger.info(f"wrote 5-fold splits -> {FOLDS_DIR}")


def main() -> None:
    download_and_extract()
    build_clinical_csv()
    build_cell_csv()
    build_folds()
    logger.info("Done.")


if __name__ == "__main__":
    main()
