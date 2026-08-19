"""
Download data files from Zenodo and GitHub into data/mif/.
Run from within HiGINE/ (one level above scripts/).
"""

import json
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/mif")

ZENODO_RECORD_ID = "20727573"
ZENODO_FILES = [
    ("BOMI2_all_cells_TIL.csv", "cell_data.csv"),
]

GITHUB_FILES = [
    (
        "https://raw.githubusercontent.com/MIDA-group/lung_cancer_BOMI2_dataset/main/binary_survival_prediction/Clinical_data_with_labels.csv",
        "clinical_data.csv",
    ),
]

GITHUB_DIRS = [
    ("MIDA-group", "lung_cancer_BOMI2_dataset", "binary_survival_prediction/10foldcrossval", "main", "folds"),
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists, skipping: {dest}")
        return
    print(f"  {url}\n    -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_github_dir(owner: str, repo: str, path: str, branch: str, dest_dir: Path) -> None:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as response:
        entries = json.loads(response.read())

    for entry in entries:
        if entry["type"] == "file":
            dest = dest_dir / entry["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            download(entry["download_url"], dest)
        elif entry["type"] == "dir":
            download_github_dir(owner, repo, entry["path"], branch, dest_dir / entry["name"])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading from Zenodo...")
    for remote_filename, local_path in ZENODO_FILES:
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{remote_filename}"
        download(url, DATA_DIR / local_path)

    print("Downloading individual files from GitHub...")
    for url, rel_path in GITHUB_FILES:
        dest = DATA_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(url, dest)

    print("Downloading directories from GitHub...")
    for owner, repo, path, branch, local_dir in GITHUB_DIRS:
        print(f"  {owner}/{repo}/{path} @ {branch}")
        download_github_dir(owner, repo, path, branch, DATA_DIR / local_dir)

    print("Done.")


if __name__ == "__main__":
    main()
