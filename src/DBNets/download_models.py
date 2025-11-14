import argparse
import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import tarfile
from .paths import get_data_dir


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    return name or "downloaded_file"


def download_file(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url)
    dest = out_dir / filename

    print(f"Downloading {url}")
    print(f" -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print("   OK")
    except Exception as e:
        print(f"   FAILED: {e}")

    return dest



def main():

    out_dir = get_data_dir()

    # Ensure directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving files to: {out_dir}")

    files = ["red_targ.npy", "training_set.npy", "dbnets2.tar"]

    tar_path = None
    base_url = "http://dbnets.fisica.unimi.it/dbnets2.0_models/data"

    for file in files:
        dest = download_file(f"{base_url}/{file}", out_dir)
        if dest.name == "dbnets2.tar":
            tar_path = dest

    # Untar dbnets2.tar if it was downloaded
    if tar_path and tar_path.exists():
        print(f"Extracting {tar_path} ...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=out_dir)  # or out_dir / "dbnets2" if you prefer
            print("   Extraction OK")
        except Exception as e:
            print(f"   Extraction FAILED: {e}")


if __name__ == "__main__":
    main()
