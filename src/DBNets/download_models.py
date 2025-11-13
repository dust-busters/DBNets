import argparse
import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import tarfile


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    return name or "downloaded_file"


def download_file(url: str, out_dir: Path):
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
    parser = argparse.ArgumentParser(description="Download models used by DBNets2.0")
    parser.add_argument(
        "-p", "--save-path",
        default="~/.cache/DBNets",
        help="Directory to save files (default: ~/.cache/DBNets)",
    )
    args = parser.parse_args()

    # Expand ~ and make sure directory exists
    out_dir = Path(args.save_path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = ["red_targ.npy", "training_set.npy", "dbnets2.tar"]

    tar_path = None
    for file in files:
        dest = download_file(
            f"http://dbnets.fisica.unimi.it/dbnets2.0_models/data/{file}",
            out_dir,
        )
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
