"""Prepare the fixed train/test subsets from public NPZ files or matching PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2] / "problems/image_classification"
RESOLUTIONS = (28, 64, 128, 224)
MD5 = {28: "28209eda62fecd6e6a2d98b1501bb15f", 64: "8f4eceb4ccffa70c672198ea285246c6", 128: "05b46931834c231683c68f40c47b2971", 224: "d6a3c71de1b945ea11211b03746c1fe1"}


def checksum(path, algorithm="sha256"):
    """Hash a file without loading the full NPZ archive into memory."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def download_npz(resolution, cache):
    """Download the official resolution-specific archive and verify its checksum."""
    name = "pneumoniamnist" + (f"_{resolution}" if resolution != 28 else "") + ".npz"
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = "https://zenodo.org/records/10519652/files/" + name + "?download=1"
        for attempt in range(3):
            try:
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
                    temporary = Path(output.name)
                    with urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, output)
                if checksum(temporary, "md5") != MD5[resolution]:
                    raise ValueError("Downloaded archive checksum mismatch")
                temporary.replace(path)
                break
            except Exception:
                if "temporary" in locals():
                    temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    if checksum(path, "md5") != MD5[resolution]:
        raise ValueError(f"NPZ checksum mismatch: {path.name}")
    return path


def prepare(args):
    """Materialize checked images and portable manifests without overwriting a dataset."""
    destination = args.output.resolve()
    if destination.exists() and any(p.name != ".gitkeep" for p in destination.iterdir()):
        raise ValueError("Output must be empty; use a new directory to preserve existing data")
    provenance = {"dataset": "PneumoniaMNIST", "source": "https://zenodo.org/records/10519652", "datasets": []}
    with tempfile.TemporaryDirectory(prefix="pneumonia-data-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for resolution in args.resolutions:
            archive = None
            if args.from_images is None:
                if args.download:
                    npz_path = download_npz(resolution, args.npz_dir)
                else:
                    name = "pneumoniamnist" + (f"_{resolution}" if resolution != 28 else "") + ".npz"
                    npz_path = args.npz_dir / name
                    if checksum(npz_path, "md5") != MD5[resolution]:
                        raise ValueError(f"NPZ checksum mismatch: {name}")
                archive = np.load(npz_path, allow_pickle=False)
            try:
                for split in ("train", "test"):
                    rows = [json.loads(line) for line in (TASK / "manifests" / f"r{resolution}" / f"{split}.jsonl").read_text().splitlines()]
                    images = archive[split + "_images"] if archive is not None else None
                    labels = archive[split + "_labels"].reshape(-1) if archive is not None else None
                    for row in rows:
                        target = staging / f"r{resolution}" / split / row["rel_path"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if args.from_images is not None:
                            source = args.from_images / f"r{resolution}" / split / row["rel_path"]
                            with Image.open(source) as image:
                                pixels = np.asarray(image).copy()
                            shutil.copyfile(source, target)
                        else:
                            index = row["original_index"]
                            pixels = images[index]
                            if int(labels[index]) != row["label_id"]:
                                raise ValueError(f"Label mismatch: r{resolution}/{split}/{index}")
                            Image.fromarray(pixels).save(target, format="PNG")
                        if pixels.shape != (resolution, resolution) or pixels.dtype != np.uint8 or hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]:
                            raise ValueError(f"Pixel mismatch: r{resolution}/{split}/{row['original_index']}")
                        row["sha256"] = checksum(target)
                    manifest = staging / f"r{resolution}" / split / "manifest.jsonl"
                    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
                    provenance["datasets"].append({"resolution": resolution, "split": split, "images": len(rows), "manifest_sha256": checksum(manifest)})
                    print(f"Verified r{resolution}/{split}: {len(rows)} images", flush=True)
            finally:
                if archive is not None:
                    archive.close()
        shutil.copyfile(TASK / "manifests/initial_prompts.json", staging / "initial_prompts.json")
        shutil.copyfile(TASK / "DATA_ATTRIBUTION.md", staging / "DATA_ATTRIBUTION.md")
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            shutil.move(str(path), str(destination / path.name))


def main():
    """Parse the data source and requested resolutions for a local preparation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, choices=RESOLUTIONS, default=list(RESOLUTIONS))
    parser.add_argument("--output", type=Path, default=TASK / "data")
    parser.add_argument("--npz-dir", type=Path, default=Path(__file__).resolve().parent / ".cache/datasets")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download", action="store_true")
    source.add_argument("--from-images", type=Path, help="Existing matching r28/r64/r128/r224 PNG tree")
    args = parser.parse_args()
    if len(set(args.resolutions)) != len(args.resolutions):
        parser.error("Resolutions must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepare(args)


if __name__ == "__main__":
    main()
