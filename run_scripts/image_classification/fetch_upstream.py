"""Download and verify the pinned engine outside the experiment repository."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

from med_evolve.upstream_integrity import verify_upstream

ROOT = Path(__file__).resolve().parent


def main():
    """Extract a verified release to a new directory, or verify an existing checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT.parents[1].parent / "gigaevo-core")
    args = parser.parse_args()
    pin = ROOT / "upstream_version.json"
    config = json.loads(pin.read_text())
    destination = args.destination.resolve()
    if destination.exists():
        print(json.dumps(verify_upstream(destination, pin)))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = config["repository"].replace("https://github.com/", "https://codeload.github.com/") + "/zip/" + config["commit"]
    with tempfile.TemporaryDirectory(prefix="engine-download-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.zip"
        with urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe path in upstream archive")
            source.extractall(temporary / "extracted")
        roots = list((temporary / "extracted").iterdir())
        if len(roots) != 1:
            raise ValueError("Expected one upstream archive root")
        result = verify_upstream(roots[0], pin)
        roots[0].rename(destination)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
