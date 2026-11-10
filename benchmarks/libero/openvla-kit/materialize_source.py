"""Materialize the pinned LIBERO tree, reusing only Git-verified cache files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source_url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}"

    def materialize(item: tuple[str, dict]) -> bool:
        relative, expected = item
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        target = args.destination / relative_path
        candidates = [target]
        if args.cache is not None:
            candidates.append(args.cache / relative_path)

        def valid(data: bytes) -> bool:
            header = f"blob {len(data)}\0".encode()
            return (
                len(data) == expected["size"]
                and hashlib.sha1(header + data).hexdigest() == expected["git_sha1"]
            )

        for candidate in candidates:
            if candidate.is_file():
                data = candidate.read_bytes()
                if valid(data):
                    if candidate != target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                    return True
        with urllib.request.urlopen(f"{source_url}/{relative}", timeout=90) as response:
            data = response.read()
        if not valid(data):
            raise ValueError(f"Source content does not match {manifest['commit']}: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        reused = list(pool.map(materialize, manifest["files"].items()))
    print(json.dumps({"commit": manifest["commit"], "files": len(reused), "reused": sum(reused)}))


if __name__ == "__main__":
    main()
