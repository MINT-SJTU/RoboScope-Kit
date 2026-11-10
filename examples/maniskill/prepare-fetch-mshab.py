"""Download the two immutable official MS-HAB PPO Open checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def valid(path: Path, record: dict) -> bool:
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"])


def download(destination: Path, proxy: str | None = None) -> None:
    pins = json.loads(Path(__file__).with_name("fetch-mshab-checkpoints.json").read_text("utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else None))
    for record in pins["models"].values():
        target = destination / record["path"]
        if valid(target, record):
            print(f"Verified {target.name}", flush=True)
            continue
        url = (f"https://huggingface.co/{pins['repository']}/resolve/{pins['revision']}/"
               f"{record['remote_path']}?download=true")
        temporary = target.with_suffix(".download")
        for attempt in range(4):
            try:
                started = last_report = time.monotonic()
                size = 0
                request = urllib.request.Request(url, headers={"User-Agent": "RoboScope-ManiSkill/1"})
                with opener.open(request, timeout=120) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                        size += len(block)
                        now = time.monotonic()
                        if now-last_report >= 15:
                            print(f"{target.name}: {size}/{record['bytes']} bytes, "
                                  f"{size/max(now-started, .001)/2**20:.2f} MiB/s", flush=True)
                            last_report = now
                if not valid(temporary, record):
                    raise ValueError(f"Published size/digest mismatch: {target.name}")
                temporary.replace(target)
                print(f"Downloaded and verified {target.name} in {time.monotonic()-started:.1f}s", flush=True)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    (destination / "models.json").write_text(
        json.dumps({"schema_version": 1, "models": pins["models"]}, indent=2)+"\n", "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--proxy")
    arguments = parser.parse_args()
    download(arguments.destination, arguments.proxy)
