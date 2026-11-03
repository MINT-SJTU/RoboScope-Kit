"""Download and verify the pinned assets and checkpoint into a RoboDojo checkout (Python 3.11+)."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
CONFIG = tomllib.loads((HERE / "recipe.toml").read_text(encoding="utf-8"))
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(url):
    for attempt in range(6):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "roboscope-robodojo-recipe/1"}),
                timeout=90,
            ) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))


def inventory(endpoint, kind):
    config = CONFIG["dataset"]
    api_kind = "datasets"
    data = request_json(
        f"{endpoint}/api/{api_kind}/{config['repo_id']}/revision/{config['revision']}?blobs=true"
    )
    if data.get("sha") != config["revision"]:
        raise ValueError("Remote metadata revision differs from recipe pin")
    files = []
    for entry in data["siblings"]:
        name = entry["rfilename"]
        selected = (
            (kind == "assets" and name.startswith("Assets/"))
            or (kind == "checkpoint" and name.startswith(config["checkpoint"] + "/"))
        )
        if not selected:
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid remote path: {name}")
        lfs = entry.get("lfs")
        files.append(
            {
                "path": name,
                "bytes": entry["size"],
                "hash": lfs["sha256"] if lfs else entry["blobId"],
                "algorithm": "sha256" if lfs else "git-blob-sha1",
            }
        )
    if not files:
        raise ValueError(f"Empty inventory for {kind}")
    files.sort(key=lambda entry: entry["path"])
    validate_inventory(files, kind)
    return files


def validate_inventory(files, kind):
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != CONFIG["inventory_sha256"][kind]:
        raise ValueError(f"{kind}: inventory differs from the official pinned metadata")


def file_hash(path, algorithm):
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-blob-sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path, entry):
    return (
        path.is_file()
        and path.stat().st_size == entry["bytes"]
        and file_hash(path, entry["algorithm"]) == entry["hash"]
    )


def fetch_file(endpoint, kind, entry, destination, transport, downloader="urllib"):
    target = destination / entry["path"]
    if valid_file(target, entry):
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    config = CONFIG["dataset"]
    prefix = "datasets/"
    url = f"{endpoint}/{prefix}{config['repo_id']}/resolve/{config['revision']}/{quote(entry['path'])}"
    mirror_kind = "datasets"
    mirror_url = (
        f"https://modelscope.cn/{mirror_kind}/{config['repo_id']}/resolve/master/{quote(entry['path'])}"
    )
    for attempt in range(8):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= entry["bytes"]:
                if valid_file(partial, entry):
                    partial.replace(target)
                    return "downloaded"
                offset = 0
            headers = {"User-Agent": "roboscope-robodojo-recipe/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            # ModelScope is only a byte transport. The accepted file hashes
            # remain those of the immutable, independently locked HF inventory.
            selected_url = mirror_url if transport == "modelscope" and attempt % 2 == 0 else url
            if downloader == "aria2" and entry["bytes"] >= 64 * 1024 * 1024:
                result = subprocess.run(
                    [
                        "aria2c", "--continue=true", "--allow-overwrite=true",
                        "--auto-file-renaming=false", "--file-allocation=none",
                        "--split=16", "--max-connection-per-server=16", "--min-split-size=8M",
                        "--max-tries=2", "--retry-wait=2", "--connect-timeout=20", "--timeout=60",
                        "--auto-save-interval=10", "--summary-interval=0",
                        "--console-log-level=warn", "--download-result=hide",
                        "--dir", str(partial.parent), "--out", partial.name, selected_url,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    raise OSError(f"aria2 exited {result.returncode}: {result.stderr[-300:]}")
            else:
                with urlopen(Request(selected_url, headers=headers), timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    content_range = response.headers.get("Content-Range", "")
                    if append and not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Incorrect range response")
                    with partial.open("ab" if append else "wb") as stream:
                        while block := response.read(4 * 1024 * 1024):
                            stream.write(block)
            if not valid_file(partial, entry):
                # A complete response may belong to a different mirror version.
                # Its bytes must not become a prefix for the pinned fallback.
                partial.unlink()
                partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
                raise ValueError(f"Checksum/size mismatch for {entry['path']}")
            partial.replace(target)
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, IncompleteRead) as error:
            if attempt == 7:
                raise RuntimeError(f"Download failed: {entry['path']}: {type(error).__name__}") from error
            delay = min(2**attempt, 30)
            if isinstance(error, HTTPError) and error.code == 429:
                delay = max(delay, min(int(error.headers.get("Retry-After", "60")), 300))
            time.sleep(delay)


def download(
    root, receipts, endpoint, kind, workers, verify_only=False, transport="primary", inventories=None,
    downloader="urllib",
):
    receipt = receipts / f"{kind}-inventory.json"
    if verify_only:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        expected_revision = CONFIG["dataset"]["revision"]
        if record["revision"] != expected_revision:
            raise ValueError("Receipt revision differs from recipe")
        files = record["files"]
        validate_inventory(files, kind)
    elif inventories is not None:
        files = inventories[kind]
        validate_inventory(files, kind)
    else:
        files = inventory(endpoint, kind)
    destination = root
    total_bytes = sum(entry["bytes"] for entry in files)
    print(f"{kind}: {len(files)} files, {total_bytes} bytes", flush=True)
    start = time.perf_counter()
    completed = reused = copied = completed_bytes = 0
    errors = []
    content_keys = {(entry["algorithm"], entry["hash"], entry["bytes"]) for entry in files}
    content_locks = {key: Lock() for key in content_keys}
    verified_sources = {}

    def process(entry):
        if verify_only:
            if not valid_file(destination / entry["path"], entry):
                raise ValueError(f"Missing or corrupt: {entry['path']}")
            return "verified"
        target = destination / entry["path"]
        key = (entry["algorithm"], entry["hash"], entry["bytes"])
        with content_locks[key]:
            source = verified_sources.get(key)
            if source is not None and not valid_file(target, entry):
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(source, partial)
                if not valid_file(partial, entry):
                    raise ValueError(f"Local copy checksum mismatch: {entry['path']}")
                partial.replace(target)
                return "copied"
            outcome = fetch_file(endpoint, kind, entry, destination, transport, downloader)
            verified_sources[key] = target
            return outcome

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, entry): entry for entry in files}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                outcome = future.result()
                completed += 1
                completed_bytes += entry["bytes"]
                reused += outcome == "reused"
                copied += outcome == "copied"
            except (OSError, RuntimeError, ValueError) as error:
                errors.append({"path": entry["path"], "error": str(error)})
            if (completed + len(errors)) % 100 == 0 or completed + len(errors) == len(files):
                print(
                    f"{kind}: ok={completed}/{len(files)} errors={len(errors)} "
                    f"bytes={completed_bytes}/{total_bytes}",
                    flush=True,
                )
                write_json(
                    receipts / f"{kind}-progress.json",
                    {
                        "complete": False,
                        "verified_files": completed,
                        "expected_files": len(files),
                        "verified_bytes": completed_bytes,
                        "expected_bytes": total_bytes,
                        "errors": errors,
                        "elapsed_s": time.perf_counter() - start,
                    },
                )
    result = {
        "kind": kind,
        "complete": not errors,
        "expected_files": len(files),
        "verified_files": completed,
        "expected_bytes": total_bytes,
        "verified_bytes": completed_bytes,
        "reused_files": reused,
        "copied_files": copied,
        "elapsed_s": time.perf_counter() - start,
        "errors": errors,
        "revision": CONFIG["dataset"]["revision"],
        "files": files,
    }
    write_json(receipt if not verify_only else receipts / f"{kind}-verification.json", result)
    write_json(
        receipts / f"{kind}-progress.json",
        {k: v for k, v in result.items() if k != "files"},
    )
    if errors:
        raise RuntimeError(f"{kind}: {len(errors)} files failed; see {receipt}")
    return {k: v for k, v in result.items() if k not in {"files", "errors"}}


def main():
    global CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["assets", "model", "verify"])
    parser.add_argument("root", type=Path, help="RoboDojo checkout installed by upstream scripts/install.sh")
    parser.add_argument("--recipe", type=Path, default=HERE / "recipe.toml")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--transport", choices=["primary", "modelscope"], default="primary")
    parser.add_argument("--downloader", choices=["urllib", "aria2"], default="urllib")
    parser.add_argument(
        "--inventory-file", type=Path, help="offline inventory JSON keyed by assets/checkpoint"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    CONFIG = tomllib.loads(args.recipe.read_text(encoding="utf-8"))
    inventories = json.loads(args.inventory_file.read_text(encoding="utf-8")) if args.inventory_file else None
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be between 1 and 128")
    if args.downloader == "aria2" and shutil.which("aria2c") is None:
        parser.error("--downloader aria2 requires aria2c on PATH")
    root = args.root.expanduser().resolve()
    receipts = root.parent / "evidence" / CONFIG.get("evidence_subdir", "preparation")
    receipts.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not (root / "scripts/robodojo.sh").is_file():
        parser.error("Clone RoboDojo and run its scripts/install.sh first")
    model_kinds = ["checkpoint"]
    kinds = ["assets"] if args.phase == "assets" else model_kinds
    if args.phase == "verify":
        kinds = ["assets", *model_kinds]
    result = {
        kind: download(
            root,
            receipts,
            args.endpoint.rstrip("/"),
            kind,
            args.workers,
            args.phase == "verify",
            args.transport,
            inventories,
            args.downloader,
        )
        for kind in kinds
    }
    summary = {
        "phase": args.phase,
        "elapsed_s": time.perf_counter() - start,
        "recipe_sha256": file_hash(args.recipe, "sha256"),
        "result": result,
    }
    write_json(receipts / f"{args.phase}-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
