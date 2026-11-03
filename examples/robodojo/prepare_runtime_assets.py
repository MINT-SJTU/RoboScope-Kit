"""Run the upstream embodiment path generator with checked backup and rollback."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["apply", "check", "rollback"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root.parent / "evidence" / "runtime-assets"
    manifest = evidence / "paths.json"
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record["root"] != str(root):
            raise SystemExit("Asset record belongs to a different runtime")
    elif args.action == "apply":
        files = {}
        for template in sorted((root / "Assets/Robots").rglob("*_tmp.yml")):
            target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
            relative = str(target.relative_to(root))
            data = (
                template.read_text(encoding="utf-8")
                .replace("${ASSETS_PATH}", str(root))
                .replace("$ASSETS_PATH", str(root))
            )
            backup = evidence / "originals" / relative
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
            files[relative] = {"before": digest(target), "after": hashlib.sha256(data.encode()).hexdigest()}
        if not files:
            raise SystemExit("No upstream embodiment templates found")
        record = {"root": str(root), "files": files}
        evidence.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    else:
        raise SystemExit("Run apply first")
    files = record["files"]
    states = [s for s in ("before", "after") if all(digest(root / p) == h[s] for p, h in files.items())]
    if not states and not (
        args.action == "rollback"
        and all(digest(root / p) in (h["before"], h["after"]) for p, h in files.items())
    ):
        raise SystemExit("Generated assets were modified or partially applied; preserve the backups")
    if args.action == "check":
        print(f"asset paths: {states[0]} ({len(files)} files)")
        return
    desired = "after" if args.action == "apply" else "before"
    if desired not in states:
        if args.action == "apply":
            subprocess.run([sys.executable, "utils/update_embodiment_config_path.py"], cwd=root, check=True)
        else:
            # Check every backup before changing any target, including partial applies.
            for path, hashes in files.items():
                if hashes["before"] is not None and digest(evidence / "originals" / path) != hashes["before"]:
                    raise SystemExit(f"Corrupt rollback backup: {path}")
            for path, hashes in files.items():
                target = root / path
                if hashes["before"] is None:
                    target.unlink(missing_ok=True)
                else:
                    backup = evidence / "originals" / path
                    shutil.copyfile(backup, target)
    if not all(digest(root / p) == h[desired] for p, h in files.items()):
        raise SystemExit("Generated asset digest check failed")
    print(f"asset paths: {desired} ({len(files)} files)")


if __name__ == "__main__":
    main()
