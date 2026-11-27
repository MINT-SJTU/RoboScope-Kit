# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Summarize ROBOSCOPE_TIMING_DIR JSONL stage spans; inclusive and parallel times are not additive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(directory: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            group_name = str(record["kind"])
            if group_name == "server":
                group_name = f"{group_name}:{record.get('operation')}"
            group = groups.setdefault(group_name, {"records": 0, "failed_captures": 0, "spans": {}})
            group["records"] += 1
            group["failed_captures"] += int(not record.get("ok", False))
            for span_name, value in record.get("spans", {}).items():
                total = group["spans"].setdefault(span_name, {"seconds": 0.0, "count": 0})
                total["seconds"] += float(value["seconds"])
                total["count"] += int(value["count"])
    if not groups:
        raise ValueError(f"no diagnostic records in {directory}")
    for group in groups.values():
        for span in group["spans"].values():
            span["mean_ms"] = 1000.0 * span["seconds"] / span["count"]
    return {"groups": groups, "note": "Nested inclusive durations and parallel sums are not wall time."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory populated via ROBOSCOPE_TIMING_DIR")
    parser.add_argument("--output", type=Path, help="write JSON summary to a new file")
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory), indent=2)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
