"""Summarize diagnostic spans; inclusive and parallel times are not additive."""

import argparse
import json
from pathlib import Path


def summarize(directory: Path) -> dict:
    groups = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            name = record["kind"]
            if name == "server":
                name += ":" + str(record["operation"])
            group = groups.setdefault(name, {"records": 0, "failed_captures": 0, "spans": {}})
            group["records"] += 1
            group["failed_captures"] += not record["ok"]
            for name, value in record["spans"].items():
                total = group["spans"].setdefault(name, {"seconds": 0.0, "count": 0})
                total["seconds"] += value["seconds"]
                total["count"] += value["count"]
    if not groups:
        raise ValueError(f"no diagnostic records in {directory}")
    for group in groups.values():
        for span in group["spans"].values():
            span["mean_ms"] = 1000 * span["seconds"] / span["count"]
    return {"groups": groups, "note": "Nested inclusive durations and parallel sums are not wall time."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory), indent=2)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result + "\n")
    else:
        print(result)
