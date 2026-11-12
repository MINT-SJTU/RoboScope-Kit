"""Native resource sampler, copied unchanged for standalone experiment use."""
import csv
import json
import subprocess
import time
from pathlib import Path

def sample_resources(args, child, stop, rows):
    import psutil

    fields = (
        "unix_s",
        "monotonic_s",
        "query_elapsed_s",
        "phase",
        "gpu_uuid",
        "gpu_index",
        "gpu_memory_mib",
        "gpu_util_pct",
        "process_tree_rss_bytes",
        "process_tree_cpu_s",
        "host_available_ram_bytes",
        "cgroup_memory_bytes",
        "error",
    )
    with (args.output_dir / "resources.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            started = time.monotonic()
            row = {"unix_s": time.time(), "monotonic_s": started, "phase": "startup"}
            try:
                phase_path = args.output_dir / "phase.json"
                if phase_path.is_file():
                    row["phase"] = json.loads(phase_path.read_text())["phase"]
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=uuid,index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                uuid, index, memory, utilization = result.stdout.strip().split(",")
                row.update(
                    gpu_uuid=uuid.strip(),
                    gpu_index=index.strip(),
                    gpu_memory_mib=float(memory),
                    gpu_util_pct=float(utilization),
                )
                processes = [psutil.Process(child.pid)]
                processes.extend(processes[0].children(recursive=True))
                row["process_tree_rss_bytes"] = sum(p.memory_info().rss for p in processes)
                cpu = [p.cpu_times() for p in processes]
                row["process_tree_cpu_s"] = sum(p.user + p.system for p in cpu)
                row["host_available_ram_bytes"] = psutil.virtual_memory().available
                memory_path = Path("/sys/fs/cgroup/memory.current")
                if memory_path.is_file():
                    row["cgroup_memory_bytes"] = int(memory_path.read_text())
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["query_elapsed_s"] = time.monotonic() - started
            rows.append(row)
            writer.writerow(row)
            file.flush()
            stop.wait(args.sample_interval)
