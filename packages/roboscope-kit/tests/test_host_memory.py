from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark import resources

GiB = 1024**3


@pytest.fixture(autouse=True)
def fixed_meminfo(monkeypatch):
    monkeypatch.setattr(resources, "_meminfo", lambda: {"MemTotal": 64 * GiB, "MemAvailable": 40 * GiB})


def make_cgroup(root: Path, relative: str, **files: str) -> Path:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (directory / name.replace("_", ".", 1)).write_text(value + "\n", encoding="utf-8")
    return directory


def proc_cgroup(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "proc-self-cgroup"
    path.write_text(text, encoding="utf-8")
    return path


def test_limit_is_the_smallest_of_the_own_cgroup_and_its_ancestors(tmp_path):
    root = tmp_path / "cgroup"
    make_cgroup(root, "")
    make_cgroup(root, "job.slice", memory_max=str(8 * GiB), memory_high="max")
    make_cgroup(root, "job.slice/step.scope", memory_max="max", memory_high=str(7 * GiB))
    own = proc_cgroup(tmp_path, "0::/job.slice/step.scope\n")

    assert resources.host_memory_limit(root, own) == 7 * GiB


def test_limit_reads_the_namespace_root_inside_a_container(tmp_path):
    root = make_cgroup(tmp_path / "cgroup", "", memory_max="max", memory_high=str(70 * GiB))
    own = proc_cgroup(tmp_path, "0::/\n")

    assert resources.host_memory_limit(root, own) == 64 * GiB  # RAM is below memory.high here
    (root / "memory.high").write_text(str(30 * GiB), encoding="utf-8")
    assert resources.host_memory_limit(root, own) == 30 * GiB


def test_limit_is_ram_without_a_unified_hierarchy(tmp_path):
    root = make_cgroup(tmp_path / "cgroup", "", memory_max=str(8 * GiB))
    own = proc_cgroup(tmp_path, "12:memory:/job\n")

    assert resources.host_memory_limit(root, own) == 64 * GiB


def test_used_memory_keeps_shared_memory(tmp_path):
    root = tmp_path / "cgroup"
    make_cgroup(root, "")
    make_cgroup(
        root,
        "job.slice",
        memory_current=str(10 * GiB),
        memory_stat=f"anon {6 * GiB}\nfile {4 * GiB}\nshmem {1 * GiB}",
    )
    own = proc_cgroup(tmp_path, "0::/job.slice\n")

    # Page cache that can be reclaimed is file minus shmem: 3 GiB.
    assert resources.host_memory_used(root, own) == 7 * GiB


def test_used_memory_falls_back_to_the_machine_without_a_memory_controller(tmp_path):
    root = make_cgroup(tmp_path / "cgroup", "")
    own = proc_cgroup(tmp_path, "0::/\n")

    assert resources.host_memory_used(root, own) == 24 * GiB
