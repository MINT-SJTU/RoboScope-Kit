"""Apply/check/roll back a RoboDojo runtime patch on the installed checkout."""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PATCHES = {
    "measurement": "0001-measurement.patch",
    "parallel": "0003-parallel-autotune.patch",
    "async-video": "0004-async-video.patch",
    "episode-count": "0005-episode-count.patch",
    "unpaced-render": "0006-unpaced-render.patch",
    "physics-threads": "0007-physics-threads.patch",
    "control-state": "0008-control-state.patch",
}
SOURCE_DIRS = {
    "robodojo": ".",
    "xpolicylab": "XPolicyLab",
    "isaaclab": "third_party/IsaacLab",
    "curobo": "third_party/curobo",
}


def patch_metadata(name):
    """Read affected paths from the committed diff; no duplicate manifest."""
    patch = HERE / "patches" / PATCHES[name]
    lines = patch.read_text(encoding="utf-8").splitlines()
    files = {}
    for index, line in enumerate(lines):
        if not line.startswith("--- "):
            continue
        before = line.removeprefix("--- ")
        after = lines[index + 1].removeprefix("+++ b/")
        path = PurePosixPath(after)
        if (
            not lines[index + 1].startswith("+++ b/")
            or path.is_absolute() or ".." in path.parts or "\\" in after
            or before not in ("/dev/null", f"a/{after}") or after in files
        ):
            raise ValueError(f"Unsupported patch path: {line}")
        files[after] = before == "/dev/null"
    if not files:
        raise ValueError(f"No files in patch: {name}")
    return {"sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "files": files}


def expected_contents(root, name):
    """Reconstruct both states from the checked-out Git blobs in an isolated directory.

    The sources come from the upstream installer, whose submodules follow their main branches, so the
    "before" state is each file at the checkout's HEAD; `git apply` then proves the patch still fits it.
    """
    files = patch_metadata(name)["files"]
    owners = {}
    for filename in files:
        owners[filename] = max(
            (source for source, directory in SOURCE_DIRS.items()
             if directory == "." or filename.startswith(directory + "/")),
            key=lambda source: len(SOURCE_DIRS[source]),
        )
    before = {}
    with tempfile.TemporaryDirectory(prefix="robodojo-patch-") as directory:
        scratch = Path(directory)
        for filename, added in files.items():
            if added:
                before[filename] = None
                continue
            source = owners[filename]
            relative = PurePosixPath(filename).relative_to(SOURCE_DIRS[source])
            before[filename] = subprocess.check_output(
                ["git", "-C", str(root / SOURCE_DIRS[source]), "show",
                 f"HEAD:{relative.as_posix()}"],
            )
            target = scratch / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(before[filename])
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", str(HERE / "patches" / PATCHES[name])],
            cwd=scratch, check=True,
        )
        after = {filename: (scratch / filename).read_bytes() for filename in files}
    return {"before": before, "after": after}


def matches(root, expected):
    for filename, content in expected.items():
        path = root / filename
        if path.is_symlink():
            return False
        if content is None:
            if path.exists():
                return False
        elif not path.is_file() or path.read_bytes() != content:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["apply", "check", "rollback"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--patch", choices=list(PATCHES), default="measurement")
    args = parser.parse_args()
    root = args.root.resolve()
    expected = expected_contents(root, args.patch)
    states = [state for state, content in expected.items() if matches(root, content)]
    if not states:
        raise SystemExit("Runtime is modified or only partly patched; refusing to overwrite it")
    state = states[0]
    if args.action == "check":
        print(f"{args.patch} patch: {state}")
        return
    desired = "after" if args.action == "apply" else "before"
    if state != desired:
        command = ["git", "-c", "core.autocrlf=false", "-C", str(root), "apply"]
        if args.action == "rollback":
            command.append("--reverse")
        patch = HERE / "patches" / PATCHES[args.patch]
        subprocess.run([*command, "--check", str(patch)], check=True)
        subprocess.run([*command, str(patch)], check=True)
    if not matches(root, expected[desired]):
        raise SystemExit("Post-application content validation failed")
    print(f"{args.patch} patch: {desired}")


if __name__ == "__main__":
    main()
