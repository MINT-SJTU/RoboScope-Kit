# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Stage read-only asset files without letting build output modify the source."""
import os
import stat
import sys
from pathlib import Path


def stage(source: Path, target: Path) -> int:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise ValueError('asset destination already exists')
    if target.resolve().is_relative_to(source):
        raise ValueError('asset destination must be outside the source')
    target.mkdir(parents=True)
    count = 0
    for directory, dirs, files in os.walk(source, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(source)
        output = target / relative
        output.mkdir(parents=True, exist_ok=True)
        for name in dirs:
            if (current / name).is_symlink():
                raise ValueError(f'source contains a directory symlink: {relative / name}')
        generated = {name.removesuffix('_tmp.yml') + '.yml'
                     for name in files if name.endswith('_tmp.yml')}
        for name in files:
            # The official update script writes these next to a *_tmp.yml;
            # they must be newly created inside the isolated runtime.
            if relative.parts and relative.parts[0] == 'Robots' and name in generated:
                continue
            if name.endswith('.partial'):
                continue
            src = current / name
            resolved = src.resolve(strict=True)
            if not resolved.is_relative_to(source) or not stat.S_ISREG(resolved.stat().st_mode):
                raise ValueError(f'invalid source asset: {relative / name}')
            if os.access(resolved, os.W_OK):
                raise PermissionError(f'source asset is writable by the build user: {relative / name}')
            (output / name).symlink_to(resolved)
            count += 1
            if count % 2000 == 0:
                print(f'linked {count} read-only asset files', flush=True)
    if not count:
        raise ValueError('source asset tree was empty')
    for template in (target / 'Robots').rglob('*_tmp.yml'):
        generated = template.with_name(template.name.removesuffix('_tmp.yml') + '.yml')
        if generated.is_symlink():
            raise ValueError('generated robot configuration would overwrite a source asset')
    return count


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: stage-pyrua-asset-links.py SOURCE DESTINATION')
    print(f'PASS staged {stage(Path(sys.argv[1]), Path(sys.argv[2]))} read-only assets', flush=True)
