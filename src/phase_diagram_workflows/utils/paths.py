"""Where results go: scratch space, mirroring the path of the notebook that makes them."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Optional, Union


def scratch_directory(notebook_directory: Union[str, Path], scratch_root: Union[str, Path], name: str, home: Optional[Union[str, Path]] = None) -> Path:
    """``<scratch_root>/<notebook_directory relative to home>/<name>``.

    Results and caches of a notebook do not belong on the (small) home file system, and mirroring the notebook's
    path under `scratch_root` makes it obvious which notebook a folder belongs to. `home` defaults to the user's home.

    Raises
    ------
    ValueError
        If `notebook_directory` is not below `home`.
    """
    home = Path(home).resolve() if home is not None else Path.home().resolve()
    relative = Path(notebook_directory).resolve().relative_to(home)
    return Path(scratch_root) / relative / name


def archive_directory(directory: Union[str, Path]) -> Optional[Path]:
    """Pack `directory` into ``<directory>.tar.gz`` next to it and remove it; the archive's path, or None if there is no such folder.

    For caches of finished runs: many small files become one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    archive = directory.with_name(directory.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(directory, arcname=directory.name)
    shutil.rmtree(directory)
    return archive
