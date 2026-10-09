"""Where results go: scratch space, mirroring the path of the notebook that makes them."""

from __future__ import annotations

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
