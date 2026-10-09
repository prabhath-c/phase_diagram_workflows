"""
`ExecutorSpec`: a picklable recipe for an executorlib executor, shared by everything that submits jobs.

Lives here (not in a feature package) because the temperature-bracket convergence, the relaxation of many
structures and the supercell-size convergence all describe "which executor, with which settings" the same way.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# executorlib executors that can be shut down with `wait=False` while their submitted jobs keep
# running in the scheduler (`check_wait_on_shutdown` rejects `wait=False` for every other one).
_DETACHABLE_EXECUTOR_NAMES = ("SlurmClusterExecutor", "FluxClusterExecutor")


@dataclass(frozen=True)
class ExecutorSpec:
    """A picklable recipe for an executor: which class, with which settings.

    A live executor owns threads and locks, so it cannot be sent to another process. But the
    job that continues a bracket sweep runs later, in a different process, and has to submit the
    next bracket itself -- so it is handed this recipe and builds its own executor from it.
    Checked at construction: a setting that cannot be sent to another process fails here, in the
    notebook, instead of hours later inside a job.

    Parameters
    ----------
    executor_class : type
        The executor class, e.g. ``executorlib.SlurmClusterExecutor``.
    kwargs : Dict[str, Any]
        Keyword arguments for ``executor_class`` (e.g. ``resource_dict``, ``cache_directory``).
    detach : Optional[bool], optional
        Whether a job may hand the next bracket to this executor and exit while it keeps running
        (fire and forget). ``None`` (default) means yes for executorlib's ``SlurmClusterExecutor``
        and ``FluxClusterExecutor`` -- the only ones that survive their submitting process -- and
        no for anything else. With no, the submitting job stays alive until the next bracket has
        finished, since something has to keep owning the running work.

    Raises
    ------
    ValueError
        If ``executor_class`` and ``kwargs`` cannot be pickled.
    """

    executor_class: type
    kwargs: Dict[str, Any] = field(default_factory=dict)
    detach: Optional[bool] = None

    def __post_init__(self) -> None:
        try:
            pickle.dumps((self.executor_class, self.kwargs, self.detach))
        except Exception as error:
            raise ValueError(
                f"ExecutorSpec must be sendable to another process, but {self.executor_class} with "
                f"kwargs {self.kwargs} cannot be pickled: {error}"
            ) from error

    @property
    def detaches(self) -> bool:
        if self.detach is not None:
            return self.detach
        return self.executor_class.__name__ in _DETACHABLE_EXECUTOR_NAMES

    def create(self) -> Any:
        """Build a fresh executor from the recipe."""
        return self.executor_class(**self.kwargs)
