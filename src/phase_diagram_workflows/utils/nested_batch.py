from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _run_batch_with_inner_executor(
    items: Sequence[Any],
    task_fn: Callable[..., Any],
    task_args: Tuple[Any, ...],
    inner_resource_dict: Dict[str, Any],
    inner_max_workers: int,
) -> List[Any]:
    """Run one `task_fn` call per item via an inner executor.

    This is the "job nesting" worker: it is itself submitted as a single job
    to an outer executor (e.g. `SlurmClusterExecutor`), and once running
    inside that job's allocation it opens its own inner `SlurmJobExecutor` to
    fan the allocation's cores out across `items`, instead of queuing one
    outer SLURM job per item. Defined at module level (not as a closure) so
    it can be pickled and sent to the outer executor.

    The inner executor's `max_workers` is `inner_max_workers *
    threads_per_core` (threads_per_core taken from `inner_resource_dict`,
    default 1): each of the `inner_max_workers` concurrently-running items
    claims `threads_per_core` of the allocation's CPUs, so the worker pool
    size has to scale with it to actually use the whole allocation, not just
    `inner_max_workers` of its CPUs. Same accounting as the outer/inner pair
    in `executorlib_dataframes.ipynb`'s "With job nesting" section.
    """
    from executorlib import SlurmJobExecutor

    threads_per_core = inner_resource_dict.get("threads_per_core", 1)
    with SlurmJobExecutor(
        max_workers=inner_max_workers * threads_per_core,
        resource_dict=inner_resource_dict,
    ) as inner_exe:
        futures = [inner_exe.submit(task_fn, item, *task_args) for item in items]
        return [f.result() for f in futures]


def run_nested_batch(
    items: Sequence[Any],
    task_fn: Callable[..., Any],
    outer_executor_cls: type,
    task_args: Tuple[Any, ...] = (),
    outer_resource_dict: Optional[Dict[str, Any]] = None,
    inner_resource_dict: Optional[Dict[str, Any]] = None,
    inner_max_workers: int = 1,
    cache_directory: Optional[str] = None,
    pysqa_config_directory: Optional[str] = None,
    wait: bool = True,
    **outer_executor_kwargs: Any,
) -> Any:
    """Run `task_fn` once per item, nesting executors so one outer SLURM
    allocation runs many inner tasks.

    Mirrors the "job nesting" pattern: rather than queuing one outer SLURM
    job per item, a single outer job is submitted (requesting however many
    cores `outer_resource_dict` specifies); once that allocation starts,
    `_run_batch_with_inner_executor` opens an inner `SlurmJobExecutor` using
    those cores to run `task_fn(item, *task_args)` once per item. Generic
    over `items` and `task_fn` -- this module has no knowledge of what it's
    computing, so it can be reused for any batchable calculation, not just
    energies.

    Parameters
    ----------
    items : Sequence[Any]
        The items to process, one `task_fn` call each (e.g. a list of ASE
        Atoms objects).
    task_fn : Callable[..., Any]
        Called as `task_fn(item, *task_args)` inside the inner executor.
    outer_executor_cls : type
        executorlib executor class for the outer allocation, e.g.
        `executorlib.SlurmClusterExecutor`.
    task_args : Tuple[Any, ...]
        Extra positional arguments passed to every `task_fn` call after
        `item` (e.g. a shared potential dataframe).
    outer_resource_dict : Optional[Dict[str, Any]]
        resource_dict for the outer executor (queue, cores, run_time_max,
        ...). Its `threads_per_core` -- the total CPUs the allocation
        reserves -- is set here automatically to `inner_max_workers *
        inner_resource_dict["threads_per_core"]`, overwriting whatever is
        passed in: it is derived from `inner_max_workers`/
        `inner_resource_dict` below, not independent configuration, so
        there is nothing to keep in sync by hand. Leave it out of
        `outer_resource_dict` entirely.
    inner_resource_dict : Optional[Dict[str, Any]]
        resource_dict for the inner `SlurmJobExecutor`, describing the
        resources *each item* claims from the outer allocation -- typically
        `{"cores": 1}` (one CPU per item) or `{"cores": 1, "threads_per_core":
        T}` if `task_fn` itself is multithreaded. Defaults to `{"cores": 1}`.
    inner_max_workers : int
        How many items run concurrently within the outer allocation. The
        inner `SlurmJobExecutor` is actually opened with `max_workers =
        inner_max_workers * inner_resource_dict["threads_per_core"]` (see
        `_run_batch_with_inner_executor`), since each concurrent item claims
        `threads_per_core` CPUs of the allocation -- and the outer
        allocation is sized to match automatically (see
        `outer_resource_dict` above).
    cache_directory : Optional[str]
        Passed to the outer executor for executorlib's on-disk result cache.
    pysqa_config_directory : Optional[str]
        Passed to the outer executor for SLURM queue configuration.
    wait : bool
        Forwarded to `outer_executor_cls` as its own `wait` constructor
        argument. If True (default), the outer executor blocks on exit until
        the batch finishes, and this function returns the list of results.
        If False, the outer executor does not block on exit and this
        function returns immediately with the raw `Future` -- results are
        retrieved later via `future.result()` or from `cache_directory` with
        `executorlib.get_cache_data`.
    **outer_executor_kwargs
        Extra keyword arguments forwarded to `outer_executor_cls`.

    Returns
    -------
    list or concurrent.futures.Future
        One `task_fn` result per item, in the same order as `items`, if
        `wait=True`; otherwise the `Future` wrapping the pending batch.
    """
    inner_resource_dict = inner_resource_dict or {"cores": 1}
    inner_threads_per_core = inner_resource_dict.get("threads_per_core", 1)

    # Derived, not user-supplied (see outer_resource_dict above): the outer
    # allocation must reserve exactly inner_max_workers *
    # inner_threads_per_core CPUs for the inner executor to actually use.
    outer_resource_dict = dict(outer_resource_dict or {})
    outer_resource_dict["threads_per_core"] = inner_max_workers * inner_threads_per_core

    executor_kwargs: Dict[str, Any] = dict(outer_executor_kwargs)
    executor_kwargs["wait"] = wait
    executor_kwargs["resource_dict"] = outer_resource_dict
    if cache_directory is not None:
        executor_kwargs["cache_directory"] = cache_directory
    if pysqa_config_directory is not None:
        executor_kwargs["pysqa_config_directory"] = pysqa_config_directory

    with outer_executor_cls(**executor_kwargs) as exe:
        future = exe.submit(
            _run_batch_with_inner_executor,
            list(items),
            task_fn,
            task_args,
            inner_resource_dict,
            inner_max_workers,
        )
        if not wait:
            return future

    return future.result()
