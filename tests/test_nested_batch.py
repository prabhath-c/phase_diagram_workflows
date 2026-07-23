"""
Unit tests for phase_diagram_workflows.utils.nested_batch.

executorlib is mocked throughout -- these tests never submit a real job,
open a real executor, or need a SLURM queue.
"""

from unittest.mock import MagicMock, patch

from phase_diagram_workflows.utils.nested_batch import (
    _run_batch_with_inner_executor,
    run_nested_batch,
)


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _FakeOuterExecutor:
    """Captures constructor kwargs and the submitted call; never runs anything for real."""

    captured_kwargs = None
    captured_submit_args = None
    submit_result = None

    def __init__(self, **kwargs):
        type(self).captured_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        type(self).captured_submit_args = (fn, args)
        return _FakeFuture(type(self).submit_result)


class TestRunBatchWithInnerExecutor:
    @patch("executorlib.SlurmJobExecutor")
    def test_runs_task_fn_once_per_item(self, mock_slurm_job_executor):
        mock_inner = MagicMock()
        mock_slurm_job_executor.return_value.__enter__.return_value = mock_inner
        mock_inner.submit.side_effect = lambda fn, item, *a: _FakeFuture(fn(item, *a))

        result = _run_batch_with_inner_executor(
            items=[1, 2, 3],
            task_fn=lambda x: x * 2,
            task_args=(),
            inner_resource_dict={"cores": 1, "threads_per_core": 4},
            inner_max_workers=5,
        )

        assert result == [2, 4, 6]

    @patch("executorlib.SlurmJobExecutor")
    def test_max_workers_scaled_by_threads_per_core(self, mock_slurm_job_executor):
        mock_inner = MagicMock()
        mock_slurm_job_executor.return_value.__enter__.return_value = mock_inner
        mock_inner.submit.side_effect = lambda fn, item, *a: _FakeFuture(item)

        _run_batch_with_inner_executor(
            items=[1],
            task_fn=lambda x: x,
            task_args=(),
            inner_resource_dict={"cores": 1, "threads_per_core": 4},
            inner_max_workers=5,
        )

        assert mock_slurm_job_executor.call_args.kwargs["max_workers"] == 20  # 5 * 4

    @patch("executorlib.SlurmJobExecutor")
    def test_defaults_threads_per_core_to_one(self, mock_slurm_job_executor):
        mock_inner = MagicMock()
        mock_slurm_job_executor.return_value.__enter__.return_value = mock_inner
        mock_inner.submit.side_effect = lambda fn, item, *a: _FakeFuture(item)

        _run_batch_with_inner_executor(
            items=[1],
            task_fn=lambda x: x,
            task_args=(),
            inner_resource_dict={"cores": 1},
            inner_max_workers=7,
        )

        assert mock_slurm_job_executor.call_args.kwargs["max_workers"] == 7

    @patch("executorlib.SlurmJobExecutor")
    def test_forwards_task_args_after_item(self, mock_slurm_job_executor):
        mock_inner = MagicMock()
        mock_slurm_job_executor.return_value.__enter__.return_value = mock_inner
        mock_inner.submit.side_effect = lambda fn, item, *a: _FakeFuture(fn(item, *a))

        result = _run_batch_with_inner_executor(
            items=["Al"],
            task_fn=lambda symbol, potential_df: (symbol, potential_df),
            task_args=("a-potential-df",),
            inner_resource_dict={"cores": 1},
            inner_max_workers=1,
        )

        assert result == [("Al", "a-potential-df")]


class TestRunNestedBatch:
    def setup_method(self):
        _FakeOuterExecutor.captured_kwargs = None
        _FakeOuterExecutor.captured_submit_args = None
        _FakeOuterExecutor.submit_result = None

    def test_derives_outer_threads_per_core_from_inner_settings(self):
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=[1, 2, 3],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            outer_resource_dict={"queue": "cmmg", "cores": 1, "run_time_max": 1200},
            inner_resource_dict={"cores": 1, "threads_per_core": 2},
            inner_max_workers=64,
        )
        assert _FakeOuterExecutor.captured_kwargs["resource_dict"]["threads_per_core"] == 128

    def test_overwrites_a_manually_supplied_threads_per_core(self):
        # threads_per_core is derived, not independent configuration -- a
        # caller-supplied value must not silently win and desync the two.
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=[1],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            outer_resource_dict={"cores": 1, "threads_per_core": 999},
            inner_resource_dict={"cores": 1, "threads_per_core": 2},
            inner_max_workers=4,
        )
        assert _FakeOuterExecutor.captured_kwargs["resource_dict"]["threads_per_core"] == 8

    def test_default_inner_resource_dict_is_one_core(self):
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=[1],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            inner_max_workers=10,
        )
        assert _FakeOuterExecutor.captured_kwargs["resource_dict"]["threads_per_core"] == 10

    def test_preserves_other_outer_resource_dict_keys(self):
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=[1],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            outer_resource_dict={"queue": "cmmg", "cores": 1, "run_time_max": 1200},
            inner_max_workers=1,
        )
        rd = _FakeOuterExecutor.captured_kwargs["resource_dict"]
        assert rd["queue"] == "cmmg"
        assert rd["run_time_max"] == 1200

    def test_wait_true_returns_result_list(self):
        _FakeOuterExecutor.submit_result = ["a", "b"]
        result = run_nested_batch(
            items=[1, 2],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            wait=True,
        )
        assert result == ["a", "b"]

    def test_wait_false_returns_future_without_resolving(self):
        result = run_nested_batch(
            items=[1],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            wait=False,
        )
        assert isinstance(result, _FakeFuture)
        assert _FakeOuterExecutor.captured_kwargs["wait"] is False

    def test_submits_run_batch_with_inner_executor_and_materialized_items(self):
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=(a for a in ["a", "b"]),  # a generator, not a list
            task_fn=str.upper,
            outer_executor_cls=_FakeOuterExecutor,
        )
        fn, args = _FakeOuterExecutor.captured_submit_args
        assert fn is _run_batch_with_inner_executor
        assert args[0] == ["a", "b"]
        assert args[1] is str.upper

    def test_forwards_cache_directory_and_pysqa_config_directory(self):
        _FakeOuterExecutor.submit_result = ["ok"]
        run_nested_batch(
            items=[1],
            task_fn=lambda x: x,
            outer_executor_cls=_FakeOuterExecutor,
            cache_directory="my_cache",
            pysqa_config_directory="my_pysqa_config",
        )
        assert _FakeOuterExecutor.captured_kwargs["cache_directory"] == "my_cache"
        assert _FakeOuterExecutor.captured_kwargs["pysqa_config_directory"] == "my_pysqa_config"
