"""Opt-in pytest checkpoint selection for the existing accuracy tests.

Example (run from the repository root)::

    python -m pytest -p candycrunch.checkpoint_plugin --checkpoint training/models/model.pt \
        tests/test_CandyCrunch.py -k candycrunch_accuracy

pytest-xdist workers each load the selected checkpoint. Tests specifically
comparing the shipped PyTorch and ONNX models should run without this plugin's
checkpoint option. --accuracy-csv exports the current accuracy results and
leaves the shared regression log on disk alone.
The controller saves current-run F1, precision and recall with checkpoint
metadata through tests/leaderboard.py, to tests/model_leaderboard.json and
tests/model_leaderboard.txt. --leaderboard-dir selects another output folder.
Batch accuracy and different dataset selections have separate entries;
quantification scores are excluded.
Overall averages weight each evaluated file equally, matching the terminal
Weighted Summary; per-dataset averages and file counts are saved in JSON.
The text leaderboard pairs raw and XLSX scores for matching configurations.
Avg-raw F1 and Avg-xlsx F1 include failed/evaluated datasets per format.
A dataset counts once per format even if several files or test cases cover it.
Skipped cases without scores are excluded; historical outcomes may be unknown.
"""

import csv
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("candycrunch checkpoint")
    group.addoption("--checkpoint", help="Training checkpoint to use as the inference default")
    group.addoption("--checkpoint-device", default=None, help="PyTorch device (default: CUDA when available, otherwise CPU)")
    group.addoption("--checkpoint-temperature", type=float, default=1.15, help="Inference calibration temperature (default: 1.15)")
    group.addoption("--leaderboard-dir", default=None, help="Leaderboard output folder (default: repository tests folder)")
    group.addoption("--accuracy-csv", default=None, help="Export current checkpoint accuracy rows without updating the regression log")


def pytest_configure(config):
    model_path = config.getoption("--checkpoint")
    if not model_path:
        return
    if getattr(getattr(config, "option", None), "numprocesses", 0) and not hasattr(config, "workerinput"):
        return
    from contextlib import ExitStack
    from candycrunch.checkpoint_context import use_checkpoint

    cleanup = ExitStack()
    config.add_cleanup(cleanup.close)
    try:
        cleanup.enter_context(use_checkpoint(
            model_path,
            device=config.getoption("--checkpoint-device"),
            temperature=config.getoption("--checkpoint-temperature"),
        ))
    except (OSError, ValueError, RuntimeError) as error:
        raise pytest.UsageError(f"Cannot select checkpoint {model_path!r}: {error}") from error


def pytest_report_header(config):
    model_path = config.getoption("--checkpoint")
    if model_path:
        return f"CandyCrunch checkpoint: {model_path}"


class _TestStatusCollector:
    def __init__(self):
        self.dataset_outcomes = {}

    def pytest_runtest_logreport(self, report):
        if report.skipped or (report.when != "call" and not report.failed):
            return
        for name, results in getattr(report, "user_properties", []):
            if name != "candycrunch_results":
                continue
            for dataset, params in results:
                key = (dataset, tuple(params))
                self.dataset_outcomes[key] = self.dataset_outcomes.get(key, False) or report.failed


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    collector = item.funcargs.get("result_collector")
    if not item.config.getoption("--checkpoint") or collector is None:
        yield
        return
    previous_counts = {
        (dataset, params): len(scores)
        for dataset, parameter_results in collector.dict_full_results.items()
        for params, scores in parameter_results.items()
    }
    yield
    results = [
        (dataset, list(params))
        for dataset, parameter_results in collector.dict_full_results.items()
        for params, scores in parameter_results.items()
        if len(scores) > previous_counts.get((dataset, params), 0)
    ]
    item.user_properties.append(("candycrunch_results", results))


def pytest_sessionstart(session):
    config = session.config
    if not config.getoption("--checkpoint") or hasattr(config, "workerinput"):
        return
    tracker = _TestStatusCollector()
    config._candycrunch_test_status = tracker
    config.pluginmanager.register(tracker, name="candycrunch_test_status")


def _leaderboard_collector(collector, model_path, temperature, dataset_outcomes=None):
    if not collector.param_names:
        return None
    original_names = list(collector.param_names)[1:]
    param_names = {
        **collector.param_names,
        "model": "model",
        "checkpoint_temperature": "checkpoint_temperature",
        "evaluation": "evaluation",
    }
    dict_results, dict_full_results, dataset_sets, statuses = {}, {}, {}, {}
    for dataset, parameter_results in collector.dict_full_results.items():
        for params, scores in parameter_results.items():
            if not scores:
                continue
            parameters = dict(zip(original_names, params))
            if parameters.get("format") == "quant":
                continue
            parameters.update(
                model=str(model_path),
                checkpoint_temperature=temperature,
                evaluation="batch" if dataset.endswith("_batch") else "accuracy",
            )
            enriched_params = tuple(parameters[name] for name in list(param_names)[1:])
            dict_results.setdefault(dataset, {})[enriched_params] = [score[0] for score in scores]
            dict_full_results.setdefault(dataset, {})[enriched_params] = list(scores)
            dataset_sets.setdefault(enriched_params, set()).add(dataset)
            if dataset_outcomes is not None and (dataset, params) in dataset_outcomes:
                statuses.setdefault(dataset, {})[enriched_params] = dataset_outcomes[dataset, params]
    if not dict_results:
        return None
    param_names["test_datasets"] = "test_datasets"

    def with_dataset_selection(results):
        return {
            dataset: {params + (", ".join(sorted(dataset_sets[params])),): scores for params, scores in parameter_results.items()}
            for dataset, parameter_results in results.items()
        }

    dict_results = with_dataset_selection(dict_results)
    dict_full_results = with_dataset_selection(dict_full_results)
    return SimpleNamespace(param_names=param_names, dict_results=dict_results, dict_full_results=dict_full_results,
                           dataset_outcomes=with_dataset_selection(statuses))


def _write_accuracy_csv(collector, path):
    parameter_names = list(collector.param_names)
    metrics = {"F1": 0, "Prec": 1, "Rec": 2, "NotPicked": 3, "Wrong": 4, "FP": 6}
    rows = []
    for dataset, parameter_results in sorted(collector.dict_full_results.items()):
        for params, scores in sorted(parameter_results.items()):
            parameters = dict(zip(parameter_names, (dataset, *params)))
            if parameters["evaluation"] != "accuracy" or not scores:
                continue
            rows.append({
                **{name: parameters[name] for name in ("test_dict", "supplement", "experimental", "format")},
                **{name: fmean(score[index] for score in scores) for name, index in metrics.items()},
            })
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _load_leaderboard_module():
    import importlib.util
    import sys

    module_name = "candycrunch._test_leaderboard"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = Path(__file__).resolve().parents[1] / "tests" / "leaderboard.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    config = session.config
    if hasattr(config, "workerinput") or getattr(config.option, "collectonly", False):
        return
    model_path = config.getoption("--checkpoint")
    collector = getattr(config, "collector", None)
    if not model_path or collector is None:
        return
    invocation_dir = Path(config.invocation_params.dir)
    model_path = (invocation_dir / Path(model_path).expanduser()).resolve()
    tracker = getattr(config, "_candycrunch_test_status", None)
    results = _leaderboard_collector(
        collector, model_path, config.getoption("--checkpoint-temperature"),
        dataset_outcomes=tracker.dataset_outcomes if tracker is not None else None,
    )
    if results is None:
        return
    output_dir = config.getoption("--leaderboard-dir")
    output_dir = ((invocation_dir / Path(output_dir).expanduser()).resolve() if output_dir
                  else Path(__file__).resolve().parents[1] / "tests")
    history_path = output_dir / "model_leaderboard.json"
    text_path = output_dir / "model_leaderboard.txt"
    csv_output = config.getoption("--accuracy-csv")
    reporter = config.pluginmanager.getplugin("terminalreporter")
    try:
        leaderboard = _load_leaderboard_module()
        output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.update_leaderboard_from_collector(results, history_path=history_path, text_path=text_path)
    except (OSError, ValueError, ImportError) as error:
        import warnings

        warnings.warn(pytest.PytestWarning(f"Could not save the checkpoint leaderboard: {error}"), stacklevel=2)
        if csv_output and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    else:
        if reporter is not None:
            reporter.write_sep("=", "Checkpoint leaderboard")
            reporter.write_line(f"JSON: {history_path}")
            reporter.write_line(f"Text: {text_path}")
    if csv_output:
        _write_accuracy_csv(results, invocation_dir / Path(csv_output).expanduser())
