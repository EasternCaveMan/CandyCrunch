"""Select a training checkpoint without editing the prediction pipeline.

Use ``with use_checkpoint(path):`` before calling the existing wrap_inference
or wrap_inference_batch functions. Their identities and signatures are retained,
so previously imported references also use the selected defaults. Explicit
arguments still override defaults. All changes are restored on context exit.

Selection changes process-wide state; use separate processes for concurrent
checkpoints. prediction.py must still be importable with its normal defaults.
"""

from contextlib import ExitStack, contextmanager
import importlib
import inspect


@contextmanager
def use_checkpoint(model_path, device=None, temperature=1.15):
    """Temporarily select a checkpoint for unchanged inference calls.

    The checkpoint supplies the model, vocabulary, number of spectrum bins and
    composition order. Calibration temperature is independent of the training
    temperature. Batch inference with additional subprocesses requires fork;
    spawned subprocesses do not inherit this process's selected defaults.
    """
    import torch
    from candycrunch.checkpoint import load_checkpoint

    selected_device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = load_checkpoint(model_path, device=selected_device, temperature=temperature)
    prediction = importlib.import_module("candycrunch.prediction")
    defaults = {
        "model": loaded["model"],
        "glycans": loaded["glycans"],
        "bin_num": loaded["bin_num"],
        "temperature": torch.tensor([loaded["temperature"]], device=selected_device),
    }
    module_values = {
        "candycrunch": defaults["model"],
        "glycans": defaults["glycans"],
        "bin_num": defaults["bin_num"],
        "temperature": defaults["temperature"],
        "comp_vector_order": loaded["comp_vector_order"],
        "device": selected_device,
    }
    with ExitStack() as cleanup:
        for name, value in module_values.items():
            cleanup.callback(setattr, prediction, name, getattr(prediction, name))
            setattr(prediction, name, value)
        for name in ("wrap_inference", "wrap_inference_batch", "get_topk", "augment_predictions", "condense_dataframe"):
            function = getattr(prediction, name)
            parameters = inspect.signature(function).parameters.values()
            positional_defaults = tuple(
                defaults.get(parameter.name, parameter.default)
                for parameter in parameters
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and parameter.default is not inspect.Parameter.empty
            )
            cleanup.callback(setattr, function, "__defaults__", function.__defaults__)
            function.__defaults__ = positional_defaults
        yield loaded
