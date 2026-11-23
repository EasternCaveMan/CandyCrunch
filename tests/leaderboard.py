from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
from tabulate import tabulate


LEADERBOARD_DIR = Path(__file__).resolve().parent
LEADERBOARD_HISTORY_PATH = LEADERBOARD_DIR / "model_leaderboard.json"
LEADERBOARD_TEXT_PATH = LEADERBOARD_DIR / "model_leaderboard.txt"

HIDDEN_PARAMETER_COLUMNS = {
    "experimental",
    "supplement",
    "model_checkpoint",
}
# Fine-tuning with a contrastive term (--combine-loss) was removed; histories recorded before drop these on load
REMOVED_PARAMETERS = {
    "combine_loss",
    "contrastive_loss_weight",
    "contrastive_loss_temperature",
    "contrastive_learning",
    "contrastive_weight",
    "contrastive_temperature",
}
REMOVED_RUN_ID_PATTERN = re.compile(r" \| (?:" + "|".join(sorted(REMOVED_PARAMETERS)) + r")=[^|]*?(?= \| |$)")
TABLE_HIDDEN_PARAMETER_COLUMNS = {
    "format",
    "pretraining_epochs",
    "supcon_temperature",
    "pretraining_optimizer",
    "test_datasets",
    "evaluation",
    "checkpoint_temperature",
    "classifier_num_experts",
    "classifier_top_k",
    "classifier_expert_hidden_dim",
    "classifier_moe_aux_loss_weight",
    "classifier_shared_expert",
    "classifier_fixed_routing",
    "classifier_routing_key",
    "nheads",
    "nlayers",
    "peak_hidden_dim",
    "ff_dim",
    "peak_encoder",
    "max_peaks",
    "use_resunits",
    "dropout",
    "se",
    "se_reduction",
    "in_channel",
    "dilations",
}
DISPLAY_PARAMETER_ORDER = [
    "model",
    "num_res_units",
    "loss_function",
    "candidate_sets",
    "pretraining",
    "classifier_moe",
    "norm_type",
    "activation",
    "encoder_activation",
    "transformer_parameters",
    "use_transformer_ff",
    "encoder_type",
    "num_experts",
    "moe_top_k",
    "hidden_dim",
    "split",
    "dataset",
    "resample_dataset",
    "num_classes",
    "input_precursor_dim",
    "input_dim",
]
DISPLAY_COLUMN_NAMES = {
    "loss_function": "LossFun",
    "candidate_sets": "CandSets",
    "pretraining": "Pretrain",
    "encoder_type": "encTyp",
    "num_experts": "moe_exp",
    "moe_top_k": "moe_top",
    "classifier_moe": "MoC",
    "hidden_dim": "HidDim",
    "num_res_units": "RU",
    "resample_dataset": "Resample",
    "transformer_parameters": "Trans_param",
    "input_dim": "Bins",
    "num_classes": "Classes",
    "input_precursor_dim": "CompDim",
    "norm_type": "NorTyp",
    "activation": "Act",
    "encoder_activation": "EncAct",
    "use_transformer_ff": "UseTransFF",
}
# Two-letter CNN2 dilation schedules of run names and the RU column (training_script.py's DILATION_CODES; Cu is a custom list)
DILATION_CODE_SCHEDULES = {"Pw": "pow2", "Na": "native", "Pe": "permethylated", "Re": "residue", "Cu": "custom"}
TRANSFORMER_PREFIX = "CandyCrunch_Transformer_"
CNN_PREFIX = "CandyCrunch_CNN_"
CNN2_PREFIX = "CandyCrunch_CNN2_"
LOSS_TAGS = {
    "XYZLOSS": "xyz_loss",
    "CELOSS": "cross_entropy",
    "FOLLOSS": "focal_loss",
    "FOCALLOSS": "focal_loss",
    "POLOSS": "PolyCrEnr",
    "CMLOSS": "custom_loss",
}
PRETRAINING_TAG_PATTERN = re.compile(
    r"_(?P<tag>SupConPtE|UnsupConPtE|PretrainE)(?P<pretraining_epochs>\d+)T(?P<supcon_temperature>[0-9.eE+-]+)"
    r"(?P<adamw>_PtAdamW)?(?=_)"
)
# Retain mode and optimizer in the history and identity, even though the table
# combines pretraining settings into one compact cell. Missing optimizer means SAM.
PRETRAINING_TAG_MODES = {"SupConPtE": "supervised", "UnsupConPtE": "unsupervised", "PretrainE": "True"}


@dataclass
class LeaderboardEntry:
    run_id: str
    timestamp: str
    average_f1: float
    average_precision: float
    average_recall: float
    dataset_count: int
    parameters: dict[str, str]
    dataset_scores: dict[str, float]
    dataset_precisions: dict[str, float]
    dataset_recalls: dict[str, float]
    dataset_file_counts: dict[str, int] = field(default_factory=dict)
    dataset_outcomes: dict[str, bool] = field(default_factory=dict)
    failed_tests: int | None = None
    total_tests: int | None = None


def _stringify_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _descending_metric(value: Any) -> float:
    numeric = float(value)
    return -numeric if np.isfinite(numeric) else float("inf")


def _ascending_metric(value: Any) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("-inf")


def _has_value(value: Any) -> bool:
    return value not in (None, "")


def _display_model_value(model_value: Any) -> str:
    model_text = _stringify_value(model_value).strip()
    model_path = Path(model_text)
    if model_path.suffix in {".pt", ".pth"}:
        return model_path.stem
    return model_path.name


def _resolve_model_checkpoint(model_value: Any) -> str | None:
    model_text = _stringify_value(model_value).strip()
    if not model_text:
        return None

    model_path = Path(model_text)
    if model_path.suffix in {".pt", ".pth"}:
        return str(model_path)

    try:
        from candycrunch.prediction import _resolve_model_path
    except Exception:
        return None

    try:
        resolved_path = _resolve_model_path(model_text)
    except Exception:
        return None

    return str(resolved_path)


def _model_stem(model_identifier: Any) -> str:
    model_text = _stringify_value(model_identifier).strip()
    if not model_text:
        return ""
    model_path = Path(model_text)
    if model_path.suffix in {".pt", ".pth"}:
        return model_path.stem
    return model_path.name


def _parse_parenthesized_token(token: str, prefix: str) -> str | None:
    if token.startswith(f"{prefix}(") and token.endswith(")"):
        return token[len(prefix) + 1:-1]
    return None


def _split_model_tokens(value: str) -> list[str]:
    tokens = []
    start = 0
    depth = 0

    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "_" and depth == 0:
            tokens.append(value[start:index])
            start = index + 1

    tokens.append(value[start:])
    return tokens


def _parse_encoder_tag(tag: str) -> dict[str, str]:
    if tag == "NOFF":
        return {"use_transformer_ff": "False"}
    if tag == "DENSE":
        return {"encoder_type": "dense", "use_transformer_ff": "True"}

    match = re.fullmatch(r"(?:T)?MoE(?P<num_experts>\d+)K(?P<moe_top_k>\d+)", tag)
    if match:
        return {
            "encoder_type": "moe",
            "num_experts": match.group("num_experts"),
            "moe_top_k": match.group("moe_top_k"),
            "use_transformer_ff": "True",
        }
    return {}


def _parse_classifier_tag(tag: str) -> dict[str, str]:
    if tag in {"CSHARED", "ShCl"}:
        return {"classifier_moe": "False"}

    # Fix (fixed routing by glycan class) or FixInst (by instrument) takes the balancing weight's place, and S (a shared expert) ends the tag:
    # MoC3K1FixS, MoC29K1FixInstS, MoC10K4W0.1S
    match = re.fullmatch(
        r"(?:CMOE|MoC)'?(?P<classifier_num_experts>\d+)K(?P<classifier_top_k>\d+)"
        r"(?:(?P<classifier_fixed_routing>Fix(?P<instrument>Inst)?)|W(?P<classifier_moe_aux_loss_weight>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))?"
        r"(?:H(?P<classifier_expert_hidden_dim>\d+))?"
        r"(?P<classifier_shared_expert>S)?",
        tag,
    )
    if match:
        parsed = {
            "classifier_moe": "True",
            "classifier_num_experts": match.group("classifier_num_experts"),
            "classifier_top_k": match.group("classifier_top_k"),
        }
        for name in ("classifier_fixed_routing", "classifier_shared_expert"):
            if match.group(name):
                parsed[name] = "True"
        if match.group("instrument"):
            parsed["classifier_routing_key"] = "instrument"
        hidden_dim = match.group("classifier_expert_hidden_dim")
        if hidden_dim is not None:
            parsed["classifier_expert_hidden_dim"] = hidden_dim
        weight = match.group("classifier_moe_aux_loss_weight")
        if weight is not None:
            parsed["classifier_moe_aux_loss_weight"] = str(float(weight))
        return parsed
    return {}


def _parse_transformer_stem(stem: str) -> dict[str, str]:
    parsed = {"model": "Transformer"}
    if not stem.startswith(TRANSFORMER_PREFIX):
        return parsed

    tokens = _split_model_tokens(stem[len(TRANSFORMER_PREFIX):])
    if len(tokens) < 4:
        return parsed

    encoder_tag, classifier_tag, split = tokens[:3]
    parsed["split"] = split
    parsed.update(_parse_encoder_tag(encoder_tag))
    parsed.update(_parse_classifier_tag(classifier_tag))
    parsed["dataset"] = tokens[-1]

    for token in tokens[3:-1]:
        if token in LOSS_TAGS:
            parsed["loss_function"] = LOSS_TAGS[token]
            continue
        width_match = re.fullmatch(
            r"H(?P<nheads>\d+)L(?P<nlayers>\d+)(?:PHD(?P<peak_hidden_dim>\d+))?(?:FFD(?P<ff_dim>\d+))?",
            token,
        )
        if width_match:
            parsed["nheads"] = width_match.group("nheads")
            parsed["nlayers"] = width_match.group("nlayers")
            peak_hidden_dim = width_match.group("peak_hidden_dim")
            if peak_hidden_dim is not None:
                parsed["peak_hidden_dim"] = peak_hidden_dim
            ff_dim = width_match.group("ff_dim")
            if ff_dim is not None:
                parsed["ff_dim"] = ff_dim
            continue

        peak_hidden_match = re.fullmatch(r"PHD(?P<peak_hidden_dim>\d+)", token)
        if peak_hidden_match:
            parsed["peak_hidden_dim"] = peak_hidden_match.group("peak_hidden_dim")
            continue

        ff_dim_match = re.fullmatch(r"FFD(?P<ff_dim>\d+)", token)
        if ff_dim_match:
            parsed["ff_dim"] = ff_dim_match.group("ff_dim")
            continue

        peak_encoder = _parse_parenthesized_token(token, "PE")
        if peak_encoder is not None:
            parsed["peak_encoder"] = peak_encoder
            continue

        if token in {"CS", "NoCS"}:
            parsed["candidate_sets"] = str(token == "CS")
            continue
        if token in {"avg", "explode"}:
            parsed["resample_dataset"] = token
            continue
        encoder_activation = _parse_parenthesized_token(token, "EACT")
        if encoder_activation is not None:
            parsed["encoder_activation"] = encoder_activation
            continue

        norm_type = _parse_parenthesized_token(token, "N")
        if norm_type is not None:
            parsed["norm_type"] = norm_type
            continue

        activation = _parse_parenthesized_token(token, "ACT")
        if activation is not None:
            parsed["activation"] = activation
            continue

        use_transformer_ff = _parse_parenthesized_token(token, "FF")
        if use_transformer_ff is not None:
            parsed["use_transformer_ff"] = use_transformer_ff
            continue

    return parsed


def _parse_cnn_stem(stem: str) -> dict[str, str]:
    model = "CNN2" if stem.startswith(CNN2_PREFIX) else "CNN"
    prefix = CNN2_PREFIX if model == "CNN2" else CNN_PREFIX
    parsed = {"model": model}
    if not stem.startswith(prefix):
        return parsed

    tokens = _split_model_tokens(stem[len(prefix):])
    if len(tokens) < 3:
        return parsed
    parsed.update(_parse_classifier_tag(tokens[0]))
    parsed["split"] = tokens[1]
    resample_index = next(
        (index for index, token in enumerate(tokens[2:], start=2) if token in {"avg", "explode"}),
        None,
    )
    if resample_index is not None:
        parsed["resample_dataset"] = tokens[resample_index]
        parsed["dataset"] = "_".join(tokens[resample_index + 1:])
        settings = tokens[2:resample_index]
    else:
        parsed["dataset"] = tokens[-1]
        settings = tokens[2:-1]

    for token in settings:
        if token in LOSS_TAGS:
            parsed["loss_function"] = LOSS_TAGS[token]
        elif token in {"CS", "NoCS"}:
            parsed["candidate_sets"] = str(token == "CS")
        elif model == "CNN2":
            hidden_dim = re.fullmatch(r"HD(?P<hidden_dim>\d+)", token)
            # RU3 (older names), RU6SE(F), RU4SE(T)RS8, RU6CH32SE(F), RU6D(Pw)CH64SE(F), RU5D(Na)CH64SE(F) (and RU5D(native)... from before the
            # two-letter codes): ResUnit2 blocks, dilation schedule, encoder channels, SE blocks and their reduction
            num_res_units = re.fullmatch(
                r"RU(?P<num_res_units>\d+)(?:D\((?P<dilations>[A-Za-z0-9-]+)\))?(?:CH(?P<in_channel>\d+))?"
                r"(?:SE\((?P<se>[TF])\)(?:RS(?P<se_reduction>\d+))?)?", token,
            )
            se = re.fullmatch(r"ES(?P<se>True|False)RS(?P<se_reduction>\d+)", token)
            if hidden_dim:
                parsed["hidden_dim"] = hidden_dim.group("hidden_dim")
            elif num_res_units:
                values = {name: value for name, value in num_res_units.groupdict().items() if value is not None}
                if "se" in values:
                    values["se"] = {"T": "True", "F": "False"}[values["se"]]
                if "dilations" in values:
                    # A custom list (Cu) is named only in the checkpoint's training_args
                    values["dilations"] = DILATION_CODE_SCHEDULES.get(values["dilations"], values["dilations"])
                parsed.update(values)
            elif se:
                parsed.update(se.groupdict())
            elif token in {"True", "False"}:
                parsed["se"] = token
            else:
                for name, tag in (("norm_type", "N"), ("activation", "ACT"), ("encoder_activation", "EACT")):
                    value = _parse_parenthesized_token(token, tag)
                    if value is not None:
                        parsed[name] = value
                        break
    return parsed


def _strip_pretraining_tag(stem: str) -> tuple[str, dict[str, str]]:
    match = PRETRAINING_TAG_PATTERN.search(stem)
    if not match:
        return stem, {}
    parsed = {
        "pretraining": PRETRAINING_TAG_MODES[match.group("tag")],
        "pretraining_epochs": match.group("pretraining_epochs"),
        "supcon_temperature": str(float(match.group("supcon_temperature"))),
    }
    if match.group("adamw"):
        parsed["pretraining_optimizer"] = "adamw"
    return stem[:match.start()] + stem[match.end():], parsed


@lru_cache(maxsize=None)
def _load_checkpoint_payload(checkpoint_path: str) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return {}

    try:
        import torch
    except Exception:
        return {}

    try:
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            payload = torch.load(checkpoint_file, map_location="cpu")
        except Exception:
            return {}
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _set_if_present(parsed: dict[str, str], key: str, *values: Any) -> None:
    for value in values:
        if _has_value(value):
            parsed[key] = _stringify_value(value)
            return


def _checkpoint_model_parameters(checkpoint_path: str) -> dict[str, str]:
    payload = _load_checkpoint_payload(checkpoint_path)
    if not payload:
        return {}

    training_args = payload.get("training_args")
    if not isinstance(training_args, dict):
        training_args = {}

    model_kwargs = payload.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    parsed: dict[str, str] = {}
    model_name = {
        "CandyCrunch_CNN": "CNN",
        "CandyCrunch_CNN2": "CNN2",
        "CandyCrunch_Transformer": "Transformer",
    }.get(payload.get("model_class")) or payload.get("model_type") or training_args.get("model")
    _set_if_present(parsed, "model", model_name)
    _set_if_present(parsed, "seed", training_args.get("current_seed"))
    _set_if_present(parsed, "loss_function", payload.get("loss_function"), training_args.get("loss_function"))
    _set_if_present(parsed, "candidate_sets", training_args.get("candidate_sets"))
    _set_if_present(parsed, "pretraining", training_args.get("pretraining"))
    if training_args.get("sup_pretraining") or payload.get("sup_pretraining"):
        parsed["pretraining"] = "supervised"
    elif training_args.get("unsup_pretraining") or payload.get("unsup_pretraining"):
        parsed["pretraining"] = "unsupervised"
    _set_if_present(parsed, "pretraining_epochs", training_args.get("pretraining_epochs"))
    _set_if_present(parsed, "supcon_temperature", training_args.get("supcon_temperature"))
    if training_args.get("pretraining_optimizer") == "adamw":
        parsed["pretraining_optimizer"] = "adamw"
    if parsed.get("pretraining") in (None, "", "False"):
        parsed.pop("pretraining", None)
        parsed.pop("pretraining_epochs", None)
        parsed.pop("supcon_temperature", None)
        parsed.pop("pretraining_optimizer", None)
    _set_if_present(parsed, "split", training_args.get("split"), payload.get("split"))
    _set_if_present(parsed, "dataset", training_args.get("dataset"), payload.get("dataset"))
    _set_if_present(parsed, "classifier_moe", training_args.get("classifier_moe"), model_kwargs.get("classifier_moe"))
    _set_if_present(
        parsed,
        "classifier_num_experts",
        training_args.get("classifier_num_experts"),
        model_kwargs.get("classifier_num_experts"),
    )
    _set_if_present(parsed, "classifier_top_k", training_args.get("classifier_top_k"), model_kwargs.get("classifier_top_k"))
    _set_if_present(
        parsed,
        "classifier_expert_hidden_dim",
        training_args.get("classifier_expert_hidden_dim"),
        model_kwargs.get("classifier_expert_hidden_dim"),
    )

    _set_if_present(
        parsed, "classifier_moe_aux_loss_weight",
        payload.get("classifier_moe_aux_loss_weight"), training_args.get("classifier_moe_aux_loss_weight"),
    )
    # Recorded only when on, so the entries of runs without them keep their identity. training_args name the fixed routing's key
    # ('instrument', 'glycan_class'; True before the key existed)
    for name in ("classifier_shared_expert", "classifier_fixed_routing"):
        if model_kwargs.get(name, training_args.get(name)) in (True, "instrument", "glycan_class"):
            parsed[name] = "True"
    if "instrument" in (model_kwargs.get("classifier_routing_key"), training_args.get("classifier_fixed_routing")):
        parsed["classifier_routing_key"] = "instrument"
    if parsed.get("model") in {"CNN2", "Transformer"}:
        for name in ("norm_type", "activation", "encoder_activation"):
            _set_if_present(parsed, name, training_args.get(name))
    if parsed.get("model") == "CNN2":
        for name in ("hidden_dim", "se", "se_reduction", "num_res_units", "in_channel", "dilations"):
            _set_if_present(parsed, name, training_args.get(name))
        # training_args name the schedule (native, permethylated, residue); without them, the checkpoint's list joined as in the run name
        if "dilations" not in parsed and isinstance(model_kwargs.get("dilations"), (list, tuple)):
            parsed["dilations"] = "-".join(str(dilation) for dilation in model_kwargs["dilations"])

    if parsed.get("model") == "Transformer":
        _set_if_present(parsed, "encoder_type", training_args.get("encoder_type"), model_kwargs.get("encoder_type"))
        _set_if_present(parsed, "num_experts", training_args.get("num_experts"), model_kwargs.get("num_experts"))
        _set_if_present(parsed, "moe_top_k", training_args.get("moe_top_k"), model_kwargs.get("moe_top_k"))
        _set_if_present(parsed, "nheads", training_args.get("nheads"), model_kwargs.get("heads"))
        _set_if_present(parsed, "nlayers", training_args.get("nlayers"), model_kwargs.get("layers"))
        _set_if_present(parsed, "peak_hidden_dim", training_args.get("peak_hidden_dim"), model_kwargs.get("peak_hidden_dim"))
        _set_if_present(parsed, "ff_dim", training_args.get("ff_dim"), model_kwargs.get("ff_dim"))
        _set_if_present(parsed, "peak_encoder", training_args.get("peak_encoder"), model_kwargs.get("peak_encoder"))
        _set_if_present(parsed, "norm_type", training_args.get("norm_type"), model_kwargs.get("norm_type"))
        _set_if_present(parsed, "activation", training_args.get("activation"), model_kwargs.get("activation"))
        _set_if_present(
            parsed,
            "encoder_activation",
            training_args.get("encoder_activation"),
            model_kwargs.get("encoder_activation"),
        )
        _set_if_present(
            parsed,
            "use_transformer_ff",
            training_args.get("use_transformer_ff"),
            model_kwargs.get("use_transformer_ff"),
        )

    # The constructor kwargs describe the saved architecture; training args may
    # contain defaults for other model types or differ from the actual kwargs.
    aliases = {"heads": "nheads", "layers": "nlayers"}
    for name, value in model_kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            _set_if_present(parsed, aliases.get(name, name), value)

    return parsed


def _merge_model_parameters(base: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in override.items():
        if _has_value(value):
            merged[key] = value
    return merged


def _finalize_model_parameters(parameters: dict[str, str]) -> dict[str, str]:
    finalized = dict(parameters)

    if finalized.get("model") == "Transformer":
        if finalized.get("use_transformer_ff") == "False":
            finalized.pop("encoder_type", None)
            finalized.pop("num_experts", None)
            finalized.pop("moe_top_k", None)
        elif finalized.get("encoder_type") != "moe":
            finalized.pop("num_experts", None)
            finalized.pop("moe_top_k", None)

    if finalized.get("classifier_moe") != "True":
        finalized.pop("classifier_num_experts", None)
        finalized.pop("classifier_top_k", None)
        finalized.pop("classifier_expert_hidden_dim", None)
        finalized.pop("classifier_moe_aux_loss_weight", None)

    # Classifier options are kept only when on (a classifier MoE with them), so runs without them keep the identity they always had
    for name in ("classifier_shared_expert", "classifier_fixed_routing"):
        if finalized.get(name) != "True" or finalized.get("classifier_moe") != "True":
            finalized.pop(name, None)
    if finalized.get("classifier_fixed_routing") == "True":
        # Fixed routing has no router, so the balancing weight a checkpoint records has no effect
        finalized.pop("classifier_moe_aux_loss_weight", None)
    # Only routing by instrument is named: glycan-class routing is fixed routing's original key, so its entries keep their identity
    if finalized.get("classifier_fixed_routing") != "True" or finalized.get("classifier_routing_key") != "instrument":
        finalized.pop("classifier_routing_key", None)
    # Only CNN2 dilation schedules other than 1, 2, 4, ... are named, so the entries of those runs keep their identity
    if finalized.get("model") != "CNN2" or finalized.get("dilations") in (None, "", "pow2"):
        finalized.pop("dilations", None)

    return finalized


def _extract_model_parameters(raw_parameters: dict[str, Any]) -> dict[str, str]:
    model_checkpoint = raw_parameters.get("model_checkpoint")
    stem = ""
    parsed_from_stem: dict[str, str] = {}
    checkpoint_parameters: dict[str, str] = {}

    if _has_value(model_checkpoint):
        checkpoint_path = _stringify_value(model_checkpoint).strip()
        checkpoint_parameters = _checkpoint_model_parameters(checkpoint_path)
        stem = _model_stem(checkpoint_path)
    else:
        model_value = raw_parameters.get("model", "")
        model_text = _stringify_value(model_value).strip()
        model_path = Path(model_text)
        if model_path.suffix in {".pt", ".pth"}:
            checkpoint_parameters = _checkpoint_model_parameters(model_text)
            stem = _model_stem(model_text)
        elif model_text.startswith((TRANSFORMER_PREFIX, CNN_PREFIX, CNN2_PREFIX)):
            stem = _model_stem(model_text)
        else:
            if not model_text:
                return {}
            return {"model": _display_model_value(model_value)}

    stem, pretraining_parameters = _strip_pretraining_tag(stem)
    if stem.startswith(TRANSFORMER_PREFIX):
        parsed_from_stem = _parse_transformer_stem(stem)
    elif stem.startswith((CNN_PREFIX, CNN2_PREFIX)):
        parsed_from_stem = _parse_cnn_stem(stem)

    parsed_from_stem.update(pretraining_parameters)

    merged_parameters = _merge_model_parameters(parsed_from_stem, checkpoint_parameters)
    if merged_parameters:
        return _finalize_model_parameters(merged_parameters)

    return {"model": _display_model_value(model_checkpoint or raw_parameters.get("model", ""))}


def _normalize_session_parameters(parameters: dict[str, str]) -> dict[str, str]:
    normalized = dict(parameters)
    resolved_checkpoint = _resolve_model_checkpoint(normalized.get("model", ""))
    if resolved_checkpoint is not None:
        normalized["model_checkpoint"] = resolved_checkpoint
    normalized.update({
        name: value for name, value in _extract_model_parameters(normalized).items()
        if name != "model"
    })
    normalized.setdefault("loss_function", "custom_loss")
    return normalized


def _display_parameters(entry: dict[str, Any]) -> dict[str, str]:
    raw_parameters = entry.get("parameters", {})
    display_parameters = {}
    model_parameters = _extract_model_parameters(raw_parameters)

    if "model" in model_parameters:
        display_parameters["model"] = model_parameters.pop("model")
    elif "model" in raw_parameters:
        display_parameters["model"] = _display_model_value(raw_parameters["model"])

    for key, value in raw_parameters.items():
        if key == "model" or key in HIDDEN_PARAMETER_COLUMNS or key in REMOVED_PARAMETERS:
            continue
        display_parameters[key] = _stringify_value(value)

    for key, value in model_parameters.items():
        if key not in HIDDEN_PARAMETER_COLUMNS and key not in REMOVED_PARAMETERS and _has_value(value):
            display_parameters[key] = _stringify_value(value)

    display_parameters.setdefault("loss_function", "custom_loss")
    # CNN2 checkpoints from before configurable depth have six ResUnit2 blocks, and those from before configurable width 64 channels.
    if display_parameters.get("model") == "CNN2":
        display_parameters.setdefault("num_res_units", "6")
        display_parameters.setdefault("in_channel", "64")
    return display_parameters


def _format_classifier_moe(parameters: dict[str, str]) -> str:
    enabled = parameters.get("classifier_moe", "")
    if enabled == "False":
        return "F"
    if enabled != "True":
        return enabled

    parts = ["T"]
    for name, prefix in (
        ("classifier_num_experts", "E"),
        ("classifier_top_k", "K"),
        ("classifier_expert_hidden_dim", "HD"),
        ("classifier_moe_aux_loss_weight", "W"),
    ):
        value = parameters.get(name)
        if _has_value(value):
            parts.append(f"{prefix}{value}")
    # Fixed routing (by glycan class, or FixInst by instrument) and a shared expert, after the settings every classifier MoE has: T_E3_K1_Fix_S,
    # T_E29_K1_FixInst_S
    if parameters.get("classifier_fixed_routing") == "True":
        parts.append("FixInst" if parameters.get("classifier_routing_key") == "instrument" else "Fix")
    if parameters.get("classifier_shared_expert") == "True":
        parts.append("S")
    return "_".join(parts)


def _format_res_units(parameters: dict[str, str]) -> str:
    if parameters.get("model") != "CNN2":
        return ""
    count = parameters.get("num_res_units")
    if not _has_value(count):
        return ""
    channels = parameters.get("in_channel")
    dilations = parameters.get("dilations")
    enabled = {"True": "T", "False": "F"}.get(parameters.get("se", ""), "?")
    # Two-letter dilation schedule: Pw (1, 2, 4, ...; also entries without a schedule), Na (native residues), Pe (permethylated), Re (both),
    # Cu (a custom list; the run name keeps the list itself)
    if not _has_value(dilations) or dilations == "pow2":
        schedule = "Pw"
    else:
        schedule = {"native": "Na", "permethylated": "Pe", "residue": "Re"}.get(dilations, "Cu")
    # RU[6]D[Pw]CH[64]SE[F]; with SE blocks RU[6]D[Pw]CH[64]SE[T]SR[8]
    formatted = f"RU[{count}]D[{schedule}]CH[{channels if _has_value(channels) else '?'}]SE[{enabled}]"
    if enabled == "T":
        reduction = parameters.get("se_reduction")
        formatted += f"SR[{reduction if _has_value(reduction) else '?'}]"
    return formatted


def _format_pretraining(parameters: dict[str, str]) -> str:
    enabled = parameters.get("pretraining", "False")
    if enabled.lower() in {"false", "f", "none", "0", ""}:
        return "F"

    parts = ["T"]
    mode = {"supervised": "S", "unsupervised": "U"}.get(enabled.lower())
    if mode:
        parts.append(mode)
    for name, prefix in (("pretraining_epochs", "Ep"), ("supcon_temperature", "T")):
        if _has_value(parameters.get(name)):
            parts.append(f"{prefix}{parameters[name]}")
    optimizer = parameters.get("pretraining_optimizer") or "sam"
    parts.append(f"Op{optimizer.capitalize()}")
    return "_".join(parts)


def _format_transformer_parameters(parameters: dict[str, str]) -> str:
    if parameters.get("model") != "Transformer":
        return ""
    return "_".join(
        f"{prefix}{parameters[name]}"
        for name, prefix in (
            ("nheads", "nH"),
            ("nlayers", "nL"),
            ("peak_hidden_dim", "PHD"),
            ("ff_dim", "FFD"),
            ("peak_encoder", "PE"),
        )
        if _has_value(parameters.get(name))
    )


def _parameter_columns(entries: list[dict[str, Any]]) -> list[str]:
    display_rows = [_display_parameters(entry) for entry in entries]
    columns = list(DISPLAY_PARAMETER_ORDER)

    for row in display_rows:
        for key in row:
            if key not in columns and key not in TABLE_HIDDEN_PARAMETER_COLUMNS:
                columns.append(key)

    return columns


def _display_column_name(column: str) -> str:
    return DISPLAY_COLUMN_NAMES.get(column, column)


def _is_legacy_alias_entry(entry: dict[str, Any]) -> bool:
    raw_parameters = entry.get("parameters", {})
    if _has_value(raw_parameters.get("model_checkpoint")):
        return False

    model_text = _stringify_value(raw_parameters.get("model", "")).strip()
    return model_text in {"CNN", "Transformer"}


def _entry_identity(entry: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    display_parameters = _display_parameters(entry)
    identity_items = sorted(
        (key, value)
        for key, value in display_parameters.items()
        if _has_value(value)
    )
    if _is_legacy_alias_entry(entry):
        identity_items.append(("legacy_run_id", entry["run_id"]))
    return tuple(identity_items)


def _entry_preference_key(entry: dict[str, Any]) -> tuple[float, float, float, str, str]:
    return (
        _ascending_metric(entry["average_f1"]),
        _ascending_metric(entry.get("average_precision", float("nan"))),
        _ascending_metric(entry.get("average_recall", float("nan"))),
        entry["timestamp"],
        entry["run_id"],
    )


def _deduplicate_best_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_identity: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for entry in entries:
        identity = _entry_identity(entry)
        current_best = best_by_identity.get(identity)
        if current_best is None or _entry_preference_key(entry) > _entry_preference_key(current_best):
            best_by_identity[identity] = entry
    return _sort_history(list(best_by_identity.values()))


def _sort_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda entry: (
            _descending_metric(entry["average_f1"]),
            _descending_metric(entry.get("average_precision", float("nan"))),
            _descending_metric(entry.get("average_recall", float("nan"))),
            entry["timestamp"],
            entry["run_id"],
        ),
    )


def _dataset_status_counts(dataset_names, dataset_outcomes):
    total_tests = len(dataset_names)
    if not all(name in dataset_outcomes for name in dataset_names):
        return None, total_tests
    return sum(dataset_outcomes[name] for name in dataset_names), total_tests


def _group_raw_file_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine raw/mzML results from the same run, retaining file weights."""
    metric_fields = (
        ("dataset_scores", "average_f1"),
        ("dataset_precisions", "average_precision"),
        ("dataset_recalls", "average_recall"),
    )
    groups = defaultdict(list)
    grouped = []
    for entry in entries:
        parameters = entry.get("parameters", {})
        counts = entry.get("dataset_file_counts", {})
        # Older entries without file counts cannot be combined accurately.
        complete = bool(entry["dataset_scores"]) and all(
            counts.get(name, 0) > 0
            and all(name in entry.get(field, {}) for field, _ in metric_fields)
            for name in entry["dataset_scores"]
        )
        if parameters.get("format") not in {"raw", "mzML"} or not complete:
            grouped.append(entry)
            continue
        key = (
            entry["timestamp"],
            tuple(sorted(
                (name, value) for name, value in parameters.items()
                if name not in {"format", "test_datasets"}
            )),
        )
        groups[key].append(entry)

    for parts in groups.values():
        # Keep already combined raw entries unchanged on subsequent loads.
        if len(parts) == 1 and parts[0]["parameters"]["format"] == "raw":
            grouped.extend(parts)
            continue
        # Multiple entries for one format may be separate partial runs.
        if len({part["parameters"]["format"] for part in parts}) != len(parts):
            grouped.extend(parts)
            continue
        merged = dict(parts[0])
        parameters = dict(merged["parameters"])
        parameters["format"] = "raw"
        names = sorted({name for part in parts for name in part["dataset_scores"]})
        if "test_datasets" in parameters:
            parameters["test_datasets"] = ", ".join(names)
        counts = {
            name: sum(part["dataset_file_counts"][name] for part in parts if name in part["dataset_scores"])
            for name in names
        }
        for field, average_field in metric_fields:
            values = {
                name: float(np.average(
                    [part[field][name] for part in parts if name in part["dataset_scores"]],
                    weights=[part["dataset_file_counts"][name] for part in parts if name in part["dataset_scores"]],
                ))
                for name in names
            }
            merged[field] = values
            merged[average_field] = float(np.average(
                [values[name] for name in names], weights=[counts[name] for name in names],
            ))
        outcomes = {}
        for name in names:
            statuses = [
                part.get("dataset_outcomes", {}).get(name)
                for part in parts if name in part["dataset_scores"]
            ]
            if any(statuses):
                outcomes[name] = True
            elif all(status is False for status in statuses):
                outcomes[name] = False
        merged.update(
            parameters=parameters,
            dataset_count=len(names),
            dataset_file_counts=counts,
            dataset_outcomes=outcomes,
        )
        merged["failed_tests"], merged["total_tests"] = _dataset_status_counts(names, outcomes)
        signature = " | ".join(f"{name}={value}" for name, value in parameters.items())
        merged["run_id"] = f"{merged['timestamp']}::{signature}"
        grouped.append(merged)
    return grouped


def build_session_leaderboard_entries(
    dict_results,
    dict_full_results,
    param_names,
    timestamp: str | None = None,
    eligible_params: set[tuple[Any, ...]] | None = None,
    dataset_outcomes: dict[str, dict[tuple[Any, ...], bool]] | None = None,
) -> list[LeaderboardEntry]:
    if not dict_results or not param_names:
        return []

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    param_labels = list(param_names.values())[1:]

    dataset_f1_by_params = defaultdict(dict)
    dataset_file_counts_by_params = defaultdict(dict)
    dataset_precision_by_params = defaultdict(dict)
    dataset_recall_by_params = defaultdict(dict)

    for dataset_name, param_results in dict_results.items():
        for params, scores in param_results.items():
            if eligible_params is not None and params not in eligible_params:
                continue
            if scores:
                dataset_f1_by_params[params][dataset_name] = float(np.mean(scores))
                dataset_file_counts_by_params[params][dataset_name] = len(scores)

    for dataset_name, param_results in dict_full_results.items():
        for params, scores in param_results.items():
            if eligible_params is not None and params not in eligible_params:
                continue
            if not scores:
                continue
            dataset_precision_by_params[params][dataset_name] = float(np.mean([score[1] for score in scores]))
            dataset_recall_by_params[params][dataset_name] = float(np.mean([score[2] for score in scores]))

    entries = []

    for params, dataset_scores in dataset_f1_by_params.items():
        dataset_names = sorted(dataset_scores)
        statuses = {
            name: dataset_outcomes[name][params]
            for name in dataset_names
            if dataset_outcomes is not None and params in dataset_outcomes.get(name, {})
        }
        failed_tests, total_tests = _dataset_status_counts(dataset_names, statuses)
        parameters = {
            label: _stringify_value(value)
            for label, value in zip(param_labels, params)
        }
        parameters = _normalize_session_parameters(parameters)
        precision_values = [
            score[1]
            for name in dataset_names
            for score in dict_full_results.get(name, {}).get(params, [])
        ]
        recall_values = [
            score[2]
            for name in dataset_names
            for score in dict_full_results.get(name, {}).get(params, [])
        ]
        average_precision = float(np.mean(precision_values)) if precision_values else float("nan")
        average_recall = float(np.mean(recall_values)) if recall_values else float("nan")
        signature = " | ".join(
            f"{label}={parameters[label]}"
            for label in param_labels
        )
        entries.append(
            LeaderboardEntry(
                run_id=f"{timestamp}::{signature}",
                timestamp=timestamp,
                average_f1=float(np.average(
                    [dataset_scores[name] for name in dataset_names],
                    weights=[dataset_file_counts_by_params[params][name] for name in dataset_names],
                )),
                average_precision=average_precision,
                average_recall=average_recall,
                dataset_count=len(dataset_names),
                parameters=parameters,
                dataset_scores={name: dataset_scores[name] for name in dataset_names},
                dataset_file_counts={name: dataset_file_counts_by_params[params][name] for name in dataset_names},
                dataset_outcomes=statuses,
                dataset_precisions={
                    name: dataset_precision_by_params[params][name]
                    for name in dataset_names
                    if name in dataset_precision_by_params[params]
                },
                dataset_recalls={
                    name: dataset_recall_by_params[params][name]
                    for name in dataset_names
                    if name in dataset_recall_by_params[params]
                },
                failed_tests=failed_tests,
                total_tests=total_tests,
            )
        )

    entries = [LeaderboardEntry(**entry) for entry in _group_raw_file_entries([asdict(entry) for entry in entries])]
    return sorted(
        entries,
        key=lambda entry: (
            _descending_metric(entry.average_f1),
            _descending_metric(entry.average_precision),
            _descending_metric(entry.average_recall),
            entry.run_id,
        ),
    )


def load_leaderboard_history(
    history_path: Path = LEADERBOARD_HISTORY_PATH,
) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []

    with open(history_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected leaderboard history list in {history_path}")

    data = _group_raw_file_entries(data)
    for entry in data:
        parameters = entry.setdefault("parameters", {})
        for name in REMOVED_PARAMETERS:
            parameters.pop(name, None)
        entry["run_id"] = REMOVED_RUN_ID_PATTERN.sub("", entry.get("run_id", ""))
        parameters.update({
            name: value for name, value in _extract_model_parameters(parameters).items()
            if name != "model"
        })
        if not _has_value(parameters.get("loss_function")):
            parameters["loss_function"] = _display_parameters(entry)["loss_function"]
        entry["failed_tests"], entry["total_tests"] = _dataset_status_counts(
            entry["dataset_scores"], entry.get("dataset_outcomes", {}),
        )

    return data


def save_leaderboard_history(
    entries: list[dict[str, Any]],
    history_path: Path = LEADERBOARD_HISTORY_PATH,
) -> None:
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def _display_format(entry: dict[str, Any]) -> str:
    file_format = entry.get("parameters", {}).get("format", "").lower()
    return "raw" if file_format in {"raw", "mzml"} else file_format


def _group_display_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair formats for display without altering per-format history or identities."""
    groups = defaultdict(list)
    for entry in _group_raw_file_entries(entries):
        identity = tuple(
            (name, value) for name, value in _entry_identity(entry)
            if name not in {"format", "test_datasets"}
        )
        # Unknown formats cannot be paired with raw/XLSX results.
        if _display_format(entry) not in {"raw", "xlsx"}:
            identity += (("unpaired_run_id", entry["run_id"]),)
        groups[identity].append(entry)

    display_entries = []
    for parts in groups.values():
        # Multiple entries for one format may cover overlapping dataset subsets.
        # Keep those rows separate rather than double-counting a partial rerun.
        batches = [parts] if len({_display_format(part) for part in parts}) == len(parts) else [[part] for part in parts]
        for batch in batches:
            merged = dict(batch[0])
            merged["format_entries"] = {_display_format(part): part for part in batch}
            if len(batch) > 1:
                weights = []
                for part in batch:
                    counts = part.get("dataset_file_counts", {})
                    names = part["dataset_scores"]
                    complete = bool(names) and all(counts.get(name, 0) > 0 for name in names)
                    weights.append(sum(counts[name] for name in names) if complete else 0)
                for metric in ("average_f1", "average_precision", "average_recall"):
                    merged[metric] = float(np.average(
                        [part.get(metric, float("nan")) for part in batch], weights=weights,
                    )) if all(weights) else float("nan")
            display_entries.append(merged)
    return _sort_history(display_entries)


def _format_f1_status(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "-"
    failed, total = _dataset_status_counts(entry["dataset_scores"], entry.get("dataset_outcomes", {}))
    return f"{float(entry['average_f1']):.4f},{failed if failed is not None else '?'}/{total}"


def render_leaderboard_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "CandyCrunch Leaderboard\n\nNo entries recorded yet.\n"

    ordered_entries = _group_display_entries(entries)
    display_rows = [_display_parameters(entry) for entry in ordered_entries]
    for display_row in display_rows:
        display_row["pretraining"] = _format_pretraining(display_row)
        display_row["num_res_units"] = _format_res_units(display_row)
        display_row["transformer_parameters"] = _format_transformer_parameters(display_row)
        if "classifier_moe" in display_row:
            display_row["classifier_moe"] = _format_classifier_moe(display_row)
    param_columns = _parameter_columns(ordered_entries)
    headers = [
        "Rank",
        "Avg F1",
        "Avg Prec",
        "Avg Rec",
        "Avg-raw F1",
        "Avg-xlsx F1",
        *[_display_column_name(column) for column in param_columns],
    ]
    rows = []

    for index, (entry, display_row) in enumerate(zip(ordered_entries, display_rows), start=1):
        row = [
            index,
            f"{float(entry['average_f1']):.4f}",
            f"{float(entry.get('average_precision', float('nan'))):.4f}",
            f"{float(entry.get('average_recall', float('nan'))):.4f}",
            _format_f1_status(entry["format_entries"].get("raw")),
            _format_f1_status(entry["format_entries"].get("xlsx")),
        ]
        row.extend(display_row.get(column, "") for column in param_columns)
        rows.append(row)

    lines = [
        "CandyCrunch Leaderboard",
        "",
        "Overall averages weight each evaluated file equally across raw and XLSX, matching the terminal Weighted Summary. Raw and mzML results are combined as raw.",
        "Avg-raw F1 and Avg-xlsx F1 show F1,failed/evaluated datasets; ? means outcomes were not recorded and - means that format was not evaluated.",
        "Combined averages are nan if historical file counts are missing. Ambiguous partial reruns remain separate rows.",
        "Pretrain: T_S_Ep<epochs>_T<temperature>_Op<optimizer> (supervised), T_U_... (unsupervised), or F. SAM is the default; legacy entries with unknown mode omit S/U.",
        "",
        tabulate(rows, headers=headers, tablefmt="github"),
        "",
        "Per-entry dataset averages:",
    ]

    for index, (entry, display_row) in enumerate(zip(ordered_entries, display_rows), start=1):
        signature = ", ".join(
            f"{_display_column_name(column)}={display_row[column]}"
            for column in param_columns
            if _has_value(display_row.get(column))
        )
        lines.append(f"{index}. {signature}")
        lines.append(f"   avg_f1={float(entry['average_f1']):.4f}")
        for file_format, source in entry["format_entries"].items():
            dataset_scores = ", ".join(
                f"{dataset}={float(score):.4f}"
                for dataset, score in source.get("dataset_scores", {}).items()
            )
            mode = _display_parameters(source).get("pretraining", "False")
            lines.append(f"   {file_format or 'unknown'}: {_format_f1_status(source)}; pretraining={mode}; TestSets: {dataset_scores}")

    lines.append("")
    return "\n".join(lines)


def update_leaderboard_from_collector(
    collector,
    history_path: Path = LEADERBOARD_HISTORY_PATH,
    text_path: Path = LEADERBOARD_TEXT_PATH,
) -> list[dict[str, Any]]:
    eligible_params = None
    if hasattr(collector, "get_leaderboard_eligible_params"):
        eligible_params = collector.get_leaderboard_eligible_params()

    session_entries = build_session_leaderboard_entries(
        collector.dict_results,
        collector.dict_full_results,
        collector.param_names,
        eligible_params=eligible_params,
        dataset_outcomes=getattr(collector, "dataset_outcomes", None),
    )

    history = load_leaderboard_history(history_path)
    combined_entries = history + [asdict(entry) for entry in session_entries]
    best_entries = _deduplicate_best_entries(combined_entries)

    if not best_entries and not history and not session_entries:
        return []

    save_leaderboard_history(best_entries, history_path)

    rendered = render_leaderboard_text(best_entries)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(rendered)

    return best_entries
