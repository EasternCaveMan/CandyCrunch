from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
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
    "contrastive_learning",
    "contrastive_weight",
    "contrastive_temperature",
}
DISPLAY_PARAMETER_ORDER = [
    "model",
    "loss_function",
    "candidate_sets",
    "pretraining",
    "pretraining_epochs",
    "supcon_temperature",
    "format",
    "split",
    "dataset",
    "encoder_type",
    "num_experts",
    "moe_top_k",
    "classifier_moe",
    "classifier_num_experts",
    "classifier_top_k",
    "classifier_expert_hidden_dim",
    "nheads",
    "nlayers",
    "peak_hidden_dim",
    "ff_dim",
    "max_peaks",
    "peak_encoder",
    "norm_type",
    "activation",
    "encoder_activation",
    "use_transformer_ff",
    "use_resunits",
]
DISPLAY_COLUMN_NAMES = {
    "Test Datasets": "TestSets",
    "loss_function": "LossFun",
    "candidate_sets": "CandSets",
    "pretraining": "Pretrain",
    "pretraining_epochs": "PtEpochs",
    "supcon_temperature": "SupConTemp",
    "combine_loss": "CombLoss",
    "contrastive_loss_weight": "ConLossWt",
    "contrastive_loss_temperature": "ConLossTemp",
    "encoder_type": "encTyp",
    "num_experts": "moe_exp",
    "moe_top_k": "moe_top",
    "classifier_moe": "MoC",
    "classifier_num_experts": "moc_exp",
    "classifier_top_k": "moc_top",
    "classifier_expert_hidden_dim": "ClaExpHidDim",
    "peak_hidden_dim": "PkHidDim",
    "max_peaks": "MaxPk",
    "peak_encoder": "PkEnc",
    "norm_type": "NorTyp",
    "activation": "Act",
    "encoder_activation": "EncAct",
    "use_transformer_ff": "UseTransFF",
    "use_resunits": "UseRes",
}
TRANSFORMER_PREFIX = "CandyCrunch_Transformer_"
CNN_PREFIX = "CandyCrunch_CNN_"
LOSS_TAGS = {
    "XYZLOSS": "xyz_loss",
    "CELOSS": "cross_entropy",
    "FOLLOSS": "focal_loss",
    "FOCALLOSS": "focal_loss",
    "POLOSS": "PolyCrEnr",
    "CMLOSS": "custom_loss",
}
PRETRAINING_TAG_PATTERN = re.compile(
    r"_(?P<tag>SupConPtE|PretrainE)(?P<pretraining_epochs>\d+)T(?P<supcon_temperature>[0-9.eE+-]+)(?=_)"
)
LEGACY_CONTRASTIVE_TAG_PATTERN = re.compile(
    r"_SupConW[0-9.eE+-]+T[0-9.eE+-]+(?=_|$)"
)
COMBINED_LOSS_TAG_PATTERN = re.compile(
    r"_CombSupConW(?P<contrastive_loss_weight>[0-9.eE+-]+)T(?P<contrastive_loss_temperature>[0-9.eE+-]+)(?=_|$)"
)


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

    match = re.fullmatch(
        r"(?:CMOE|MoC)(?P<classifier_num_experts>\d+)K(?P<classifier_top_k>\d+)(?:H(?P<classifier_expert_hidden_dim>\d+))?",
        tag,
    )
    if match:
        parsed = {
            "classifier_moe": "True",
            "classifier_num_experts": match.group("classifier_num_experts"),
            "classifier_top_k": match.group("classifier_top_k"),
        }
        hidden_dim = match.group("classifier_expert_hidden_dim")
        if hidden_dim is not None:
            parsed["classifier_expert_hidden_dim"] = hidden_dim
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

        max_peaks_match = re.fullmatch(r"MP(?P<max_peaks>\d+)", token)
        if max_peaks_match:
            parsed["max_peaks"] = max_peaks_match.group("max_peaks")
            continue

        peak_encoder = _parse_parenthesized_token(token, "PE")
        if peak_encoder is not None:
            parsed["peak_encoder"] = peak_encoder
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

        use_resunits = _parse_parenthesized_token(token, "RU")
        if use_resunits is not None:
            parsed["use_resunits"] = use_resunits

    return parsed


def _parse_cnn_stem(stem: str) -> dict[str, str]:
    parsed = {"model": "CNN"}
    if not stem.startswith(CNN_PREFIX):
        return parsed

    tokens = _split_model_tokens(stem[len(CNN_PREFIX):])
    for loss_tag, loss_function in LOSS_TAGS.items():
        if loss_tag in tokens:
            parsed["loss_function"] = loss_function
            tokens = [token for token in tokens if token != loss_tag]
    tokens = [token for token in tokens if token != "CS"]
    if len(tokens) >= 2:
        parsed["split"] = tokens[-2]
        parsed["dataset"] = tokens[-1]
    if len(tokens) > 2:
        parsed.update(_parse_classifier_tag("_".join(tokens[:-2])))
    return parsed


def _strip_pretraining_tag(stem: str) -> tuple[str, dict[str, str]]:
    match = PRETRAINING_TAG_PATTERN.search(stem)
    if not match:
        return stem, {}
    parsed = {
        "pretraining": "True",
        "pretraining_epochs": match.group("pretraining_epochs"),
        "supcon_temperature": str(float(match.group("supcon_temperature"))),
    }
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
    model_name = training_args.get("model") or payload.get("model_type")
    if not _has_value(model_name):
        model_class = payload.get("model_class")
        if model_class == "CandyCrunch_Transformer":
            model_name = "Transformer"
        elif model_class == "CandyCrunch_CNN":
            model_name = "CNN"
    _set_if_present(parsed, "model", model_name)
    _set_if_present(parsed, "loss_function", payload.get("loss_function"), training_args.get("loss_function"))
    _set_if_present(parsed, "candidate_sets", training_args.get("candidate_sets"))
    _set_if_present(parsed, "combine_loss", payload.get("combine_loss"), training_args.get("combine_loss"))
    _set_if_present(
        parsed,
        "contrastive_loss_weight",
        payload.get("contrastive_loss_weight"),
        training_args.get("contrastive_loss_weight"),
    )
    _set_if_present(
        parsed,
        "contrastive_loss_temperature",
        payload.get("contrastive_loss_temperature"),
        training_args.get("contrastive_loss_temperature"),
        training_args.get("supcon_temperature"),
    )
    if parsed.get("combine_loss") != "True":
        parsed.pop("contrastive_loss_weight", None)
        parsed.pop("contrastive_loss_temperature", None)
    _set_if_present(parsed, "pretraining", training_args.get("pretraining"))
    _set_if_present(parsed, "pretraining_epochs", training_args.get("pretraining_epochs"))
    _set_if_present(parsed, "supcon_temperature", training_args.get("supcon_temperature"))
    if parsed.get("pretraining") != "True":
        parsed.pop("pretraining_epochs", None)
        parsed.pop("supcon_temperature", None)
    _set_if_present(parsed, "split", training_args.get("split"), payload.get("split"))
    _set_if_present(parsed, "dataset", training_args.get("dataset"), payload.get("dataset"))
    _set_if_present(parsed, "max_peaks", training_args.get("max_peaks"), payload.get("max_peaks"))
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
        _set_if_present(parsed, "use_resunits", training_args.get("use_resunits"), model_kwargs.get("use_resunits"))

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
        elif model_text.startswith((TRANSFORMER_PREFIX, CNN_PREFIX)):
            stem = _model_stem(model_text)
        else:
            if not model_text:
                return {}
            return {"model": _display_model_value(model_value)}

    combine_loss_match = COMBINED_LOSS_TAG_PATTERN.search(stem)
    if combine_loss_match:
        stem = stem[:combine_loss_match.start()] + stem[combine_loss_match.end():]
    stem = LEGACY_CONTRASTIVE_TAG_PATTERN.sub("", stem)
    stem, pretraining_parameters = _strip_pretraining_tag(stem)
    if stem.startswith(TRANSFORMER_PREFIX):
        parsed_from_stem = _parse_transformer_stem(stem)
    elif stem.startswith(CNN_PREFIX):
        parsed_from_stem = _parse_cnn_stem(stem)

    parsed_from_stem.update(pretraining_parameters)
    if combine_loss_match:
        parsed_from_stem.update(
            combine_loss="True",
            contrastive_loss_weight=str(float(combine_loss_match.group("contrastive_loss_weight"))),
            contrastive_loss_temperature=str(float(combine_loss_match.group("contrastive_loss_temperature"))),
        )

    merged_parameters = _merge_model_parameters(parsed_from_stem, checkpoint_parameters)
    if merged_parameters:
        return _finalize_model_parameters(merged_parameters)

    return {"model": _display_model_value(model_checkpoint or raw_parameters.get("model", ""))}


def _normalize_session_parameters(parameters: dict[str, str]) -> dict[str, str]:
    normalized = dict(parameters)
    resolved_checkpoint = _resolve_model_checkpoint(normalized.get("model", ""))
    if resolved_checkpoint is not None:
        normalized["model_checkpoint"] = resolved_checkpoint
    normalized["loss_function"] = _display_parameters({"parameters": normalized})["loss_function"]
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
        if key == "model" or key in HIDDEN_PARAMETER_COLUMNS:
            continue
        display_parameters[key] = _stringify_value(value)

    for key, value in model_parameters.items():
        if key not in HIDDEN_PARAMETER_COLUMNS and _has_value(value):
            display_parameters[key] = _stringify_value(value)

    display_parameters.setdefault("loss_function", "custom_loss")
    return display_parameters


def _parameter_columns(entries: list[dict[str, Any]]) -> list[str]:
    display_rows = [_display_parameters(entry) for entry in entries]
    columns = list(DISPLAY_PARAMETER_ORDER)

    for row in display_rows:
        for key in row:
            if key not in columns:
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


def build_session_leaderboard_entries(
    dict_results,
    dict_full_results,
    param_names,
    timestamp: str | None = None,
    eligible_params: set[tuple[Any, ...]] | None = None,
) -> list[LeaderboardEntry]:
    if not dict_results or not param_names:
        return []

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    param_labels = list(param_names.values())[1:]

    dataset_f1_by_params = defaultdict(dict)
    dataset_precision_by_params = defaultdict(dict)
    dataset_recall_by_params = defaultdict(dict)

    for dataset_name, param_results in dict_results.items():
        for params, scores in param_results.items():
            if eligible_params is not None and params not in eligible_params:
                continue
            if scores:
                dataset_f1_by_params[params][dataset_name] = float(np.mean(scores))

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
        parameters = {
            label: _stringify_value(value)
            for label, value in zip(param_labels, params)
        }
        parameters = _normalize_session_parameters(parameters)
        precision_values = [
            dataset_precision_by_params[params][name]
            for name in dataset_names
            if name in dataset_precision_by_params[params]
        ]
        recall_values = [
            dataset_recall_by_params[params][name]
            for name in dataset_names
            if name in dataset_recall_by_params[params]
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
                average_f1=float(np.mean([dataset_scores[name] for name in dataset_names])),
                average_precision=average_precision,
                average_recall=average_recall,
                dataset_count=len(dataset_names),
                parameters=parameters,
                dataset_scores={name: dataset_scores[name] for name in dataset_names},
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
            )
        )

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

    for entry in data:
        parameters = entry.setdefault("parameters", {})
        if not _has_value(parameters.get("loss_function")):
            parameters["loss_function"] = _display_parameters(entry)["loss_function"]

    return data


def save_leaderboard_history(
    entries: list[dict[str, Any]],
    history_path: Path = LEADERBOARD_HISTORY_PATH,
) -> None:
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def render_leaderboard_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "CandyCrunch Leaderboard\n\nNo entries recorded yet.\n"

    ordered_entries = _sort_history(entries)
    display_rows = [_display_parameters(entry) for entry in ordered_entries]
    param_columns = _parameter_columns(ordered_entries)
    headers = [
        "Rank",
        "Avg F1",
        "Avg Prec",
        "Avg Rec",
        _display_column_name("Test Datasets"),
        *[_display_column_name(column) for column in param_columns],
    ]
    rows = []

    for index, (entry, display_row) in enumerate(zip(ordered_entries, display_rows), start=1):
        row = [
            index,
            f"{float(entry['average_f1']):.4f}",
            f"{float(entry.get('average_precision', float('nan'))):.4f}",
            f"{float(entry.get('average_recall', float('nan'))):.4f}",
            int(entry["dataset_count"]),
        ]
        row.extend(display_row.get(column, "") for column in param_columns)
        rows.append(row)

    lines = [
        "CandyCrunch Leaderboard",
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
        dataset_scores = ", ".join(
            f"{dataset}={float(score):.4f}"
            for dataset, score in entry.get("dataset_scores", {}).items()
        )
        lines.append(f"{index}. {signature}")
        lines.append(f"   avg_f1={float(entry['average_f1']):.4f}; TestSets: {dataset_scores}")

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
