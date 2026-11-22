"""Thin adapter around the MAADBench authors' split and scoring utilities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from sklearn.model_selection import train_test_split


def official_root_for_input(input_path: str, override: Optional[str] = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    source_path = Path(input_path).expanduser().resolve()
    # The default MAADBench layout is MAADBench/Data/<source-model>/...
    if source_path.name.lower() in {"gpt-4.1", "gpt-5.4", "claude-opus-4-8", "claude-sonnet-4-6", "deepseek-reasoner"}:
        preferred = source_path.parent.parent
    elif source_path.name.lower() == "data":
        preferred = source_path.parent
    else:
        preferred = source_path.parent

    def has_utility(root: Path) -> bool:
        return (root / "benchmark_AD" / "utils.py").is_file() or (root / "utils.py").is_file()

    if has_utility(preferred):
        return preferred
    # The trace archive can be copied into Final/Data while its official helper
    # remains in the adjacent MAADBench source checkout.
    for ancestor in source_path.parents:
        adjacent = ancestor / "MAADBench"
        if has_utility(adjacent):
            return adjacent
    return preferred


def load_official_utils(official_root: Path):
    candidates = (
        official_root / "benchmark_AD" / "utils.py",
        official_root / "utils.py",
    )
    utility_path = next((path for path in candidates if path.is_file()), None)
    if utility_path is None:
        raise FileNotFoundError(
            "Could not locate MAADBench official utilities. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    for path in (utility_path.parent, utility_path.parent.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_name = "independent_apps_maadbench_official_utils"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() != utility_path.resolve():
        raise RuntimeError(
            f"A different MAADBench utility module is already loaded: {existing.__file__}"
        )
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, utility_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MAADBench utility module at {utility_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_official_args(input_path: str, run_dir: str, options: dict[str, Any]) -> SimpleNamespace:
    dataset = str(options.get("maadbench_domain", "gsm-hard"))
    dataset_name = {
        "gsm-hard": "EscapeRoom_v3_gsm_hard",
        "livecodebench": "EscapeRoom_v3_livecodebench",
    }.get(dataset)
    if dataset_name is None:
        raise ValueError("MAADBench domain must be 'gsm-hard' or 'livecodebench'")
    source_models = options.get("maadbench_source_models") or [Path(input_path).name]
    if isinstance(source_models, str):
        source_models = [source_models]
    temperatures = options.get("maadbench_source_temperatures") or [0.0]
    if isinstance(temperatures, (int, float)):
        temperatures = [float(temperatures)]
    return SimpleNamespace(
        path_traces=str(Path(input_path).expanduser()),
        trace_dataset_name=dataset_name,
        trace_type=str(options.get("maadbench_trace_type", "tool")),
        dataset_level="action",
        temp_llm=[float(value) for value in temperatures],
        llm_in_mas=[str(value) for value in source_models],
        seed=int(options.get("maadbench_seed", 5)),
        test_size=float(options.get("maadbench_test_size", 0.2)),
        unknown_status=str(options.get("maadbench_unknown_status", "error")),
        run_dir=str(Path(run_dir).expanduser()),
        threshold_policy=str(options.get("maadbench_threshold_policy", "oracle")),
        anomaly_fraction=float(options.get("maadbench_anomaly_fraction", 0.1)),
    )


def prepare_official_test_split(
    input_path: str,
    run_dir: str,
    options: dict[str, Any],
) -> tuple[Any, SimpleNamespace, Any, Any, Path]:
    """Recreate the official test split and save its manifests for auditability."""
    official_root = official_root_for_input(
        input_path,
        str(options["maadbench_official_root"]) if options.get("maadbench_official_root") else None,
    )
    official = load_official_utils(official_root)
    output_dir = Path(run_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = make_official_args(input_path, str(output_dir), options)
    train_df, test_df = official.setup_datasets(args)
    official.save_split_manifests(args, train_df, test_df)
    sources = sorted({str(value).replace("\\", "/") for value in test_df.source_file})
    (output_dir / "test_sources.json").write_text(
        json.dumps({"source_files": sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "official_root": str(official_root),
        "path_traces": str(Path(input_path).expanduser()),
        "dataset": args.trace_dataset_name,
        "trace_type": args.trace_type,
        "source_models": args.llm_in_mas,
        "source_temperatures": args.temp_llm,
        "seed": args.seed,
        "test_size": args.test_size,
        "unknown_status": args.unknown_status,
        "threshold_policy": args.threshold_policy,
        "anomaly_fraction": args.anomaly_fraction,
        "dataset_level": "action",
    }
    (output_dir / "eval_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return official, args, train_df, test_df, output_dir


def prepare_official_dev_split(
    input_path: str,
    run_dir: str,
    options: dict[str, Any],
) -> tuple[Any, SimpleNamespace, Any, Any, Path]:
    """Create a deterministic, room-disjoint dev holdout from official train."""
    official, args, train_df, _test_df, output_dir = prepare_official_test_split(
        input_path, run_dir, options,
    )
    dev_size = float(options.get("maadbench_dev_size", 0.2))
    dev_seed = int(options.get("maadbench_dev_seed", int(options.get("maadbench_seed", 5)) + 101))
    if not 0.0 < dev_size < 1.0:
        raise ValueError("MAADBench development holdout size must be between 0 and 1")
    if "room_group" not in train_df or "step_label" not in train_df:
        raise ValueError("Official training rows need room_group and step_label for a dev split")

    room_labels = train_df.groupby("room_group", sort=True)["step_label"].max()
    try:
        _inner_train_groups, dev_groups = train_test_split(
            room_labels.index.to_numpy(),
            test_size=dev_size,
            stratify=room_labels.to_numpy(),
            random_state=dev_seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not make a stratified room-disjoint development holdout from the official "
            f"training split (rooms={len(room_labels)}, labels={room_labels.value_counts().to_dict()}, "
            f"dev_size={dev_size}, seed={dev_seed})."
        ) from exc

    dev_pool = train_df[train_df["room_group"].isin(set(dev_groups))].copy().reset_index(drop=True)
    dev_df = dev_pool
    dev_limit = options.get("maadbench_dev_trace_limit")
    dev_pilot_seed = int(options.get("maadbench_dev_pilot_seed", 107))
    if dev_limit is not None:
        dev_limit = int(dev_limit)
        labels = dev_pool.groupby("room_group", sort=True)["step_label"].max()
        if dev_limit < 1 or dev_limit > len(labels):
            raise ValueError(f"MAADBench dev trace limit must be in 1..{len(labels)}")
        if dev_limit < len(labels):
            try:
                _unused_groups, selected_groups = train_test_split(
                    labels.index.to_numpy(),
                    test_size=dev_limit / len(labels),
                    stratify=labels.to_numpy(),
                    random_state=dev_pilot_seed,
                )
            except ValueError as exc:
                raise ValueError(
                    "Could not stratify the requested dev pilot subset "
                    f"(rooms={len(labels)}, labels={labels.value_counts().to_dict()}, limit={dev_limit})"
                ) from exc
            dev_df = dev_pool[dev_pool["room_group"].isin(set(selected_groups))].copy().reset_index(drop=True)
        else:
            dev_limit = None

    manifest_columns = [column for column in getattr(official, "MANIFEST_COLUMNS", []) if column in dev_df]
    if not manifest_columns:
        raise ValueError("Official MAADBench utility did not expose manifest columns")
    dev_manifest = dev_df[manifest_columns].copy()
    if dev_manifest.empty or not dev_manifest["row_id"].is_unique:
        raise ValueError("Development manifest is empty or has duplicate stable row IDs")
    dev_pool_manifest = dev_pool[[column for column in manifest_columns if column in dev_pool]].copy()
    dev_pool_manifest.to_csv(output_dir / "dev_pool_manifest.csv", index=False)
    dev_manifest.to_csv(output_dir / "dev_manifest.csv", index=False)

    config_path = output_dir / "eval_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "development_split": "room_group_holdout_from_official_train",
        "development_size": dev_size,
        "development_seed": dev_seed,
        "development_room_groups": int(dev_pool["room_group"].nunique()),
        "development_actions": int(len(dev_pool_manifest)),
        "development_pilot_seed": dev_pilot_seed if dev_limit is not None else None,
        "development_trace_limit": dev_limit,
        "development_evaluated_room_groups": int(dev_df["room_group"].nunique()),
        "development_evaluated_actions": int(len(dev_manifest)),
    })
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return official, args, train_df, dev_df, output_dir


def source_matches_trace_id(trace_id: str, source_file: str) -> bool:
    trace = str(trace_id).replace("\\", "/").strip("/").lower()
    source = str(source_file).replace("\\", "/").strip("/").lower()
    if not trace.endswith(".json"):
        trace += ".json"
    return trace == source or trace.endswith("/" + source) or source.endswith("/" + trace)

