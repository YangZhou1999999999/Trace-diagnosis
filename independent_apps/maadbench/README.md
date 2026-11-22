# MAADBench independent project

This directory owns the current MAADBench Action diagnosis implementation. It has its own physical copy of the diagnosis engine, MAAD Legacy/Path Cause State policies, graph converter, runner, LLM client, checkpoint I/O, split handling, and evaluator. It imports neither the legacy `trace_diagnosis` package nor either sibling project. CLI parsing and common runtime settings are centralized under `independent_apps/shared/`; MAADBench-only flags are registered by this project.

## Runtime and data

Run from the repository root with the existing project Python environment. The repository has no `requirements.txt`, `pyproject.toml`, `setup.py`, or Conda environment file, so this split does not invent an installation recipe. The baseline ran with `D:\anaconda\anaconda\envs\OpenAI\python.exe` and the dependencies already present in that environment.

The registry defaults to `Data\gpt-4.1`; set `MAADBENCH_DATA_DIR` to use another data root and `TRACE_DIAGNOSIS_BASE_DIR` to move the repository data base. The official benchmark utilities are external and are located from the configured `--input`/`--maadbench-official-root`. Outputs default to `outputs\independent\maadbench`; set `MAADBENCH_OUTPUT_DIR` to change this project's root. API configuration is external (default `E:\Work\paper9\api_config.json`) and can be selected with `--api-config`. No credentials are stored here.

## Commands

```powershell
python main.py --help
python main.py --dataset maadbench --help
python main.py --dataset maadbench --input Data\gpt-4.1 --convert-only --graph-path outputs\independent\maadbench\maadbench_graphs.jsonl
```

To run a small `official_dev` slice with an existing Graph:

```powershell
python main.py --dataset maadbench --input Data\gpt-4.1 --skip-convert --graph-path outputs\independent\maadbench\maadbench_graphs.jsonl --checkpoint-file outputs\independent\maadbench\predictions.json --end 5 --maadbench-diagnosis-scope official_dev --llm-model <model-id> --api-config <external-api-config.json>
```

The diagnosis command makes model requests. MAADBench keeps its all-Action local scheduling and the official Action `has_error`/continuous `anomaly_score` outputs. `--maadbench-cause-state-mode` selects `legacy` or `path`; call mode is `combined` or `separate`, and Local-only uses `--no-cascade`. The graph `--edge-types` option controls graph construction only. Use a separate graph/checkpoint path for each edge subset and Cause State arm.

Evaluate existing predictions without model calls:

```powershell
python main.py --dataset maadbench --input Data\gpt-4.1 --skip-convert --graph-path outputs\independent\maadbench\maadbench_graphs.jsonl --checkpoint-file outputs\independent\maadbench\predictions.json --evaluation-output outputs\independent\maadbench\evaluation.json --evaluate-only
```

Evaluation uses this project's copied MAADBench split/evaluator implementation and external official benchmark utilities. The configured evaluation profile and score source are explicit CLI options; historical `legacy` and `local_complete_cause_partial` views remain distinct.

## Checkpoints and code map

Checkpoint identity includes Graph/runtime configuration; do not reuse checkpoints between model, edge subset, Cause State mode, split, or Separate order. Core files are `datasets/maadbench/diagnoser.py`, `datasets/maadbench/runner.py`, this project's `common/diagnosis/engine.py`, `policies.py`, `path_state.py`, `datasets/maadbench/graph_converter.py`, and `evaluation/maadbench.py`.

See `../../docs/dataset_split/08_SHARED_ENTRYPOINT_MIGRATION_MAP.md`, `../../docs/dataset_split/09_DATASET_ISOLATION_REPORT.md`, and `../../docs/dataset_split/10_BEHAVIOR_PARITY_REPORT.md` for the migration map and offline evidence.
