# TRAIL independent project

This directory owns the current TRAIL implementation for GAIA and SWE-bench. It includes its own physical copy of the diagnosis engine, H0/H1 Cause State policies, converters, runner, LLM client, checkpoint I/O, and evaluator. It imports neither the legacy `trace_diagnosis` package nor either sibling project. CLI parsing and common runtime settings are the single thin interface under `independent_apps/shared/`; Trail-specific flags are registered by this project.

## Runtime and data

Run from the repository root with the existing project Python environment. The repository has no `requirements.txt`, `pyproject.toml`, `setup.py`, or Conda environment file, so this split does not invent an installation recipe. The baseline ran with `D:\anaconda\anaconda\envs\OpenAI\python.exe` and the dependencies present in that environment.

The registry defaults to `gaia-00000-of-00001-33a2e72d362d688a.parquet` and `swe_bench-00000-of-00001-91aa04220f7198b4.parquet` under the configured base directory. Set `TRACE_DIAGNOSIS_BASE_DIR` to change the data base. Outputs default to `outputs\independent\trail`; set `TRAIL_OUTPUT_DIR` to change this project's output root. Model API configuration is external and defaults to `E:\Work\paper9\api_config.json`; pass `--api-config` to select another config. The config contains `api_key`, `base_url`, and optionally `model`; do not place credentials in the repository.

## Commands

```powershell
python main.py --help
python main.py --dataset trail_gaia --help
python main.py --dataset trail_gaia --input .\gaia-00000-of-00001-33a2e72d362d688a.parquet --convert-only --graph-path outputs\independent\trail\gaia_graphs.jsonl
```

To diagnose up to five Graphs using a prebuilt graph:

```powershell
python main.py --dataset trail_gaia --skip-convert --graph-path outputs\independent\trail\gaia_graphs.jsonl --checkpoint-file outputs\independent\trail\gaia_predictions.json --end 5 --llm-model <model-id> --api-config <external-api-config.json>
```

That command makes model requests. For separate Cascade-first diagnosis, set `--cascade-call-mode separate --separate-order cascade_then_local`; for Local-only use `--no-cascade`. H0/H1 is selected with `--trail-cause-state-mode H0|H1`. Graph construction's `--edge-types` selects physical edges and is independent from the H0/H1 diagnostic EdgePolicy; every subset arm should use its own Graph and checkpoint path.

Evaluate an existing prediction file without model calls:

```powershell
python main.py --dataset trail_gaia --input .\gaia-00000-of-00001-33a2e72d362d688a.parquet --skip-convert --graph-path outputs\independent\trail\gaia_graphs.jsonl --checkpoint-file outputs\independent\trail\gaia_predictions.json --evaluation-output outputs\independent\trail\gaia_evaluation.json --evaluate-only
```

This invokes the copied TRAIL evaluator against the supplied reference input and graph. Official metrics are computed from the `errors`/`scores` prediction fields; Cascade/Root metadata is retained for diagnosis audits and does not replace the official prediction contract.

## Checkpoints and code map

The runner fingerprint binds predictions to graph and runtime identity. Keep outputs separate by dataset, model, H0/H1, call mode/order, and physical edge subset. The main implementation is `datasets/trail/diagnoser.py` plus this project's `common/diagnosis/engine.py`, `policies.py`, and `path_state.py`; converter, runner, CLI, and evaluator are all private copies.

See `../../docs/dataset_split/08_SHARED_ENTRYPOINT_MIGRATION_MAP.md`, `../../docs/dataset_split/09_DATASET_ISOLATION_REPORT.md`, and `../../docs/dataset_split/10_BEHAVIOR_PARITY_REPORT.md` for the module map and offline verification evidence.
