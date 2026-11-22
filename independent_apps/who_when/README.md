# Who&When independent project

This directory owns the current Who&When implementation. Its diagnosis remains the dataset's `ReverseDependencyDFS` plus Stage 2 root selector; it does not use TRAIL/MAADBench's SharedDiagnosisEngine. It imports neither the legacy `trace_diagnosis` package nor either sibling project. The one shared launcher and common runtime settings are `main.py` and `independent_apps/shared/`; all Who&When processing remains here.

## Runtime and data

Run commands from the repository root with the existing project Python environment. The repository has no `requirements.txt`, `pyproject.toml`, `setup.py`, or Conda environment file, so this split does not invent an installation recipe. The baseline ran with `D:\anaconda\anaconda\envs\OpenAI\python.exe` and the dependencies present in that environment.

Inputs default to `Who&When\Algorithm-Generated` and `Who&When\Hand-Crafted`, as defined in this project's `registry.py`. Set `TRACE_DIAGNOSIS_BASE_DIR` to change the base data directory. Outputs default to `outputs\independent\who_when`; set `WHO_WHEN_OUTPUT_DIR` to change the project-private output root. Graph JSONL, checkpoint/prediction JSON, and evaluation JSON are kept under that root unless explicitly overridden.

## Commands

```powershell
python main.py --help
python main.py --dataset who_when_algorithm --help
python main.py --dataset who_when_algorithm --input "Who&When\Algorithm-Generated" --convert-only --graph-path outputs\independent\who_when\algorithm_graphs.jsonl
```

Diagnosis uses the selected model/API config and can be limited with `--end`:

```powershell
python main.py --dataset who_when_algorithm --skip-convert --graph-path outputs\independent\who_when\algorithm_graphs.jsonl --checkpoint-file outputs\independent\who_when\algorithm_predictions.json --end 5 --llm-model <model-id> --api-config <external-api-config.json>
```

The diagnosis command sends model requests when run. The API JSON stays outside the repository; the project default is `E:\Work\paper9\api_config.json`, and `--api-config` takes precedence. The CLI accepts `api_key`, `base_url`, and optional `model`; credentials are never copied into this project.

Evaluate an existing checkpoint without diagnosis calls:

```powershell
python main.py --dataset who_when_algorithm --input "Who&When\Algorithm-Generated" --skip-convert --graph-path outputs\independent\who_when\algorithm_graphs.jsonl --checkpoint-file outputs\independent\who_when\algorithm_predictions.json --evaluation-output outputs\independent\who_when\algorithm_evaluation.json --evaluate-only
```

Evaluation uses the copied Who&When evaluation adapter and existing official identity/metric protocol. `--evaluate-only` reads the supplied graph, prediction file, and reference input; it does not call a model.

## Checkpoints and code map

The checkpoint records trace identities and diagnosis output using the copied Who&When runner/I/O. Keep separate files when changing model or settings; do not point this project at legacy checkpoints unless their identity and schema have been checked. Core files are `datasets/who_when/diagnoser.py`, `causal_traversal.py`, `causal_state.py`, `stage2_payload.py`, `runner.py`, `common/llm.py`, and `evaluation/who_when.py`.

The single-entry architecture, migration map, and offline evidence are in `../../docs/dataset_split/08_SHARED_ENTRYPOINT_MIGRATION_MAP.md`, `../../docs/dataset_split/09_DATASET_ISOLATION_REPORT.md`, and `../../docs/dataset_split/10_BEHAVIOR_PARITY_REPORT.md`.
