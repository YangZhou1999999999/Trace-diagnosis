# Trace Diagnosis — User Guide

This guide describes the supported datasets, execution methods, public parameters and their accepted values, output files, and evaluation procedures. Users can normally run experiments from the command line or the accompanying Jupyter Notebook without modifying the source code.

## 1. Supported Datasets

| Dataset ID | Dataset | Input data |
|---|---|---|
| `who_when_algorithm` | Who&When Algorithm-Generated | JSON / JSONL file or directory |
| `who_when_handcrafted` | Who&When Hand-Crafted | JSON / JSONL file or directory |
| `trail_gaia` | TRAIL GAIA | Parquet / JSON / JSONL |
| `trail_swe_bench` | TRAIL SWE-bench | Parquet / JSON / JSONL |
| `maadbench` | MAADBench | Directory of MAADBench trace JSON files |

Each benchmark uses graph conversion, diagnosis, and evaluation procedures appropriate to its data format. Select the dataset ID corresponding to your input data. Do not reuse a graph or checkpoint from a different dataset.

## 2. Environment and API Configuration

### 2.1 Python Environment

Use the project's existing Python or Conda environment. The current package does not provide a version-pinned `requirements.txt`, `pyproject.toml`, or Conda environment file. Use a verified environment and ensure that the following dependencies are available:

- `openai`
- `httpx`
- `pydantic`
- `tiktoken`
- `pandas`
- `numpy`
- `scikit-learn`
- A Parquet reading backend (e.g., `pyarrow`)

The official MAADBench test/development splits and evaluation also require `benchmark_AD/utils.py` from the official MAADBench repository.

### 2.2 API Configuration JSON

Store the API configuration outside the project directory. Do not put API keys directly in the Notebook or commit them to the repository.

```json
{
  "api_key": "<YOUR_API_KEY>",
  "base_url": "<API_BASE_URL>",
  "model": "<OPTIONAL_MODEL_ID>"
}
```

- `api_key`: Required API key for the model service.
- `base_url`: Required model service API endpoint.
- `model`: Optional model identifier. An explicit `--llm-model` command-line argument or `MODEL_ID` Notebook setting takes precedence. Otherwise, the model from the configuration file or the project default is used.

Specify the configuration file with `--api-config` in the CLI or `API_CONFIG` in the Notebook. Use `--proxy` / `PROXY` to configure an HTTP proxy.

Graph conversion and evaluation do not call an LLM. Diagnosis makes model requests and may incur API charges.

## 3. Command-Line Usage

Run the following commands from the project root containing `main.py`.

### 3.1 View Help

```bash
python main.py --help
python main.py --dataset who_when_algorithm --help
python main.py --dataset trail_gaia --help
python main.py --dataset maadbench --help
```

`--dataset` selects the dataset for the current run. The available dataset-specific arguments depend on that selection.

### 3.2 Graph Conversion

Convert the input data without calling a model:

```bash
python main.py --dataset trail_gaia \
  --input <PATH_TO_GAIA_PARQUET> \
  --convert-only \
  --graph-path <PATH_TO_GRAPH_JSONL>
```

The output is typically a JSONL file containing one graph record per line.

### 3.3 Diagnosis

Run diagnosis using an existing graph:

```bash
python main.py --dataset trail_gaia \
  --input <PATH_TO_GAIA_PARQUET> \
  --skip-convert \
  --graph-path <PATH_TO_GRAPH_JSONL> \
  --checkpoint-file <PATH_TO_CHECKPOINT_JSON> \
  --llm-model <MODEL_ID> \
  --api-config <PATH_TO_API_CONFIG_JSON> \
  --end 5
```

- `--skip-convert`: Reuse an existing graph.
- `--end 5`: Diagnose only the first five records for a small-scale test.
- Omit `--end` to diagnose all records in the selected scope.

Without `--skip-convert`, the launcher converts the input data before running diagnosis.

### 3.4 Evaluation

Evaluate saved predictions without calling a model:

```bash
python main.py --dataset trail_gaia \
  --input <PATH_TO_GAIA_PARQUET> \
  --skip-convert \
  --graph-path <PATH_TO_GRAPH_JSONL> \
  --checkpoint-file <PATH_TO_CHECKPOINT_JSON> \
  --evaluation-output <PATH_TO_EVALUATION_JSON> \
  --evaluate-only
```

Evaluation requires the graph and original reference data corresponding to the checkpoint. MAADBench also requires access to its official evaluation utilities.

### 3.5 Run Multiple Stages Together

- `--convert-only`: Run graph conversion only.
- Default execution: Convert the input data, unless `--skip-convert` is specified, and then run diagnosis.
- `--evaluate`: Run evaluation after diagnosis.
- `--evaluate-only`: Evaluate an existing checkpoint only.
- Do not combine `--convert-only` with `--evaluate` or `--evaluate-only`.

## 4. Common Parameters

The following parameters are available across multiple datasets.

| Parameter | Accepted values / default | Purpose |
|---|---|---|
| `--dataset` | One of the five dataset IDs | Required dataset selection. |
| `--input` | Path; dataset default if omitted | Original input file or directory. |
| `--graph-path` | File or directory path | Graph JSONL location. |
| `--checkpoint-file` | File or directory path | Prediction and resumption checkpoint location. |
| `--end` | Positive integer; all records if omitted | Maximum number of records to diagnose. |
| `--skip-convert` | Flag | Skip graph conversion and reuse an existing graph. |
| `--convert-only` | Flag | Convert graphs without calling a model. |
| `--evaluate` | Flag | Run evaluation after diagnosis. |
| `--evaluate-only` | Flag | Evaluate existing predictions without calling a model. |
| `--evaluation-output` | File or directory path | Evaluation JSON output location. |
| `--llm-model` | Model ID | Diagnosis model identifier. |
| `--api-config` | JSON file path | External API configuration. |
| `--proxy` | HTTP proxy URL | Override proxy settings. |
| `--no-cascade` | Flag | Disable cascade inspection and use local-only diagnosis. |
| `--cascade-call-mode` | `combined` / `separate` | Select how local and cascade inspections are called. |
| `--no-ground-truth` | Flag | Exclude the top-level `ground_truth` field from model-visible graph data. |
| `--include-eval-labels` | Flag | Include evaluation labels in model-visible data; intended only for explicitly labeled debugging. |
| `--max-causes` | Positive integer; default 8 | Limit retained and propagated candidate causes. |
| `--traversal-mode` | `auto` / `bfs` / `dfs` | Select graph traversal mode. |
| `--max-prompt-upstream-chars` | Non-negative integer | Character budget for upstream context. |

### 4.1 `combined` and `separate`

The two modes differ in the way model calls are organized:

| Mode | Execution |
|---|---|
| `combined` | A joint model call performs local inspection and cascade attribution and returns both results. |
| `separate` | Local inspection and cascade attribution are performed in separate model calls. |

These call patterns refer to the local and cascade portions of node inspection, not the total number of requests for an entire trace. Other stages, such as final-hypothesis initialization, may require additional calls.

All three benchmarks support `--separate-order` to select the order in separate mode:

| Value | Order | Information used by the second stage |
|---|---|---|
| `local_then_cascade` | Local, then Cascade | Cascade uses the local inspection result. |
| `cascade_then_local` | Cascade, then Local | Local uses cascade attribution as evidence. |

Who&When defaults to `local_then_cascade`; TRAIL and MAADBench default to `cascade_then_local`. Either order can be selected explicitly. The `--no-cascade` flag disables the cascade stage.

### 4.2 Traversal Mode

- `bfs`: Breadth-first traversal.
- `dfs`: Depth-first traversal.
- `auto`: Use the benchmark's default traversal setting. Who&When defaults to DFS, while TRAIL and MAADBench default to BFS.

### 4.3 Prompt Context Policy

The available policies depend on the dataset:

- `legacy`: Use the existing upstream-context selection method.
- `ranked_budget`: Rank candidate upstream context and select content within the `--max-prompt-upstream-chars` character budget.
- `prioritized_two_kinds`: TRAIL only. Select up to two non-empty edge-type groups in the priority order `data_flow`, `execution_flow`, `parent`, `other`, subject to the character budget.

| Dataset | Supported prompt context policies |
|---|---|
| Who&When | `legacy`, `ranked_budget` |
| TRAIL | `legacy`, `ranked_budget`, `prioritized_two_kinds` |
| MAADBench | `legacy`, `ranked_budget` |

## 5. Graph Edge Types

`EDGE_TYPES=None` in the Notebook uses the full default edge set for the selected dataset. All three benchmarks also support selecting a non-empty subset of their allowed edge types.

| Edge type | Meaning |
|---|---|
| `sequential_flow` | Temporal order between adjacent steps or events. |
| `execution_flow` | Execution or control dependencies, such as reasoning to action or a parent action to a tool call. |
| `data_flow` | A data dependency between the output of an earlier node and the input of a later node. |
| `contains` | Containment between parent and child spans. |

| Dataset | Full default edge set | Custom edge selection |
|---|---|---|
| Who&When | `sequential_flow`, `execution_flow`, `data_flow` | Any non-empty subset of these three types. |
| TRAIL | `contains`, `execution_flow`, `data_flow` | Any non-empty subset of these three types. |
| MAADBench | `sequential_flow`, `execution_flow`, `data_flow` | Any non-empty subset of these three types. |

Command-line option (all three benchmarks):

```bash
--edge-types execution_flow data_flow
```

Equivalent Notebook setting:

```python
EDGE_TYPES = ("execution_flow", "data_flow")
```

**Who&When converter API:** The underlying `process_json_path(...)`, `process_json_directory(...)`, `convert_history_record(...)`, and `runner.convert_to_graphs(...)` interfaces accept an `edge_types` argument. The following example demonstrates a direct converter call:

```python
from independent_apps.who_when.datasets.who_when.graph_converter import process_json_path

graphs = process_json_path(
    input_path,
    output_path=graph_path,
    edge_types=("execution_flow", "data_flow"),
)
```

Changing the edge set changes the graph structure. Use distinct graph files and checkpoints for different edge configurations.

## 6. Who&When-Specific Parameters

| Parameter | Accepted values / default | Purpose |
|---|---|---|
| `--max-stage2-candidates` | Positive integer; Notebook default 20 | Maximum number of root-cause candidate nodes considered during Stage 2 selection. |
| `--include-full-analysis` | Flag | Include complete analysis details in prediction records. |
| `--prompt-context-policy` | `legacy` / `ranked_budget` | Select the upstream prompt context construction policy. |

The two Who&When dataset IDs correspond to the Algorithm-Generated and Hand-Crafted subsets.

## 7. TRAIL-Specific Parameters

### 7.1 Cause State: H0 / H1

| Value | Meaning |
|---|---|
| `H0` | Conventional cause state. Causes are merged primarily by cause ID, so the same cause arriving from different branches may share a state. |
| `H1` | Path-aware cause state. States are maintained separately by `(Cause ID, Incoming Path)`, so the same cause can be tracked independently along different paths without one branch overwriting another. Cause propagation uses `execution_flow` and `data_flow` dependency edges. |

H0/H1 controls how cause states are organized, rather than which physical edges are constructed by `--edge-types`. H1 can use a graph containing `contains` edges while restricting cause-state propagation to dependency edges.

### 7.2 Scheduling and Final Hypotheses

| Parameter | Accepted values | Meaning |
|---|---|---|
| `--trail-schedule-scope` | `legacy` / `all_nodes` | `legacy` retains the existing scheduling scope associated with final nodes; `all_nodes` includes every graph node in the diagnosis schedule. |
| `--trail-final-hypothesis-policy` | `gated` / `retain_candidates` | `gated` initializes upstream cause candidates only when the final output is judged to have failed; `retain_candidates` retains trace-supported upstream candidates even if the final output is not judged to have failed. |

In the Notebook, `TRAIL_SCHEDULE_SCOPE=None` uses the default `legacy`, and `TRAIL_FINAL_HYPOTHESIS_POLICY=None` uses the default `gated`.

### 7.3 Request and Token Budgets

| Parameter | Meaning |
|---|---|
| `--trail-max-completion-tokens` | Maximum completion tokens for a single model response. |
| `--trail-api-request-limit` | Maximum number of API requests allowed in a run; uses the project default if omitted. |
| `--trail-request-token-budget` | Enable request token-budget management. |
| `--trail-request-context-limit` | Request context token limit. |
| `--trail-request-hard-projected-limit` | Hard limit on projected tokens before a request is sent. |
| `--trail-request-completion-reserve` | Completion tokens reserved for the model output. |
| `--trail-request-safety-factor` | Safety multiplier for token estimates, allowing additional headroom. |
| `--trail-request-tokenizer` | Tokenizer used for token estimation; uses the project default if omitted. |
| `--enable-scoring` | Enable optional TRAIL scoring output. |
| `--use-llm-error-synthesis` | Enable an additional LLM-based error synthesis step. |

The corresponding Notebook variables use the `TRAIL_REQUEST_*` prefix and related TRAIL settings. If unsure about request budgets, retain the project defaults rather than increasing limits arbitrarily.

## 8. MAADBench-Specific Parameters

### 8.1 Diagnosis Scope

`--maadbench-diagnosis-scope` / Notebook `MAAD_SCOPE`:

| Value | Meaning |
|---|---|
| `all` | Diagnose all traces in the graph index. Evaluation still matches test actions using the official test manifest. |
| `official_test` | Diagnose only traces in the official test split. |
| `official_dev` | Diagnose the development split constructed from the official training portion by grouping rooms; evaluate using the corresponding development manifest. |

MAADBench graph conversion builds a full graph index of the source data. `MAAD_SCOPE` selects which traces are actually diagnosed. `END` limits diagnosis, not the construction of the full graph index.

### 8.2 Cause State: legacy / path

| Value | Meaning |
|---|---|
| `legacy` | Conventional cause-state management, merging and updating states by cause ID. |
| `path` | Path-aware cause-state management, maintaining independent states by `(Cause ID, Incoming Path)` and retaining evidence and states from different paths. |

`path` requires cascade inspection. `--maadbench-max-cause-path-states` / `MAAD_MAX_CAUSE_PATH_STATES` limits the number of cause-path states retained (default 4096). Some paths may be truncated when this limit is reached; inspect path coverage.

### 8.3 Evaluation Profile

`--maadbench-evaluation-profile` / `MAAD_EVALUATION_PROFILE`:

| Value | Meaning |
|---|---|
| `legacy` | Include only complete prediction records with `SUCCESS` / `OK` status. |
| `local_complete_cause_partial` | Additionally include path-mode records whose local predictions are complete but whose cause-path coverage is `PARTIAL`. Every action must have a valid local prediction; not all partial records qualify. |

When using `local_complete_cause_partial`, report the profile and the numbers of included and excluded partial records.

### 8.4 Score Source and Threshold Policy

`--maadbench-evaluation-score-source` / `MAAD_EVALUATION_SCORE_SOURCE`:

- `final`: Use the final action-level `anomaly_score`.
- `local`: Use the anomaly score from local inspection (`anomaly_score` in the local prediction or `local_anomaly_score`).

`--maadbench-threshold-policy` / `MAAD_THRESHOLD_POLICY`:

- `fixed`: Use the official fixed-threshold rule. `MAAD_ANOMALY_FRACTION` controls the anomaly fraction used by this rule (Notebook default 0.1).
- `oracle`: Use test-set ground-truth labels to determine the top-k threshold as an oracle comparison. This is not an unlabeled threshold available at deployment time and must not be confused with the primary `fixed` results.

### 8.5 Data Filtering and Official Splits

| Notebook / CLI parameter | Accepted values / purpose |
|---|---|
| `MAAD_DOMAIN` / `--maadbench-domain` | `gsm-hard` or `livecodebench`; select the official evaluation domain. |
| `MAAD_TRACE_TYPE` / `--maadbench-trace-type` | `tool` or `no_tool`; select the tool-use condition. |
| `MAAD_SOURCE_MODELS` / `--maadbench-source-model` | Filter by the model that generated the original traces; repeat the CLI option for multiple models. Defaults to the model corresponding to the input directory. |
| `MAAD_SOURCE_TEMPERATURES` / `--maadbench-source-temperature` | Filter by the temperature used to generate the original traces; repeat for multiple values. Defaults to 0.0 if omitted. |
| `--maadbench-seed` | Random seed for the official test split. |
| `--maadbench-test-size` | Official test split fraction; default 0.2. |
| `--maadbench-unknown-status` | Handle unknown labels: `error` raises an error, `anomaly` treats them as anomalous, and `drop` excludes them. |
| `MAAD_DEV_SIZE` / `--maadbench-dev-size` | Development split fraction taken from the official training portion. |
| `MAAD_DEV_SEED` / `--maadbench-dev-seed` | Development split random seed. |
| `MAAD_DEV_TRACE_LIMIT` / `--maadbench-dev-trace-limit` | Maximum number of traces for a development pilot; `None` uses the full development split. |
| `MAAD_DEV_PILOT_SEED` / `--maadbench-dev-pilot-seed` | Random seed for selecting a smaller pilot subset from the development pool. |
| `MAAD_OFFICIAL_ROOT` / `--maadbench-official-root` | Official MAADBench repository root containing `benchmark_AD/utils.py` or a discoverable `utils.py`. By default, discovery starts from the input path. |
| `MAAD_DUMP_VISIBLE_CONTEXT` / `--maadbench-dump-visible-context` | Export model-visible context for debugging; disabled by default. |

## 9. Using the Jupyter Notebook

Accompanying Notebook:

`trace_diagnosis_independent_run_and_evaluate_WhoWhen_cascade_call_mode.ipynb`

### 9.1 Location and Paths

Place the Notebook in the project root containing `independent_apps/`. If it is located elsewhere, set `PROJECT_ROOT_OVERRIDE` in Section 1.

### 9.2 Recommended Execution Order

1. **Section 1:** Check the project path and Python kernel.
2. **Section 2:** Select `DATASET`, the model, API configuration, input/output paths, and common parameters.
3. **Section 3:** Configure the selected dataset's specific parameters.
4. **Sections 4–5:** Construct graph, checkpoint, and evaluation paths and validate the parameters.
5. **Section 6:** Enable `RUN_CONVERSION=True` if a new graph is needed.
6. **Section 7:** Inspect graph counts, nodes, edges, and edge types.
7. **Section 8:** Enable `RUN_DIAGNOSIS=True` to diagnose traces. This stage makes model API requests.
8. **Section 9:** Inspect checkpoint status and failed records.
9. **Section 10:** Enable `RUN_EVALUATION=True` to evaluate saved predictions.
10. **Section 11:** View evaluation metrics and previous results for the same dataset.

`RUN_CONVERSION`, `RUN_DIAGNOSIS`, and `RUN_EVALUATION` are all `False` by default. Run the configuration and validation cells before enabling the required stages individually to avoid unintended model requests.

### 9.3 Common Notebook Variables

| Variable | Purpose |
|---|---|
| `DATASET` | Select one of the five dataset IDs. |
| `MODEL_ID` | Diagnosis model ID; `None` uses the API configuration or project default. |
| `API_CONFIG` | External API configuration JSON path. |
| `PROXY` | HTTP proxy; `None` uses the default setting, and an empty string clears the explicit proxy. |
| `INPUT_OVERRIDE` | Override the default dataset input path. |
| `OUTPUT_ROOT` | Notebook output root directory. |
| `GRAPH_OVERRIDE` | Override the graph path. |
| `CHECKPOINT_OVERRIDE` | Specify an existing checkpoint path. |
| `EVALUATION_OVERRIDE` | Override the evaluation JSON path. |
| `EXPERIMENT_TAG` | Experiment label; use different labels for different experiment conditions. |
| `END` | Number of records to diagnose; `None` uses the full selected scope. |
| `GRAPH_BUILD_LIMIT` | Graph conversion record limit, distinct from `END`. MAADBench builds a full graph index. |
| `REBUILD_GRAPH` | Rebuild an existing graph. Do not rebuild at the same path when a corresponding checkpoint exists. |
| `RUN_CONVERSION` | Enable graph conversion. |
| `RUN_DIAGNOSIS` | Enable diagnosis. |
| `RUN_EVALUATION` | Enable evaluation. |
| `CASCADE_ENABLED` | Enable cascade inspection. |
| `CASCADE_CALL_MODE` | `combined` / `separate`. |
| `SEPARATE_ORDER` | Separate-mode call order for all three benchmarks; see Section 4.1. |
| `TRAVERSAL_MODE` | `auto` / `bfs` / `dfs`. |
| `PROMPT_CONTEXT_POLICY` | Upstream prompt context selection policy. |
| `MAX_CAUSES` | Maximum number of candidate causes retained. |
| `MAX_PROMPT_UPSTREAM_CHARS` | Upstream context character budget. |
| `NO_GROUND_TRUTH` | When `True`, exclude the top-level graph `ground_truth` field from model-visible data. |
| `INCLUDE_EVAL_LABELS` | Include evaluation labels in model-visible data; disabled by default. |
| `EDGE_TYPES` | Select graph edge types; see Section 5 for the available subsets. |

### 9.4 Output Paths

The Notebook uses the following default structure:

```text
outputs/
└── notebook_independent/
    ├── graphs/
    │   └── <dataset>/
    │       └── <graph-configuration>.jsonl
    └── runs/
        └── <dataset>/
            └── <experiment-id>/
                ├── predictions.json
                └── evaluation.json
```

`predictions.json` is the diagnosis checkpoint, and `evaluation.json` is the evaluation report. MAADBench also produces official evaluation directories, manifests, and action-level detail files.

## 10. Checkpoints, Resumption, and Result Interpretation

- Resume from an existing checkpoint only when the model, graph, and diagnosis configuration match.
- Use a new checkpoint path when changing the model, graph edge set, cause state, diagnosis mode, separate order, or MAADBench split.
- Partial runs such as `END=2` are intended for workflow checks, not full-dataset results.
- Inspect `coverage`, `evaluation_status`, the evaluated sample count, and the split during evaluation.
- `SUCCESS`, `PARTIAL`, and `FAILED` are execution statuses, not measures of diagnostic accuracy.

## 11. Evaluation Metrics

| Dataset | Main metrics | Description |
|---|---|---|
| Who&When | Agent Accuracy, Step Accuracy | Accuracy of agent and step localization, respectively. |
| TRAIL | Category F1, Location Accuracy, Joint | Error category, location, and joint diagnosis performance, respectively. |
| MAADBench | F1, Accuracy, AUC, Balanced Accuracy | Action-level anomaly detection metrics, with additional direct error localization and threshold-policy comparisons. |

Evaluation units and metric definitions differ across datasets. Compare numerical results only under matching dataset, split, threshold policy, score source, and coverage conditions, where applicable.

## 12. Usage Notes

1. Start with a small `END` value to verify graph, checkpoint, and evaluation paths.
2. Regenerate the graph and use a separate checkpoint when changing the graph edge set.
3. For blind diagnosis, set `NO_GROUND_TRUTH=True` and keep `INCLUDE_EVAL_LABELS=False`.
4. `MAAD_SCOPE=all` diagnoses all graph records. Use `official_test` to diagnose only the official test split.
5. The `oracle` threshold uses evaluation labels to select the threshold and is suitable only as an oracle comparison. Explicitly report the threshold policy used for primary results.
6. Set `MAAD_OFFICIAL_ROOT` if the official MAADBench evaluation utilities cannot be located automatically.
7. After changing `EDGE_TYPES` or `SEPARATE_ORDER`, rerun Sections 4–5 of the Notebook. Do not manually reuse checkpoints from a different configuration.
