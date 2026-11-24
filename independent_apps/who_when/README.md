# Who&When: Original Separate + Structured Combined

This package uses one current Graph Converter and one diagnosis module. The
shared `--cascade-call-mode` option is the only strategy selector.

## Architecture

- One Graph Converter: `datasets/who_when/graph_converter.py` (current, fixed).
- One Diagnosis module: `datasets/who_when/diagnoser.py`.
  - `ReverseDependencyDFS`: Original ZIP implementation, preserved unchanged.
  - `CombinedReverseDependencyDFS`: additive structured Combined implementation.
- One Runner: `datasets/who_when/runner.py`.
- Combined support modules (schemas, evidence audit, causal state, traversal,
  DAG schedule, and Stage-2 payload) remain separate helpers, not extra
  Diagnoser implementations.

## Select the diagnosis strategy

Use the shared `--cascade-call-mode` option:

- `separate`: Original ZIP `ReverseDependencyDFS` baseline.
- `combined`: structured Combined diagnosis with joint Local/Cascade
  inspection and branch-aware cause tracking.

The Graph Converter is always the current implementation. There is no
Who&When-specific Graph Converter selector.

## Examples

Original baseline:

```bash
python main.py --dataset who_when_algorithm --cascade-call-mode separate --llm-model gpt-5-mini --end 2
```

Combined:

```bash
python main.py --dataset who_when_algorithm --cascade-call-mode combined --llm-model gpt-5-mini --max-causes 8 --end 2
```

Both strategies use the same current Graph file. Checkpoint and Evaluation
paths are isolated by diagnosis mode (`separate_current_graph` vs
`combined_current_graph`).

## Reproducibility notes

- Original diagnosis source is copied from the Original ZIP as the exact byte
  prefix of `diagnoser.py`; the Combined implementation is appended after it.
- The current Graph Converter is used for both strategies, so Diagnosis is the
  intended experimental variable.
- Original ZIP's old Graph Converter, Runner, transport, and evaluator are not
  restored. This is an Original-diagnosis baseline under the current
  independent_apps pipeline, not a byte-for-byte reproduction of the entire
  old project.
- `MAX_CAUSES`, prompt-context policy, and upstream prompt budget are consumed
  by Combined; the Original baseline keeps its own original behavior.
