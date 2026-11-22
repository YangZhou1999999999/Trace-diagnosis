"""TRAIL-only CLI options."""


def add_cli_arguments(parser, dataset):
    parser.add_argument("--enable-scoring", action="store_true")
    parser.add_argument("--use-llm-error-synthesis", action="store_true")
    parser.add_argument(
        "--prompt-context-policy",
        choices=["legacy", "ranked_budget", "prioritized_two_kinds"],
        default=None,
    )
    parser.add_argument(
        "--separate-order",
        choices=["cascade_then_local", "local_then_cascade"],
        default=None,
    )
    parser.add_argument("--edge-types", nargs="+", choices=["contains", "execution_flow", "data_flow"], default=None)
    parser.add_argument("--trail-cause-state-mode", choices=["H0", "H1"], default=None)
    parser.add_argument("--trail-schedule-scope", choices=["legacy", "all_nodes"], default=None)
    parser.add_argument("--trail-final-hypothesis-policy", choices=["gated", "retain_candidates"], default=None)
    parser.add_argument("--trail-max-completion-tokens", type=int, default=None)
    parser.add_argument("--trail-api-request-limit", type=int, default=None)
    parser.add_argument("--trail-request-token-budget", action="store_true")
    parser.add_argument("--trail-request-context-limit", type=int, default=None)
    parser.add_argument("--trail-request-hard-projected-limit", type=int, default=None)
    parser.add_argument("--trail-request-completion-reserve", type=int, default=None)
    parser.add_argument("--trail-request-safety-factor", type=float, default=None)
    parser.add_argument("--trail-request-tokenizer", default=None)
