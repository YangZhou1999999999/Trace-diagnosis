"""Who&When-only CLI options.

The shared --cascade-call-mode selects the diagnosis strategy:
separate -> Original ZIP ReverseDependencyDFS;
combined -> structured CombinedReverseDependencyDFS.
"""


def add_cli_arguments(parser, dataset):
    parser.add_argument("--separate-order", choices=["local_then_cascade", "cascade_then_local"], default=None)
    parser.add_argument("--edge-types", nargs="+", choices=["sequential_flow", "execution_flow", "data_flow"], default=None)
    parser.add_argument("--max-stage2-candidates", type=int, default=None)
    parser.add_argument("--include-full-analysis", action="store_true")
    parser.add_argument(
        "--prompt-context-policy",
        choices=["legacy", "ranked_budget"],
        default=None,
        help="Who&When prompt context construction policy.",
    )
