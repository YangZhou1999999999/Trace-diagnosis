"""Who&When-only CLI options."""


def add_cli_arguments(parser, dataset):
    parser.add_argument("--max-stage2-candidates", type=int, default=None)
    parser.add_argument("--include-full-analysis", action="store_true")
    parser.add_argument(
        "--prompt-context-policy",
        choices=["legacy", "ranked_budget"],
        default=None,
        help="Who&When prompt context construction policy.",
    )
