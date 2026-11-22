"""Compatibility re-export; the canonical CLI is ``python main.py``."""

from typing import Optional, Sequence

from independent_apps.shared.cli import build_parser_for_dataset, main as shared_main
from independent_apps.shared.settings import load_api_config, settings_from_args as common_settings_from_args


def build_parser():
    return build_parser_for_dataset("who_when_algorithm", default_dataset="who_when_algorithm")


def settings_from_args(args, api_config=None):
    return common_settings_from_args(args, api_config)


def main(argv: Optional[Sequence[str]] = None):
    return shared_main(argv)


if __name__ == "__main__":
    main()
