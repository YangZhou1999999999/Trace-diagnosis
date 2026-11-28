"""Offline tests: no credentials, benchmark downloads, or LLM requests required."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from independent_apps.shared.cli import DATASET_IDS, build_parser_for_dataset
from independent_apps.shared.settings import load_api_config, settings_from_args


class CliTests(unittest.TestCase):
    def test_dataset_selection(self):
        self.assertEqual(len(DATASET_IDS), 5)
        for dataset in DATASET_IDS:
            with self.subTest(dataset=dataset):
                args = build_parser_for_dataset(dataset).parse_args(
                    ["--dataset", dataset, "--convert-only"]
                )
                self.assertEqual(args.dataset, dataset)
                self.assertTrue(args.convert_only)

    def test_separate_order_and_nonempty_edge_subsets(self):
        for dataset in DATASET_IDS:
            edges = ("contains" if dataset.startswith("trail_") else "sequential_flow",
                     "execution_flow", "data_flow")
            parser = build_parser_for_dataset(dataset)
            for order in ("local_then_cascade", "cascade_then_local"):
                for size in range(1, 4):
                    for subset in itertools.combinations(edges, size):
                        with self.subTest(dataset=dataset, order=order, edges=subset):
                            args = parser.parse_args([
                                "--dataset", dataset,
                                "--cascade-call-mode", "separate",
                                "--separate-order", order,
                                "--edge-types", *subset,
                            ])
                            settings = settings_from_args(args)
                            self.assertEqual(settings.separate_order, order)
                            self.assertEqual(settings.edge_types, list(subset))
                            self.assertEqual(settings.cascade_call_mode, "separate")

    def test_api_config_and_explicit_model(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps({
                "api_key": "test-placeholder",
                "base_url": "https://example.invalid/v1",
                "model": "example-model",
            }), encoding="utf-8")
            config = load_api_config(str(config_path))
            parser = build_parser_for_dataset("who_when_algorithm")
            args = parser.parse_args(["--dataset", "who_when_algorithm"])
            self.assertEqual(settings_from_args(args, config).llm_model, "example-model")
            args = parser.parse_args([
                "--dataset", "who_when_algorithm", "--llm-model", "override-model"
            ])
            self.assertEqual(settings_from_args(args, config).llm_model, "override-model")

    def test_label_visibility_switches(self):
        for dataset in DATASET_IDS:
            parser = build_parser_for_dataset(dataset)
            args = parser.parse_args([
                "--dataset", dataset, "--no-ground-truth", "--include-eval-labels"
            ])
            settings = settings_from_args(args)
            self.assertFalse(settings.include_ground_truth)
            self.assertTrue(settings.include_eval_labels)


if __name__ == "__main__":
    unittest.main()
