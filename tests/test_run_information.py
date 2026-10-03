"""Tests for run-level configuration and distribution outputs."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from prognosais.IO import constants
from prognosais.model.development.run_information import (
    copy_config_to_information,
    count_class_labels,
    plot_class_distributions,
)


class RunInformationTests(unittest.TestCase):
    def test_training_counts_skip_missing_labels(self):
        records = [
            {"label_idh": [1, 0], "label_grade": [0, 1, 0]},
            {"label_idh": [0, 1], "label_grade": [0, 0, 0]},
            {"label_idh": [0, 0], "label_grade": [0, 0, 1]},
        ]
        counts = count_class_labels(
            records, [constants.TASK_IDH, constants.TASK_GRADE]
        )
        self.assertEqual(counts[constants.TASK_IDH], [1, 1])
        self.assertEqual(counts[constants.TASK_GRADE], [0, 1, 1])

    def test_config_copy_is_saved_in_information(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "input.yml"
            config.write_text("example: original\n")
            information_dir = root / "information"
            destination = copy_config_to_information(config, information_dir)
            self.assertEqual(destination, information_dir / "config.yml")
            self.assertEqual(destination.read_text(), "example: original\n")

            config.write_text("example: updated\n")
            copy_config_to_information(config, information_dir)
            self.assertEqual(destination.read_text(), "example: updated\n")

    def test_training_and_test_plots_use_distinct_fold_paths(self):
        saved_paths = []
        pyplot = types.ModuleType("matplotlib.pyplot")

        class Axes:
            def bar(self, labels, values):
                self.labels = labels
                self.values = values

            def set_title(self, title):
                self.title = title

        class Figure:
            def tight_layout(self):
                pass

            def savefig(self, path):
                saved_paths.append(path)

        pyplot.subplots = lambda: (Figure(), Axes())
        pyplot.close = lambda figure: None
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = pyplot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                sys.modules,
                {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
            ):
                train_0 = plot_class_distributions(
                    {constants.TASK_IDH: [2, 3]},
                    root / "information",
                    "train",
                    ".svg",
                    fold_name="fold_0",
                )
                train_1 = plot_class_distributions(
                    {constants.TASK_IDH: [3, 2]},
                    root / "information",
                    "train",
                    ".svg",
                    fold_name="fold_1",
                )
                test_0 = plot_class_distributions(
                    {constants.TASK_IDH: [4, 1]},
                    root / "fold_0" / "results",
                    "test",
                    ".svg",
                )
                test_1 = plot_class_distributions(
                    {constants.TASK_IDH: [4, 1]},
                    root / "fold_1" / "results",
                    "test",
                    ".svg",
                )

        self.assertEqual(
            train_0[0].name, "IDH_train_distribution_fold_0.svg"
        )
        self.assertEqual(
            train_1[0].name, "IDH_train_distribution_fold_1.svg"
        )
        self.assertEqual(test_0[0].name, "IDH_test_distribution.svg")
        self.assertNotEqual(train_0, train_1)
        self.assertNotEqual(test_0, test_1)
        self.assertEqual(saved_paths, train_0 + train_1 + test_0 + test_1)


if __name__ == "__main__":
    unittest.main()
