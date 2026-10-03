"""Save run configuration and class-distribution figures."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from prognosais.IO import constants


def copy_config_to_information(config_path: Path, information_dir: Path) -> Path:
    """Keep a copy of the configuration actually used by a training process.

    K-fold jobs may start concurrently, so replace the shared copy atomically.
    """
    information_dir.mkdir(parents=True, exist_ok=True)
    destination = information_dir / "config.yml"
    fd, temporary_path = tempfile.mkstemp(
        prefix=".config_", suffix=".yml", dir=information_dir
    )
    try:
        with os.fdopen(fd, "wb") as output, config_path.open("rb") as source:
            shutil.copyfileobj(source, output)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination


def count_class_labels(
    records: Sequence[dict], classification_tasks: Sequence[str]
) -> dict[str, list[int]]:
    """Count known patient labels in the selected, pre-augmentation records."""
    counts = {
        task: [0] * constants.NUM_CLASSES_PER_LABEL[task]
        for task in classification_tasks
    }
    for record in records:
        for task in classification_tasks:
            label = [float(value) for value in record[f"label_{task}"]]
            if len(label) != len(counts[task]):
                raise ValueError(f"Unexpected label vector length for {task}.")
            if not any(label):
                continue  # Missing labels are encoded as all-zero vectors.
            class_index = max(range(len(label)), key=label.__getitem__)
            counts[task][class_index] += 1
    return counts


def plot_class_distributions(
    counts: Mapping[str, Sequence[int]],
    output_dir: Path,
    split: str,
    image_extension: str,
    fold_name: str | None = None,
) -> list[Path]:
    """Save one labeled-cohort class-count figure per configured task."""
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = f"_{fold_name}" if fold_name is not None else ""
    output_paths = []
    for task, task_counts in counts.items():
        labels = constants.CLASS_NAMES_PER_LABEL[task]
        if len(task_counts) != len(labels):
            raise ValueError(f"Unexpected class-count length for {task}.")
        stem = constants.CLASSIFICATION_TASK_DEFINITIONS[task][
            constants.FILENAME_STEM_KEY
        ]
        output_path = output_dir / (
            f"{stem}_{split}_distribution{fold_suffix}{image_extension}"
        )
        figure, axes = plt.subplots()
        try:
            axes.bar(labels, task_counts)
            axes.set_title(
                f"{split.capitalize()} dataset distribution for "
                f"{constants.CLASSIFICATION_DISPLAY_NAMES[task]}"
            )
            figure.tight_layout()
            figure.savefig(output_path)
        finally:
            plt.close(figure)
        output_paths.append(output_path)
    return output_paths
