"""Shared classification-task metadata for uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prognosais.IO import constants


@dataclass(frozen=True)
class ClassificationTaskSpec:
    """Describe one configured classification task.

    Args:
        key: Task key used by models and configuration files.
        display_name: Human-readable task name used in reports and figures.
        label_column: Ground-truth column in deterministic prediction tables.
        probabilities_column: Probability-vector column in deterministic
            prediction tables.
        prediction_column: Predicted-class column in deterministic prediction
            tables.
        output_name: Canonical task token used in UQ filenames.
        class_names: Human-readable names in encoded class order.
        class_values: Original label values in encoded class order.
        n_classes: Number of output classes.
    """

    key: str
    display_name: str
    label_column: str
    probabilities_column: str
    prediction_column: str
    output_name: str
    class_names: tuple[str, ...]
    class_values: tuple[int, ...]
    n_classes: int


def build_task_spec(task_key: str) -> ClassificationTaskSpec:
    """Build a task specification from centralized project constants.

    Args:
        task_key: Classification task key.

    Returns:
        Complete metadata specification for the requested task.

    Raises:
        KeyError: If the task is not supported by PrognosAIs.
        ValueError: If the centralized task definition is inconsistent.
    """

    try:
        definition = constants.CLASSIFICATION_TASK_DEFINITIONS[task_key]
    except KeyError as error:
        supported = ", ".join(constants.CLASSIFICATION_TASK_DEFINITIONS)
        raise KeyError(
            f"Unknown classification task '{task_key}'. Supported tasks: "
            f"{supported}."
        ) from error

    class_names = tuple(definition[constants.CLASS_NAMES_KEY])
    class_values = tuple(definition[constants.CLASS_VALUES_KEY])
    n_classes = int(definition[constants.NUM_CLASSES_KEY])
    if len(class_names) != n_classes or len(class_values) != n_classes:
        raise ValueError(
            f"Task '{task_key}' defines {n_classes} classes but provides "
            f"{len(class_names)} class names and {len(class_values)} class values."
        )

    return ClassificationTaskSpec(
        key=task_key,
        display_name=definition[constants.PRETTY_NAME_KEY],
        label_column=constants.CLASSIFICATION_LABEL_COLUMNS[task_key],
        probabilities_column=constants.CLASSIFICATION_PROBABILITY_COLUMNS[task_key],
        prediction_column=constants.CLASSIFICATION_PREDICTION_COLUMNS[task_key],
        output_name=definition[constants.FILENAME_STEM_KEY],
        class_names=class_names,
        class_values=class_values,
        n_classes=n_classes,
    )


def build_task_specs(task_keys: Iterable[str]) -> list[ClassificationTaskSpec]:
    """Build validated task specifications in the requested order.

    Args:
        task_keys: Explicit sequence of enabled task keys.

    Returns:
        Task specifications in the same order as task_keys.

    Raises:
        ValueError: If no tasks are supplied or a task is repeated.
        KeyError: If any task key is unsupported.
    """

    keys = [str(task_key) for task_key in task_keys]
    if not keys:
        raise ValueError("At least one classification task must be supplied.")
    duplicated = sorted({task for task in keys if keys.count(task) > 1})
    if duplicated:
        raise ValueError(
            "Classification tasks must be unique. Duplicated tasks: "
            + ", ".join(duplicated)
        )
    return [build_task_spec(task_key) for task_key in keys]


def classification_uq_filename(
    task: ClassificationTaskSpec,
    suffix: str,
    table_extension: str,
) -> str:
    """Return the canonical UQ table filename for one task and run.

    Args:
        task: Classification task metadata.
        suffix: Method-specific filename suffix.
        table_extension: Configured extension for generated tables.

    Returns:
        Canonical filename based on the public filename stem.
    """

    return f"UQ_{task.output_name}_{suffix}{table_extension}"
