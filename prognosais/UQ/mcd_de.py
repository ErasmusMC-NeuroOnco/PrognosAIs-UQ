"""Monte Carlo deep ensemble aggregation from seed-level MCD outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from prognosais.IO import constants
from prognosais.IO.output import read_output_table, write_output_table
from prognosais.UQ.de import (
    EnsembleUQOutputs,
    build_ensemble_task_table,
    build_summary_table,
    normalize_tasks,
    validate_probability_matrix,
    validate_required_sequence,
    validate_required_value,
    validate_seed_folders_exist,
    write_ensemble_outputs,
)
from prognosais.UQ.task_utils import (
    ClassificationTaskSpec,
    classification_uq_filename,
)

SEGMENTATION_TASK_NAME = constants.SEGMENTATION_TASK_NAME


def dropout_token(dropout_rate: float) -> str:
    """Convert a dropout rate into the repository filename token.

    Args:
        dropout_rate: Dropout rate used for MC dropout.

    Returns:
        Filename token, for example 025 for 0.25.
    """

    return str(dropout_rate).replace(".", "")


def build_mcd_de_output_suffix(
    dropout_rate: float,
    n_models: int,
    n_samples: int,
) -> str:
    """Build the collision-safe classification-output suffix for MCD-DE.

    Args:
        dropout_rate: Dropout rate used for each member's MCD samples.
        n_models: Number of ensemble members.
        n_samples: Number of MCD samples per member.

    Returns:
        Suffix containing dropout, member count, and sample count.
    """

    return (
        f"{constants.UQ_METHOD_MCD_DE}_do{dropout_token(dropout_rate)}_"
        f"{n_models}m_{n_samples}s"
    )


def build_seed_mc_dropout_dir(
    ensemble_root: Path,
    seed: int,
    seed_folder_template: str,
    mc_folder: str | Path,
) -> Path:
    """Build the MC dropout output directory for one ensemble member.

    Args:
        ensemble_root: Root containing the seed folders.
        seed: Ensemble-member seed.
        seed_folder_template: Folder-name template containing the seed field.
        mc_folder: Relative or absolute MC dropout output folder.

    Returns:
        Path to the seed-level MC dropout output directory.
    """

    mc_folder = Path(mc_folder)
    if mc_folder.is_absolute():
        return mc_folder
    return Path(ensemble_root) / seed_folder_template.format(seed=seed) / mc_folder


def build_seed_task_uq_path(
    mc_dropout_dir: Path,
    task: ClassificationTaskSpec,
    dropout_rate: float,
    n_samples: int,
    table_extension: str,
) -> Path:
    """Build a seed-level task UQ table path.

    Args:
        mc_dropout_dir: Seed-level MC dropout output directory.
        task: Task specification.
        dropout_rate: Dropout rate used by the seed-level MCD run.
        n_samples: Number of MC dropout samples per seed.
        table_extension: Configured extension for generated tables.

    Returns:
        Path to the task UQ table.
    """

    token = dropout_token(dropout_rate)
    suffix = f"do{token}_{n_samples}s"
    return Path(mc_dropout_dir) / classification_uq_filename(
        task, suffix, table_extension
    )


def build_seed_segmentation_pickle_path(
    mc_dropout_dir: Path,
    dropout_rate: float,
    n_samples: int,
) -> Path:
    """Find the seed-level MCD segmentation pickle.

    MCD writes the full segmentation uncertainty dictionary to a generated
    subfolder and records its path in a run manifest. Use that path when a
    manifest is available, because previous MCD runs can leave older
    subfolders in the same output directory.

    Args:
        mc_dropout_dir: Seed-level MC dropout output directory.
        dropout_rate: Dropout rate used by the seed-level MCD run.
        n_samples: Number of MC dropout samples per seed.

    Returns:
        Path to the active segmentation pickle.

    Raises:
        FileNotFoundError: If no matching pickle exists.
        ValueError: If multiple matching pickles are found.
    """

    token = dropout_token(dropout_rate)
    filename = f"UQ_seg_do{token}_{n_samples}s.pkl.gz"
    manifest_path = Path(mc_dropout_dir) / f"UQ_manifest_do{token}_{n_samples}s.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("method") != constants.UQ_METHOD_MCD
            or manifest.get("dropout_rate") != dropout_rate
            or manifest.get("mc_samples") != n_samples
        ):
            raise ValueError(f"MCD manifest parameters do not match: {manifest_path}")
        recorded_path = manifest.get("output_paths", {}).get("seg_pkl")
        if not recorded_path:
            raise ValueError(
                f"MCD manifest has no segmentation pickle: {manifest_path}"
            )
        pickle_path = Path(recorded_path)
        if not pickle_path.is_absolute():
            pickle_path = Path(mc_dropout_dir) / pickle_path
        if pickle_path.name != filename or not pickle_path.is_file():
            raise FileNotFoundError(
                f"MCD manifest references a missing or mismatched segmentation "
                f"pickle: {pickle_path}"
            )
        return pickle_path

    matches = sorted(Path(mc_dropout_dir).rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Missing seed-level MCD segmentation pickle under {mc_dropout_dir}: "
            f"{filename}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple seed-level MCD segmentation pickles under "
            f"{mc_dropout_dir}: " + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def get_mean_predictions(segmentation_data: dict, path: Path) -> np.ndarray:
    """Retrieve the mean segmentation probabilities from an MCD pickle.

    Args:
        segmentation_data: Dictionary loaded from the seed-level MCD pickle.
        path: Source path, used in error messages.

    Returns:
        Array shaped as cases by classes by x by y by z.

    Raises:
        KeyError: If the required mean-prediction key is absent.
        ValueError: If the array does not have the expected rank.
    """

    if constants.SEGMENTATION_MEAN_PREDICTIONS_KEY not in segmentation_data:
        raise KeyError(
            f"{path} is missing required mean-prediction key "
            f"'{constants.SEGMENTATION_MEAN_PREDICTIONS_KEY}'."
        )
    mean_predictions = np.asarray(
        segmentation_data[constants.SEGMENTATION_MEAN_PREDICTIONS_KEY]
    )

    if mean_predictions.ndim != 5:
        raise ValueError(
            f"{path} mean predictions must be shaped as cases by classes by "
            f"x by y by z. Got shape {mean_predictions.shape}."
        )
    return mean_predictions


def validate_segmentation_probabilities(
    probabilities: np.ndarray,
    path: Path,
    tolerance: float,
) -> None:
    """Validate voxelwise segmentation probabilities.

    Args:
        probabilities: Probability array shaped as cases by classes by x by y
            by z.
        path: Source file path used in error messages.
        tolerance: Allowed absolute deviation from probability sums of one.

    Raises:
        ValueError: If probabilities are malformed.
    """

    if probabilities.shape[1] != constants.NUM_SEGMENTATION_CLASSES:
        raise ValueError(
            f"{path} has {probabilities.shape[1]} segmentation classes; "
            f"expected {constants.NUM_SEGMENTATION_CLASSES}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{path} contains non-finite segmentation probabilities.")
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1.0 + tolerance):
        raise ValueError(f"{path} contains probabilities outside [0, 1].")

    probability_sum = probabilities.sum(axis=1)
    max_deviation = float(np.max(np.abs(probability_sum - 1.0)))
    if max_deviation > tolerance:
        raise ValueError(
            f"{path} segmentation probabilities do not sum to one over classes. "
            f"Max deviation: {max_deviation:.6g}."
        )


def segmentation_entropy(
    probabilities: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compute voxelwise entropy for segmentation probabilities.

    Args:
        probabilities: Probability array shaped as cases by classes by x by y
            by z.
        epsilon: Numerical-stability constant for the logarithm.

    Returns:
        Entropy map shaped as cases by x by y by z.
    """

    return np.sum(probabilities * -np.log(probabilities + epsilon), axis=1)


def dice_scores_from_arrays(
    true_class: np.ndarray,
    predicted_class: np.ndarray,
) -> np.ndarray:
    """Compute binary Dice scores from true and predicted segmentation masks.

    Args:
        true_class: Ground-truth masks shaped as cases by x by y by z.
        predicted_class: Predicted masks shaped as cases by x by y by z.

    Returns:
        One Dice score per case.
    """

    true_binary = true_class.astype(bool)
    pred_binary = predicted_class.astype(bool)
    intersection = np.logical_and(true_binary, pred_binary).sum(axis=(1, 2, 3))
    volumes = true_binary.sum(axis=(1, 2, 3)) + pred_binary.sum(axis=(1, 2, 3))
    dice = np.ones_like(volumes, dtype=float)
    valid = volumes != 0
    dice[valid] = 2.0 * intersection[valid] / volumes[valid]
    return dice


def read_seed_task_uq_table(
    mc_dropout_dir: Path,
    seed: int,
    task: ClassificationTaskSpec,
    dropout_rate: float,
    n_samples: int,
    table_extension: str,
) -> pd.DataFrame:
    """Read a seed-level MC dropout UQ table for one task.

    Args:
        mc_dropout_dir: Seed-level MC dropout output directory.
        seed: Ensemble-member seed.
        task: Task specification.
        dropout_rate: Dropout rate used by the seed-level MCD run.
        n_samples: Number of MC dropout samples per seed.
        table_extension: Configured extension for generated tables.

    Returns:
        Seed-level UQ dataframe.

    Raises:
        FileNotFoundError: If the task UQ table is missing.
        ValueError: If the table lacks required columns or has duplicated cases.
    """

    uq_path = build_seed_task_uq_path(
        mc_dropout_dir=mc_dropout_dir,
        task=task,
        dropout_rate=dropout_rate,
        n_samples=n_samples,
        table_extension=table_extension,
    )
    if not uq_path.is_file():
        raise FileNotFoundError(
            f"Missing MC dropout UQ file for seed {seed} and task {task.key}: "
            f"{uq_path}"
        )

    table = read_output_table(uq_path, index_col=None)
    confidence_columns = [
        constants.CONFIDENCE_LABEL_COLUMN_TEMPLATE.format(class_idx=idx)
        for idx in range(task.n_classes)
    ]
    required_columns = {
        constants.CASE_COLUMN_NAME,
        constants.PREDICTED_CLASS_COLUMN_NAME,
        *confidence_columns,
    }
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"MC dropout UQ file {uq_path} is missing columns for {task.key}: "
            + ", ".join(sorted(missing_columns))
        )
    duplicated = (
        table.loc[
            table[constants.CASE_COLUMN_NAME].duplicated(), constants.CASE_COLUMN_NAME
        ]
        .astype(str)
        .tolist()
    )
    if duplicated:
        raise ValueError(
            f"MC dropout UQ file {uq_path} contains duplicated cases: "
            + ", ".join(duplicated[:10])
        )
    return table


def mcd_uq_table_to_member_frame(
    table: pd.DataFrame,
    seed: int,
    task: ClassificationTaskSpec,
    probability_sum_tolerance: float,
    inference_mode: str,
) -> pd.DataFrame:
    """Convert a seed-level MCD task table into a member frame.

    Args:
        table: Seed-level MCD task UQ table.
        seed: Ensemble-member seed.
        task: Task specification.
        probability_sum_tolerance: Allowed probability-sum deviation.
        inference_mode: Whether the source MCD output must contain labels.

    Returns:
        Member frame with case identifiers, true classes when labeled, and
        confidence vectors.

    Raises:
        ValueError: If confidence vectors are malformed.
    """

    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )
    labels_available = inference_mode == constants.INFERENCE_MODE_LABELED
    label_column_present = constants.TRUE_CLASS_COLUMN_NAME in table.columns
    if labels_available and not label_column_present:
        raise ValueError(
            f"Seed {seed} MC dropout table for {task.key} is configured as "
            "labeled but does not contain the true-class column."
        )
    if not labels_available and label_column_present:
        raise ValueError(
            f"Seed {seed} MC dropout table for {task.key} is configured as "
            "unlabeled but contains the true-class column."
        )

    confidence_columns = [
        constants.CONFIDENCE_LABEL_COLUMN_TEMPLATE.format(class_idx=idx)
        for idx in range(task.n_classes)
    ]
    probabilities = table[confidence_columns].to_numpy(dtype=float)
    validate_probability_matrix(
        probabilities,
        task=task,
        context=f"Seed {seed} MC dropout confidence vectors for {task.key}",
        tolerance=probability_sum_tolerance,
    )

    member_predicted_class = table[constants.PREDICTED_CLASS_COLUMN_NAME].to_numpy(
        dtype=int
    )
    if np.any(member_predicted_class < 0) or np.any(
        member_predicted_class >= task.n_classes
    ):
        raise ValueError(
            f"Seed {seed} MC dropout table for {task.key} contains predicted "
            f"classes outside the expected range 0 to {task.n_classes - 1}."
        )
    probability_predictions = np.argmax(probabilities, axis=1)
    if not np.array_equal(member_predicted_class, probability_predictions):
        raise ValueError(
            f"Seed {seed} MC dropout table for {task.key} has saved predicted "
            "classes that disagree with the confidence-vector argmax."
        )

    frame_data: dict[str, object] = {
        constants.CASE_COLUMN_NAME: table[constants.CASE_COLUMN_NAME]
        .astype(str)
        .to_numpy(),
        "probability_vector": [probabilities[idx] for idx in range(len(table))],
        "member_predicted_class": member_predicted_class,
        "Seed": int(seed),
    }
    if labels_available:
        true_class = table[constants.TRUE_CLASS_COLUMN_NAME].to_numpy(dtype=int)
        if np.any(true_class < 0) or np.any(true_class >= task.n_classes):
            raise ValueError(
                f"Seed {seed} MC dropout table for {task.key} contains true "
                f"classes outside the expected range 0 to {task.n_classes - 1}."
            )
        frame_data["true_class"] = true_class
    return pd.DataFrame(frame_data)


def read_mcd_member_frames(
    ensemble_root: Path,
    seeds: Iterable[int],
    task: ClassificationTaskSpec,
    dropout_rate: float,
    n_samples: int,
    seed_folder_template: str,
    mc_folder: str | Path,
    probability_sum_tolerance: float,
    table_extension: str,
    inference_mode: str,
) -> dict[int, pd.DataFrame]:
    """Read member frames for one task from seed-level MCD outputs.

    Args:
        ensemble_root: Root containing seed folders.
        seeds: Ensemble-member seeds.
        task: Task to process.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MCD samples per seed.
        seed_folder_template: Folder-name template containing the seed field.
        mc_folder: Relative or absolute MC dropout output folder.
        probability_sum_tolerance: Allowed probability-sum deviation.
        table_extension: Configured extension for generated tables.
        inference_mode: Whether the source MCD output must contain labels.

    Returns:
        Dictionary mapping seeds to member frames.
    """

    member_frames = {}
    for seed in seeds:
        mc_dropout_dir = build_seed_mc_dropout_dir(
            ensemble_root=ensemble_root,
            seed=int(seed),
            seed_folder_template=seed_folder_template,
            mc_folder=mc_folder,
        )
        if not mc_dropout_dir.exists():
            raise FileNotFoundError(
                f"Missing MC dropout directory for seed {seed}: {mc_dropout_dir}"
            )
        table = read_seed_task_uq_table(
            mc_dropout_dir=mc_dropout_dir,
            seed=int(seed),
            task=task,
            dropout_rate=dropout_rate,
            n_samples=n_samples,
            table_extension=table_extension,
        )
        member_frames[int(seed)] = mcd_uq_table_to_member_frame(
            table=table,
            seed=int(seed),
            task=task,
            probability_sum_tolerance=probability_sum_tolerance,
            inference_mode=inference_mode,
        )
    return member_frames


def build_mcd_de_segmentation_table(
    ensemble_root: Path,
    seeds: Iterable[int],
    dropout_rate: float,
    n_samples: int,
    seed_folder_template: str,
    mc_folder: str | Path,
    probability_sum_tolerance: float,
    entropy_epsilon: float,
    inference_mode: str,
) -> tuple[pd.DataFrame, dict]:
    """Build MCDE segmentation uncertainty from seed-level MCD pickles.

    Each seed-level pickle contributes the MCD mean segmentation probability
    map across its MC samples. MCDE then treats those seed-level mean maps as
    ensemble-member probability maps and computes predictive entropy, expected
    entropy, mutual information, and the final mean prediction.

    Args:
        ensemble_root: Root containing seed folders.
        seeds: Ensemble-member seeds.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MCD samples per seed.
        seed_folder_template: Folder-name template containing the seed field.
        mc_folder: Relative or absolute seed-level MC dropout output folder.
        probability_sum_tolerance: Allowed probability-sum deviation.
        entropy_epsilon: Numerical-stability constant used for entropy.
        inference_mode: Whether source MCD segmentation artifacts must contain
            ground-truth masks.

    Returns:
        Segmentation summary table and extended map dictionary.
    """

    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )
    labels_available = inference_mode == constants.INFERENCE_MODE_LABELED

    seed_list = [int(seed) for seed in seeds]
    reference_cases: list[str] | None = None
    reference_true_class: np.ndarray | None = None
    reference_has_true_class: bool | None = None
    probability_sum: np.ndarray | None = None
    expected_entropy_sum: np.ndarray | None = None
    source_paths: dict[int, str] = {}

    for seed in seed_list:
        mc_dropout_dir = build_seed_mc_dropout_dir(
            ensemble_root=ensemble_root,
            seed=seed,
            seed_folder_template=seed_folder_template,
            mc_folder=mc_folder,
        )
        pickle_path = build_seed_segmentation_pickle_path(
            mc_dropout_dir=mc_dropout_dir,
            dropout_rate=dropout_rate,
            n_samples=n_samples,
        )
        print(
            f"Loading seed {seed} MCD segmentation mean predictions from "
            f"{pickle_path}",
            flush=True,
        )
        seed_data = joblib.load(pickle_path)
        if not isinstance(seed_data, dict):
            raise TypeError(
                f"Seed {seed} segmentation artifact must contain a dictionary, "
                f"but {pickle_path} contains {type(seed_data).__name__}."
            )
        if constants.CASE_COLUMN_NAME not in seed_data:
            raise KeyError(f"{pickle_path} is missing {constants.CASE_COLUMN_NAME}.")
        seed_has_true_class = constants.TRUE_CLASS_COLUMN_NAME in seed_data
        if seed_has_true_class != labels_available:
            raise ValueError(
                f"Seed {seed} MCD segmentation artifact {pickle_path} is "
                f"configured as {inference_mode} but is "
                f"{'labeled' if seed_has_true_class else 'unlabeled'}."
            )
        seed_cases = [str(case) for case in seed_data[constants.CASE_COLUMN_NAME]]
        if len(seed_cases) != len(set(seed_cases)):
            raise ValueError(
                f"{pickle_path} contains duplicated segmentation case IDs."
            )
        seed_mean_predictions = get_mean_predictions(seed_data, pickle_path).astype(
            np.float32,
            copy=False,
        )
        validate_segmentation_probabilities(
            seed_mean_predictions,
            path=pickle_path,
            tolerance=probability_sum_tolerance,
        )
        if seed_mean_predictions.shape[0] != len(seed_cases):
            raise ValueError(
                f"{pickle_path} contains {len(seed_cases)} case IDs but "
                f"{seed_mean_predictions.shape[0]} mean-prediction tensors."
            )
        if seed_has_true_class:
            seed_true_class = np.asarray(seed_data[constants.TRUE_CLASS_COLUMN_NAME])
            if (
                seed_true_class.shape
                != seed_mean_predictions.shape[:1] + seed_mean_predictions.shape[2:]
            ):
                raise ValueError(
                    f"{pickle_path} true masks have shape {seed_true_class.shape}; "
                    "expected cases by x by y by z matching mean predictions."
                )

        if reference_cases is None:
            reference_cases = seed_cases
            reference_has_true_class = seed_has_true_class
            probability_sum = np.array(seed_mean_predictions, dtype=np.float32)
            expected_entropy_sum = segmentation_entropy(
                seed_mean_predictions, epsilon=entropy_epsilon
            ).astype(
                np.float32,
                copy=False,
            )
            if seed_has_true_class:
                reference_true_class = seed_true_class
        else:
            if seed_has_true_class != reference_has_true_class:
                raise ValueError(
                    "All MCD ensemble members must use the same segmentation "
                    f"label mode. Seed {seed} is "
                    f"{'labeled' if seed_has_true_class else 'unlabeled'}, while "
                    "the reference seed is "
                    f"{'labeled' if reference_has_true_class else 'unlabeled'}."
                )
            if set(seed_cases) != set(reference_cases):
                missing_from_seed = sorted(set(reference_cases) - set(seed_cases))
                extra_in_seed = sorted(set(seed_cases) - set(reference_cases))
                raise ValueError(
                    f"Seed {seed} segmentation cases do not match the reference "
                    f"seed. Missing from seed: {missing_from_seed[:10]}; "
                    f"extra in seed: {extra_in_seed[:10]}."
                )
            if seed_cases != reference_cases:
                order = [seed_cases.index(case) for case in reference_cases]
                seed_mean_predictions = seed_mean_predictions[order]
                if seed_has_true_class:
                    seed_data[constants.TRUE_CLASS_COLUMN_NAME] = np.asarray(
                        seed_data[constants.TRUE_CLASS_COLUMN_NAME]
                    )[order]

            if reference_true_class is not None and seed_has_true_class:
                seed_true_class = np.asarray(
                    seed_data[constants.TRUE_CLASS_COLUMN_NAME]
                )
                if not np.array_equal(reference_true_class, seed_true_class):
                    raise ValueError(
                        f"Seed {seed} segmentation true masks differ from the "
                        "reference seed."
                    )

            probability_sum += seed_mean_predictions
            expected_entropy_sum += segmentation_entropy(
                seed_mean_predictions, epsilon=entropy_epsilon
            ).astype(
                np.float32,
                copy=False,
            )

        source_paths[seed] = str(pickle_path)

    if (
        reference_cases is None
        or probability_sum is None
        or expected_entropy_sum is None
    ):
        raise ValueError("No seed-level segmentation pickles were processed.")

    mean_predictions = probability_sum / float(len(seed_list))
    predictive_entropy = segmentation_entropy(
        mean_predictions, epsilon=entropy_epsilon
    ).astype(
        np.float32,
        copy=False,
    )
    expected_entropy = expected_entropy_sum / float(len(seed_list))
    mutual_information = predictive_entropy - expected_entropy
    predicted_class = np.argmax(mean_predictions, axis=1).astype(np.uint8)

    table_data = {
        constants.CASE_COLUMN_NAME: reference_cases,
        "Predictive entropy": np.mean(predictive_entropy, axis=(1, 2, 3)),
        "Expected entropy": np.mean(expected_entropy, axis=(1, 2, 3)),
        "Mutual information": np.mean(mutual_information, axis=(1, 2, 3)),
        "Seeds": ",".join(str(seed) for seed in seed_list),
        "Number of ensemble members": len(seed_list),
        "MC samples per member": int(n_samples),
    }
    if reference_true_class is not None:
        table_data["Dice score"] = dice_scores_from_arrays(
            reference_true_class,
            predicted_class,
        )

    segmentation_table = pd.DataFrame.from_dict(table_data)
    segmentation_maps = {
        constants.CASE_COLUMN_NAME: reference_cases,
        constants.PREDICTED_CLASS_COLUMN_NAME: predicted_class,
        "Predictive entropy": predictive_entropy,
        "Expected entropy": expected_entropy,
        "Mutual information": mutual_information,
        "Mean predictions": mean_predictions,
        "Source MCD segmentation pickles": source_paths,
    }
    if reference_true_class is not None:
        segmentation_maps[constants.TRUE_CLASS_COLUMN_NAME] = reference_true_class

    return segmentation_table, segmentation_maps


def build_mcd_de_tables(
    ensemble_root: Path,
    seeds: Iterable[int],
    tasks: Iterable[ClassificationTaskSpec],
    dropout_rate: float,
    n_samples: int,
    seed_folder_template: str,
    mc_folder: str | Path,
    probability_sum_tolerance: float,
    table_extension: str,
    inference_mode: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build MCDE UQ tables from seed-level MCD outputs.

    Args:
        ensemble_root: Root containing seed folders.
        seeds: Ensemble-member seeds.
        tasks: Tasks to process.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MCD samples per seed.
        seed_folder_template: Folder-name template containing the seed field.
        mc_folder: Relative or absolute MC dropout output folder.
        probability_sum_tolerance: Allowed probability-sum deviation.
        table_extension: Configured extension for generated tables.
        inference_mode: Whether the source MCD output must contain labels.

    Returns:
        Per-task MCDE tables and one combined exclusion table.
    """

    seed_list = [int(seed) for seed in seeds]
    task_tables: dict[str, pd.DataFrame] = {}
    all_exclusions: list[pd.DataFrame] = []
    for task in tasks:
        member_frames = read_mcd_member_frames(
            ensemble_root=ensemble_root,
            seeds=seed_list,
            task=task,
            dropout_rate=dropout_rate,
            n_samples=n_samples,
            seed_folder_template=seed_folder_template,
            mc_folder=mc_folder,
            probability_sum_tolerance=probability_sum_tolerance,
            table_extension=table_extension,
            inference_mode=inference_mode,
        )
        task_table, alignment_exclusions = build_ensemble_task_table(
            member_frames=member_frames,
            task=task,
            method_name="MCDE",
            member_count_column="Number of ensemble members",
            member_count_value=len(seed_list),
            extra_constant_columns={"MC samples per member": int(n_samples)},
            probability_sum_tolerance=probability_sum_tolerance,
        )
        task_tables[task.key] = task_table
        if not alignment_exclusions.empty:
            all_exclusions.append(alignment_exclusions)

    exclusion_table = (
        pd.concat(all_exclusions, ignore_index=True)
        if all_exclusions
        else pd.DataFrame(
            columns=[
                constants.TASK_COLUMN_NAME,
                constants.CASE_COLUMN_NAME,
                "Seed",
                "Reason",
                "Detail",
            ]
        )
    )
    return task_tables, exclusion_table


def write_mcd_de_segmentation_outputs(
    segmentation_table: pd.DataFrame,
    segmentation_maps: dict,
    output_dir: Path,
    filename_suffix: str,
    table_extension: str,
) -> dict[str, Path]:
    """Write MCDE segmentation table and map pickle outputs.

    Args:
        segmentation_table: Per-case segmentation uncertainty table.
        segmentation_maps: Extended dictionary containing full uncertainty maps.
        output_dir: Directory where outputs are written.
        filename_suffix: Suffix used in output filenames.
        table_extension: Configured extension for generated tables.

    Returns:
        Dictionary with written segmentation output paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"UQ_seg_{filename_suffix}{table_extension}"
    pkl_path = output_dir / f"UQ_seg_{filename_suffix}.pkl.gz"
    labels_path = output_dir / "labels.pkl"
    write_output_table(segmentation_table, csv_path, index=False)
    joblib.dump(segmentation_maps, pkl_path, compress=True)
    with open(labels_path, "wb") as labels_file:
        pickle.dump(
            {"cases_id": list(segmentation_maps[constants.CASE_COLUMN_NAME])},
            labels_file,
        )
    return {"seg_csv": csv_path, "seg_pkl": pkl_path, "labels_pkl": labels_path}


def run_mcd_de(
    ensemble_root: Path,
    seeds: Iterable[int],
    output_dir: Path,
    tasks: Iterable[str],
    dropout_rate: float,
    n_samples: int,
    seed_folder_template: str,
    mc_folder: str | Path,
    probability_sum_tolerance: float,
    include_segmentation: bool,
    entropy_epsilon: float,
    table_extension: str,
    inference_mode: str,
) -> EnsembleUQOutputs:
    """Run Monte Carlo deep ensemble aggregation.

    Args:
        ensemble_root: Root containing independently trained seed folders.
        seeds: Ensemble-member seeds to include.
        output_dir: Destination directory.
        tasks: Explicit task keys to process.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MC dropout samples per seed.
        seed_folder_template: Folder-name template containing the seed field.
        mc_folder: Relative or absolute seed-level MC dropout output folder.
        probability_sum_tolerance: Allowed probability-sum deviation.
        include_segmentation: Whether to aggregate seed-level MCD segmentation
            mean probability maps.
        entropy_epsilon: Numerical-stability constant for entropy.
        table_extension: Configured extension for generated tables.
        inference_mode: Whether source MCD outputs are labeled or unlabeled.

    Returns:
        Object containing outputs, summaries, and written paths.
    """

    validate_required_value(ensemble_root, "ensemble_root")
    seed_values = validate_required_sequence(seeds, "seeds")
    validate_required_value(output_dir, "output_dir")
    task_values = validate_required_sequence(tasks, "tasks")
    validate_required_value(dropout_rate, "dropout_rate")
    validate_required_value(n_samples, "n_samples")
    validate_required_value(seed_folder_template, "seed_folder_template")
    validate_required_value(mc_folder, "mc_folder")
    validate_required_value(probability_sum_tolerance, "probability_sum_tolerance")
    validate_required_value(include_segmentation, "include_segmentation")
    validate_required_value(entropy_epsilon, "entropy_epsilon")
    validate_required_value(table_extension, "table_extension")
    validate_required_value(inference_mode, "inference_mode")
    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )
    if table_extension not in constants.SUPPORTED_TABLE_OUTPUT_FORMATS:
        raise ValueError(
            "Unsupported table extension: "
            f"{table_extension}. Supported values: "
            + ", ".join(constants.SUPPORTED_TABLE_OUTPUT_FORMATS)
        )

    ensemble_root = Path(ensemble_root)
    seed_list = [int(seed) for seed in seed_values]
    validate_seed_folders_exist(
        ensemble_root=ensemble_root,
        seeds=seed_list,
        seed_folder_template=seed_folder_template,
        method=constants.UQ_METHOD_MCD_DE,
    )
    task_specs = normalize_tasks(task_values)
    output_dir = Path(output_dir)

    task_tables, exclusion_table = build_mcd_de_tables(
        ensemble_root=ensemble_root,
        seeds=seed_list,
        tasks=task_specs,
        dropout_rate=dropout_rate,
        n_samples=n_samples,
        seed_folder_template=seed_folder_template,
        mc_folder=mc_folder,
        probability_sum_tolerance=probability_sum_tolerance,
        table_extension=table_extension,
        inference_mode=inference_mode,
    )
    summary_table = build_summary_table(
        task_tables=task_tables,
        exclusion_table=exclusion_table,
        tasks=task_specs,
    )
    dropout_rate_token = dropout_token(dropout_rate)
    suffix = build_mcd_de_output_suffix(
        dropout_rate=dropout_rate,
        n_models=len(seed_list),
        n_samples=n_samples,
    )
    output_paths = write_ensemble_outputs(
        task_tables=task_tables,
        exclusion_table=exclusion_table,
        summary_table=summary_table,
        tasks=task_specs,
        output_dir=output_dir,
        filename_suffix=suffix,
        method_prefix=constants.UQ_METHOD_MCD_DE,
        table_extension=table_extension,
    )
    if include_segmentation:
        segmentation_suffix = f"do{dropout_rate_token}_{len(seed_list)}m_{n_samples}s"
        segmentation_table, segmentation_maps = build_mcd_de_segmentation_table(
            ensemble_root=ensemble_root,
            seeds=seed_list,
            dropout_rate=dropout_rate,
            n_samples=n_samples,
            seed_folder_template=seed_folder_template,
            mc_folder=mc_folder,
            probability_sum_tolerance=probability_sum_tolerance,
            entropy_epsilon=entropy_epsilon,
            inference_mode=inference_mode,
        )
        task_tables["seg"] = segmentation_table
        output_paths.update(
            write_mcd_de_segmentation_outputs(
                segmentation_table=segmentation_table,
                segmentation_maps=segmentation_maps,
                output_dir=output_dir,
                filename_suffix=segmentation_suffix,
                table_extension=table_extension,
            )
        )

    manifest_path = (
        output_dir / f"UQ_{constants.UQ_METHOD_MCD_DE}_manifest_{suffix}.json"
    )
    manifest = {
        "schema_version": constants.UQ_SCHEMA_VERSION,
        "method": constants.UQ_METHOD_MCD_DE,
        "inference_mode": inference_mode,
        "tasks": [task.key for task in task_specs],
        "labels_available_by_task": {
            task.key: bool(constants.ERROR_COLUMN_NAME in task_tables[task.key].columns)
            for task in task_specs
        },
        "include_segmentation": bool(include_segmentation),
        "seeds": seed_list,
        "mc_samples_per_member": int(n_samples),
        "dropout_rate": float(dropout_rate),
        "output_format": {"tables": table_extension},
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_paths["manifest_json"] = manifest_path

    return EnsembleUQOutputs(
        task_tables=task_tables,
        exclusion_table=exclusion_table,
        summary_table=summary_table,
        output_paths=output_paths,
    )
