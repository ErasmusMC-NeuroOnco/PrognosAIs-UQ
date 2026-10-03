"""Deep ensemble uncertainty aggregation from deterministic prediction tables."""

from __future__ import annotations

import ast
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

from prognosais.IO import constants
from prognosais.IO.output import read_output_table, write_output_table
from prognosais.UQ.metrics import ExpectedEntropy, PredictiveEntropy
from prognosais.UQ.task_utils import (
    ClassificationTaskSpec,
    build_task_specs,
    classification_uq_filename,
)


def dropout_token(dropout_rate: float) -> str:
    """Convert a dropout rate into the repository filename token.

    Args:
        dropout_rate: Dropout rate used by the experiment.

    Returns:
        Filename token, for example 025 for 0.25.
    """

    return str(dropout_rate).replace(".", "")


def validate_required_value(value: object, name: str) -> None:
    """Validate that a required parameter was explicitly provided.

    Args:
        value: Parameter value to validate.
        name: Human-readable parameter name used in the error message.

    Raises:
        ValueError: If the value is None or an empty string.
    """

    if value is None:
        raise ValueError(f"Missing required parameter: {name}.")
    if isinstance(value, str) and value.strip() == "":
        raise ValueError(f"Missing required parameter: {name}.")


def validate_required_sequence(
    values: Iterable[object] | None, name: str
) -> list[object]:
    """Validate that a required sequence parameter is not empty.

    Args:
        values: Sequence of values to validate.
        name: Human-readable parameter name used in the error message.

    Returns:
        list[object]: Materialized sequence values.

    Raises:
        ValueError: If the sequence is None or empty.
    """

    if values is None:
        raise ValueError(f"Missing required parameter: {name}.")
    materialized_values = list(values)
    if len(materialized_values) == 0:
        raise ValueError(f"Missing required parameter: {name}.")
    return materialized_values


def validate_seed_folders_exist(
    ensemble_root: Path,
    seeds: Iterable[int],
    seed_folder_template: str,
    method: str,
) -> None:
    """Validate that all configured ensemble seed folders exist.

    Args:
        ensemble_root: Root directory that should contain seed-specific run
            folders.
        seeds: Seed values that identify the ensemble members to aggregate.
        seed_folder_template: Folder-name template containing the seed field.
        method: UQ method name used in the error message.

    Raises:
        ValueError: If the template cannot format seed folders correctly.
        FileNotFoundError: If the ensemble root or any expected seed folder is
            missing.
    """

    if "{seed}" not in seed_folder_template:
        raise ValueError(
            f"seed_folder_template for UQ method '{method}' must contain the "
            f"{{seed}} field. Fill model.UQ.{method}.seed_folder_template "
            "with a pattern such as 'dropout_025_seed_{seed}'."
        )

    ensemble_root = Path(ensemble_root)
    seed_list = [int(seed) for seed in seeds]
    if not ensemble_root.exists():
        raise FileNotFoundError(
            f"Ensemble root directory does not exist for UQ method '{method}': "
            f"{ensemble_root}. Fill model.UQ.{method}.ensemble_root_dir "
            "with the directory that contains the seed folders."
        )
    if not ensemble_root.is_dir():
        raise NotADirectoryError(
            f"ensemble_root_dir for UQ method '{method}' is not a directory: "
            f"{ensemble_root}."
        )

    expected_paths = [
        ensemble_root / seed_folder_template.format(seed=seed) for seed in seed_list
    ]
    missing_paths = [path for path in expected_paths if not path.is_dir()]
    if missing_paths:
        expected_examples = ", ".join(str(path) for path in expected_paths[:5])
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Missing seed folders for UQ method '{method}'.\n"
            f"Configured ensemble_root_dir: {ensemble_root}\n"
            f"Configured seed_folder_template: {seed_folder_template}\n"
            f"Configured seeds: {seed_list}\n"
            f"Expected seed-folder examples: {expected_examples}\n"
            f"Missing folders:\n{missing_list}\n"
            f"Fill model.UQ.{method}.ensemble_root_dir with the root "
            f"containing the ensemble runs, model.UQ.{method}.seeds with "
            f"the trained seeds, and model.UQ.{method}.seed_folder_template "
            "with the folder naming pattern. Recommended pattern: "
            "'dropout_{dropout_rate}_seed_{seed}', for example "
            "'dropout_025_seed_{seed}'."
        )


@dataclass(frozen=True)
class EnsembleUQOutputs:
    """Paths and summary tables produced by one ensemble aggregation run.

    Args:
        task_tables: Per-task UQ tables keyed by task.
        exclusion_table: Table describing cases not used by the aggregation.
        summary_table: Per-task summary of included and excluded cases.
        output_paths: Paths written by the run, keyed by a descriptive name.
    """

    task_tables: dict[str, pd.DataFrame]
    exclusion_table: pd.DataFrame
    summary_table: pd.DataFrame
    output_paths: dict[str, Path]


def normalize_tasks(tasks: Iterable[str]) -> list[ClassificationTaskSpec]:
    """Convert task keys into task specifications.

    Args:
        tasks: Explicit classification task keys to process.

    Returns:
        List of task specifications in a stable order.

    Raises:
        ValueError: If no task is supplied or a task is repeated.
        KeyError: If an unknown task key is requested.
    """

    return build_task_specs(tasks)


def probability_vector_to_string(probabilities: np.ndarray) -> str:
    """Serialize a probability vector for table output.

    Args:
        probabilities: One-dimensional probability vector.

    Returns:
        JSON list string with compact separators.
    """

    return json.dumps(
        np.asarray(probabilities, dtype=float).tolist(), separators=(",", ":")
    )


def parse_probability_vector(
    value: object, case_id: str, column_name: str
) -> np.ndarray:
    """Parse a probability vector from a dataframe value.

    Args:
        value: Value read from a probability-vector column.
        case_id: Case identifier used in error messages.
        column_name: Source column name used in error messages.

    Returns:
        One-dimensional float probability vector.

    Raises:
        ValueError: If the value cannot be parsed into a numeric vector.
    """

    if isinstance(value, np.ndarray):
        vector = value
    elif isinstance(value, (list, tuple)):
        vector = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse probability vector for case {case_id} "
                f"from column {column_name}: {value}"
            ) from exc
        vector = np.asarray(parsed)
    else:
        raise ValueError(
            f"Unsupported probability vector type for case {case_id} "
            f"from column {column_name}: {type(value).__name__}"
        )

    try:
        vector = np.asarray(vector, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Probability vector for case {case_id} from column {column_name} "
            "contains non-numeric values."
        ) from exc

    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Probability vector for case {case_id} from column {column_name} "
            "is empty or non-finite."
        )
    return vector


def validate_probability_matrix(
    probabilities: np.ndarray,
    task: ClassificationTaskSpec,
    context: str,
    tolerance: float,
) -> None:
    """Validate probability-vector shape, values, and sums.

    Args:
        probabilities: Array shaped as cases by classes or members by cases by
            classes.
        task: Task specification defining the expected class count.
        context: Human-readable source description used in error messages.
        tolerance: Allowed absolute deviation from a probability sum of one.

    Raises:
        ValueError: If any probability vector is malformed.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape[-1] != task.n_classes:
        raise ValueError(
            f"{context}: expected {task.n_classes} classes for {task.key}, "
            f"got trailing shape {probabilities.shape[-1]}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{context}: probability matrix contains non-finite values.")
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1.0 + tolerance):
        raise ValueError(
            f"{context}: probability matrix contains values outside [0, 1]."
        )

    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0):
        max_deviation = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(
            f"{context}: probability vectors for {task.key} do not sum to one "
            f"within tolerance {tolerance}. Max deviation: {max_deviation:.6g}."
        )


def compute_ensemble_uncertainties(
    member_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute predictive entropy, expected entropy, and disagreement.

    The input is one probability tensor with shape members by cases by classes.
    Predictive entropy is computed from the mean probability vector. Expected
    entropy is the mean entropy of member-level probability vectors. The
    returned difference is analogous to mutual information, but for DE and MCDE
    it reflects variability across ensemble members rather than the within-model
    MC-dropout decomposition.

    Args:
        member_probabilities: Probability tensor shaped as M by N by C.

    Returns:
        Predictive entropy, expected entropy, and their difference.
    """

    predictive_entropy = PredictiveEntropy(
        class_axis=-1,
        epsilon=constants.ENTROPY_EPSILON,
    )
    expected_entropy = ExpectedEntropy(
        class_axis=-1,
        epsilon=constants.ENTROPY_EPSILON,
    )
    for member_probs in member_probabilities:
        predictive_entropy.update(member_probs)
        expected_entropy.update(member_probs)

    pe = predictive_entropy.compute()
    ee = expected_entropy.compute()
    return pe, ee, pe - ee


def build_summary_path(
    ensemble_root: Path,
    seed: int,
    seed_folder_template: str,
    summary_relative_path: str,
) -> Path:
    """Build one deterministic summary-table path for an ensemble seed.

    Args:
        ensemble_root: Root containing the seed folders.
        seed: Ensemble-member seed.
        seed_folder_template: Folder-name template containing the seed field.
        summary_relative_path: Relative path from a seed folder to a labeled or
            unlabeled prediction summary file.

    Returns:
        Full path to the requested summary file.
    """

    seed_folder = seed_folder_template.format(seed=seed)
    return Path(ensemble_root) / seed_folder / summary_relative_path


def read_seed_prediction_tables(
    ensemble_root: Path,
    seeds: Iterable[int],
    seed_folder_template: str,
    predictions_relative_path: str,
    inference_mode: str,
) -> tuple[dict[int, pd.DataFrame], dict[int, Path]]:
    """Read the configured prediction table for every ensemble member.

    The configuration selects one inference mode before aggregation begins.
    Therefore this function checks only the summary path associated with that
    mode. A summary from the opposite mode is irrelevant and is never used as
    a fallback.

    Args:
        ensemble_root: Root containing the seed folders.
        seeds: Ensemble-member seeds.
        seed_folder_template: Folder-name template containing the seed field.
        predictions_relative_path: Relative path from a seed folder to the
            summary file required for the configured inference mode.
        inference_mode: Configured inference mode, either labeled or unlabeled.

    Returns:
        Per-seed prediction tables and their resolved source paths.

    Raises:
        FileNotFoundError: If the selected summary file does not exist for a
            seed.
        ValueError: If inference_mode is invalid or a table has duplicated case
            identifiers.
    """

    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )

    tables: dict[int, pd.DataFrame] = {}
    source_paths: dict[int, Path] = {}
    for seed in (int(seed) for seed in seeds):
        summary_path = build_summary_path(
            ensemble_root=Path(ensemble_root),
            seed=seed,
            seed_folder_template=seed_folder_template,
            summary_relative_path=predictions_relative_path,
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Seed {seed} is configured for {inference_mode} inference, but "
                f"the required summary file does not exist: {summary_path}."
            )

        table = read_output_table(summary_path, index_col=None)
        if constants.CASE_COLUMN_NAME not in table.columns:
            raise ValueError(
                f"Prediction file {summary_path} is missing the "
                f"{constants.CASE_COLUMN_NAME} column."
            )
        duplicated = (
            table.loc[
                table[constants.CASE_COLUMN_NAME].duplicated(),
                constants.CASE_COLUMN_NAME,
            ]
            .astype(str)
            .tolist()
        )
        if duplicated:
            raise ValueError(
                f"Prediction file {summary_path} contains duplicated cases: "
                + ", ".join(duplicated[:10])
            )
        tables[seed] = table
        source_paths[seed] = summary_path
    return tables, source_paths


def build_probability_map_path(
    ensemble_root: Path,
    seed: int,
    case_id: str,
    seed_folder_template: str,
    probability_map_relative_path: str,
) -> Path:
    """Build the saved probability-map path for one case and seed.

    Args:
        ensemble_root: Root containing the seed folders.
        seed: Ensemble-member seed.
        case_id: Case identifier.
        seed_folder_template: Folder-name template containing the seed field.
        probability_map_relative_path: Relative path template containing the
            case field.

    Returns:
        Path to the case-level probability map.
    """

    seed_folder = seed_folder_template.format(seed=seed)
    relative_path = probability_map_relative_path.format(case=case_id)
    return Path(ensemble_root) / seed_folder / relative_path


def load_segmentation_probability_map(path: Path) -> np.ndarray:
    """Load a saved segmentation probability map.

    The inference exporter stores maps as x by y by z by classes. This function
    converts that layout to classes by x by y by z, matching the MCD and MCDE
    segmentation pickle convention.

    Args:
        path: Path to PROB_MAP.nii.gz.

    Returns:
        Probability array shaped as classes by x by y by z.

    Raises:
        FileNotFoundError: If the probability map is missing.
        ValueError: If the map has an unsupported shape.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation probability map: {path}")

    probability_map = np.asarray(nib.load(path).get_fdata(dtype=np.float32))
    if probability_map.ndim != 4:
        raise ValueError(
            f"Probability map must be 4D. Got shape {probability_map.shape}: {path}"
        )
    if probability_map.shape[-1] == constants.NUM_SEGMENTATION_CLASSES:
        return np.moveaxis(probability_map, -1, 0).astype(np.float32, copy=False)
    if probability_map.shape[0] == constants.NUM_SEGMENTATION_CLASSES:
        return probability_map.astype(np.float32, copy=False)
    raise ValueError(
        f"Could not identify class axis in probability map with shape "
        f"{probability_map.shape}: {path}"
    )


def validate_segmentation_probability_map(
    probabilities: np.ndarray,
    path: Path,
    tolerance: float,
) -> None:
    """Validate one segmentation probability map.

    Args:
        probabilities: Probability array shaped as classes by x by y by z.
        path: Source path used in error messages.
        tolerance: Allowed absolute deviation from probability sums of one.

    Raises:
        ValueError: If probabilities are malformed.
    """

    if probabilities.ndim != 4:
        raise ValueError(
            f"{path} probabilities must be shaped as classes by x by y by z. "
            f"Got shape {probabilities.shape}."
        )
    if probabilities.shape[0] != constants.NUM_SEGMENTATION_CLASSES:
        raise ValueError(
            f"{path} has {probabilities.shape[0]} segmentation classes; "
            f"expected {constants.NUM_SEGMENTATION_CLASSES}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{path} contains non-finite segmentation probabilities.")
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1.0 + tolerance):
        raise ValueError(f"{path} contains probabilities outside [0, 1].")

    probability_sum = probabilities.sum(axis=0)
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
            by z, or classes by x by y by z.
        epsilon: Numerical-stability constant for the logarithm.

    Returns:
        Entropy map. For case-level input the output is cases by x by y by z.
    """

    class_axis = 1 if probabilities.ndim == 5 else 0
    return np.sum(probabilities * -np.log(probabilities + epsilon), axis=class_axis)


def build_de_segmentation_table(
    ensemble_root: Path,
    seeds: Iterable[int],
    metrics_tables: dict[int, pd.DataFrame],
    seed_folder_template: str,
    probability_map_relative_path: str,
    probability_sum_tolerance: float,
    entropy_epsilon: float,
) -> tuple[pd.DataFrame, dict]:
    """Build DE segmentation uncertainty from saved probability maps.

    Each deterministic model contributes one voxelwise segmentation probability
    map per case. The function stacks those maps across seeds and computes
    predictive entropy, expected entropy, mutual information, and the final
    mean prediction.

    Args:
        ensemble_root: Root containing independently trained seed folders.
        seeds: Ensemble-member seeds.
        metrics_tables: Per-seed metrics tables used to define and validate
            case order.
        seed_folder_template: Folder-name template containing the seed field.
        probability_map_relative_path: Relative path template containing the
            case field.
        probability_sum_tolerance: Allowed probability-sum deviation.

    Returns:
        Segmentation summary table and extended map dictionary.
    """

    seed_list = [int(seed) for seed in seeds]
    reference_seed = seed_list[0]
    reference_cases = (
        metrics_tables[reference_seed][constants.CASE_COLUMN_NAME]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    reference_case_set = set(reference_cases)
    for seed in seed_list[1:]:
        seed_cases = set(
            metrics_tables[seed][constants.CASE_COLUMN_NAME].astype(str).tolist()
        )
        if seed_cases != reference_case_set:
            raise ValueError(
                f"Seed {seed} metrics cases do not match seed {reference_seed}. "
                f"Missing from seed: {sorted(reference_case_set - seed_cases)[:10]}; "
                f"extra in seed: {sorted(seed_cases - reference_case_set)[:10]}."
            )

    probability_sum: np.ndarray | None = None
    expected_entropy_sum: np.ndarray | None = None
    source_paths: dict[int, list[str]] = {}

    for seed in seed_list:
        print(f"Loading DE segmentation probability maps for seed {seed}", flush=True)
        seed_paths = []
        for case_idx, case_id in enumerate(reference_cases):
            probability_map_path = build_probability_map_path(
                ensemble_root=ensemble_root,
                seed=seed,
                case_id=case_id,
                seed_folder_template=seed_folder_template,
                probability_map_relative_path=probability_map_relative_path,
            )
            probability_map = load_segmentation_probability_map(probability_map_path)
            validate_segmentation_probability_map(
                probability_map,
                path=probability_map_path,
                tolerance=probability_sum_tolerance,
            )
            seed_paths.append(str(probability_map_path))

            if probability_sum is None:
                output_shape = (len(reference_cases),) + probability_map.shape
                entropy_shape = (len(reference_cases),) + probability_map.shape[1:]
                probability_sum = np.zeros(output_shape, dtype=np.float32)
                expected_entropy_sum = np.zeros(entropy_shape, dtype=np.float32)
            elif probability_map.shape != probability_sum.shape[1:]:
                raise ValueError(
                    f"{probability_map_path} has shape {probability_map.shape}, "
                    f"expected {probability_sum.shape[1:]}."
                )

            probability_sum[case_idx] += probability_map
            expected_entropy_sum[case_idx] += segmentation_entropy(
                probability_map, epsilon=entropy_epsilon
            ).astype(
                np.float32,
                copy=False,
            )
        source_paths[seed] = seed_paths

    if probability_sum is None or expected_entropy_sum is None:
        raise ValueError("No deterministic probability maps were processed.")

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

    segmentation_table = pd.DataFrame.from_dict(
        {
            constants.CASE_COLUMN_NAME: reference_cases,
            "Predictive entropy": np.mean(predictive_entropy, axis=(1, 2, 3)),
            "Expected entropy": np.mean(expected_entropy, axis=(1, 2, 3)),
            "Mutual information": np.mean(mutual_information, axis=(1, 2, 3)),
            "Seeds": ",".join(str(seed) for seed in seed_list),
            "Number of ensemble members": len(seed_list),
        }
    )
    segmentation_maps = {
        constants.CASE_COLUMN_NAME: reference_cases,
        constants.PREDICTED_CLASS_COLUMN_NAME: predicted_class,
        "Predictive entropy": predictive_entropy,
        "Expected entropy": expected_entropy,
        "Mutual information": mutual_information,
        "Mean predictions": mean_predictions,
        "Source probability maps": source_paths,
    }
    return segmentation_table, segmentation_maps


def write_de_segmentation_outputs(
    segmentation_table: pd.DataFrame,
    segmentation_maps: dict,
    output_dir: Path,
    filename_suffix: str,
    table_extension: str,
) -> dict[str, Path]:
    """Write DE segmentation table, full-map pickle, and labels pointer.

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


def metrics_table_to_member_frame(
    table: pd.DataFrame,
    seed: int,
    task: ClassificationTaskSpec,
    missing_label: int,
    probability_sum_tolerance: float,
    inference_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one task from a deterministic metrics table.

    Args:
        table: Metrics summary dataframe for one seed.
        seed: Seed represented by the dataframe.
        task: Task to extract.
        missing_label: Label value used for unavailable annotations.
        probability_sum_tolerance: Allowed probability-sum deviation.
        inference_mode: Whether the selected summary must contain labels.

    Returns:
        Valid member dataframe and exclusion dataframe.

    Raises:
        ValueError: If required columns are missing or probabilities are invalid.
    """

    if constants.CASE_COLUMN_NAME not in table.columns:
        raise ValueError(
            f"Seed {seed} metrics table is missing columns for {task.key}: "
            f"{constants.CASE_COLUMN_NAME}"
        )

    probability_column = constants.CLASSIFICATION_PROBABILITY_COLUMNS[task.key]
    if probability_column not in table.columns:
        raise ValueError(
            f"Seed {seed} metrics table is missing required column for {task.key}: "
            f"{probability_column}"
        )

    prediction_column = constants.CLASSIFICATION_PREDICTION_COLUMNS[task.key]
    if prediction_column not in table.columns:
        raise ValueError(
            f"Seed {seed} metrics table is missing required column for {task.key}: "
            f"{prediction_column}"
        )

    label_column = constants.CLASSIFICATION_LABEL_COLUMNS[task.key]
    labels_available = inference_mode == constants.INFERENCE_MODE_LABELED
    label_column_present = label_column in table.columns
    if labels_available and not label_column_present:
        raise ValueError(
            f"Seed {seed} is configured for labeled inference, but {task.key} "
            f"is missing its required label column: {label_column}."
        )
    if not labels_available and label_column_present:
        raise ValueError(
            f"Seed {seed} is configured for unlabeled inference, but {task.key} "
            f"contains a label column: {label_column}. Use labeled inference "
            "mode for this summary."
        )
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for _, row in table.iterrows():
        case_id = str(row[constants.CASE_COLUMN_NAME])
        true_class: int | None = None
        if labels_available:
            label = row[label_column]
            if pd.isna(label) or int(label) == missing_label:
                exclusions.append(
                    {
                        constants.TASK_COLUMN_NAME: task.display_name,
                        constants.CASE_COLUMN_NAME: case_id,
                        "Seed": int(seed),
                        "Reason": "missing_label",
                        "Detail": f"{label_column}={label}",
                    }
                )
                continue

            true_class = int(label)
            if true_class < 0 or true_class >= task.n_classes:
                raise ValueError(
                    f"Seed {seed}, case {case_id}, task {task.key}: true class "
                    f"{true_class} is outside the expected range 0 to "
                    f"{task.n_classes - 1}."
                )

        probability_vector = parse_probability_vector(
            row[probability_column],
            case_id=case_id,
            column_name=probability_column,
        )
        validate_probability_matrix(
            probability_vector,
            task=task,
            context=f"Seed {seed}, case {case_id}",
            tolerance=probability_sum_tolerance,
        )

        member_predicted_class = int(row[prediction_column])
        if member_predicted_class < 0 or member_predicted_class >= task.n_classes:
            raise ValueError(
                f"Seed {seed}, case {case_id}, task {task.key}: predicted class "
                f"{member_predicted_class} is outside the expected range 0 to "
                f"{task.n_classes - 1}."
            )
        probability_prediction = int(np.argmax(probability_vector))
        if member_predicted_class != probability_prediction:
            raise ValueError(
                f"Seed {seed}, case {case_id}, task {task.key}: saved predicted "
                f"class {member_predicted_class} disagrees with probability "
                f"argmax {probability_prediction}."
            )

        output_row: dict[str, object] = {
            constants.CASE_COLUMN_NAME: case_id,
            "probability_vector": probability_vector,
            "member_predicted_class": member_predicted_class,
            "Seed": int(seed),
        }
        if labels_available:
            output_row["true_class"] = true_class
        rows.append(output_row)

    member_columns = [
        constants.CASE_COLUMN_NAME,
        "probability_vector",
        "member_predicted_class",
        "Seed",
    ]
    if labels_available:
        member_columns.insert(1, "true_class")
    return pd.DataFrame(rows, columns=member_columns), pd.DataFrame(exclusions)


def build_ensemble_task_table(
    member_frames: dict[int, pd.DataFrame],
    task: ClassificationTaskSpec,
    method_name: str,
    member_count_column: str,
    member_count_value: int,
    extra_constant_columns: dict[str, object] | None,
    probability_sum_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one final UQ table from aligned member probability frames.

    Args:
        member_frames: Per-seed frames containing Case, true_class, and
            probability_vector columns.
        task: Task specification.
        method_name: Method label saved in the output table.
        member_count_column: Name of the column describing ensemble size.
        member_count_value: Ensemble size value to write.
        extra_constant_columns: Additional constant metadata columns.
        probability_sum_tolerance: Allowed probability-sum deviation.

    Returns:
        Final task dataframe and exclusions due to incomplete member coverage.

    Raises:
        ValueError: If no cases are retained or true labels are inconsistent.
    """

    if not member_frames:
        raise ValueError("At least one member frame is required.")

    seeds = list(member_frames)
    label_presence = {
        seed: "true_class" in frame.columns for seed, frame in member_frames.items()
    }
    if len(set(label_presence.values())) != 1:
        states = ", ".join(
            f"seed {seed}: {'labeled' if available else 'unlabeled'}"
            for seed, available in label_presence.items()
        )
        raise ValueError(
            f"All ensemble members must use the same label mode for {task.key}. "
            f"Observed {states}."
        )
    labels_available = next(iter(label_presence.values()))
    case_sets = {
        seed: set(frame[constants.CASE_COLUMN_NAME].astype(str))
        for seed, frame in member_frames.items()
    }
    case_intersection = set.intersection(*case_sets.values())
    case_union = set.union(*case_sets.values())
    reference_seed = seeds[0]
    reference_order = (
        member_frames[reference_seed][constants.CASE_COLUMN_NAME].astype(str).tolist()
    )
    retained_cases = [case for case in reference_order if case in case_intersection]

    exclusions = []
    for case in sorted(case_union - case_intersection):
        missing_from = [seed for seed in seeds if case not in case_sets[seed]]
        exclusions.append(
            {
                constants.TASK_COLUMN_NAME: task.display_name,
                constants.CASE_COLUMN_NAME: case,
                "Seed": ",".join(str(seed) for seed in missing_from),
                "Reason": "not_present_in_all_members",
                "Detail": "Case was valid for at least one member but not all members.",
            }
        )

    if not retained_cases:
        raise ValueError(f"No cases retained after member alignment for {task.key}.")

    indexed_frames = {
        seed: frame.set_index(constants.CASE_COLUMN_NAME, drop=False).loc[
            retained_cases
        ]
        for seed, frame in member_frames.items()
    }

    labels_by_seed: np.ndarray | None = None
    if labels_available:
        labels_by_seed = np.vstack(
            [indexed_frames[seed]["true_class"].to_numpy(dtype=int) for seed in seeds]
        )
        inconsistent = np.any(labels_by_seed != labels_by_seed[0], axis=0)
        if np.any(inconsistent):
            examples = [
                f"{case}: {labels_by_seed[:, idx].tolist()}"
                for idx, case in enumerate(retained_cases)
                if inconsistent[idx]
            ]
            raise ValueError(
                f"True labels are inconsistent across ensemble members for "
                f"{task.key}: " + "; ".join(examples[:10])
            )

    member_probabilities = np.stack(
        [
            np.vstack(indexed_frames[seed]["probability_vector"].to_numpy())
            for seed in seeds
        ],
        axis=0,
    )
    validate_probability_matrix(
        member_probabilities,
        task=task,
        context=f"{method_name} member probabilities for {task.key}",
        tolerance=probability_sum_tolerance,
    )

    mean_probabilities = np.mean(member_probabilities, axis=0)
    validate_probability_matrix(
        mean_probabilities,
        task=task,
        context=f"{method_name} mean probabilities for {task.key}",
        tolerance=probability_sum_tolerance,
    )
    predicted_class = np.argmax(mean_probabilities, axis=1)
    predictive_entropy, expected_entropy, disagreement = compute_ensemble_uncertainties(
        member_probabilities
    )

    output_data: dict[str, object] = {
        constants.CASE_COLUMN_NAME: retained_cases,
        constants.TASK_COLUMN_NAME: task.display_name,
        constants.METHOD_COLUMN_NAME: method_name,
        constants.PREDICTED_CLASS_COLUMN_NAME: predicted_class,
        constants.PREDICTIVE_ENTROPY_COLUMN_NAME: predictive_entropy,
        constants.EXPECTED_ENTROPY_COLUMN_NAME: expected_entropy,
        constants.MUTUAL_INFORMATION_COLUMN_NAME: disagreement,
        constants.MEAN_PROBABILITY_VECTOR_COLUMN_NAME: [
            probability_vector_to_string(vector) for vector in mean_probabilities
        ],
        "Seeds": ",".join(str(seed) for seed in seeds),
        member_count_column: int(member_count_value),
    }
    if labels_available:
        if labels_by_seed is None:
            raise RuntimeError("Internal error: labeled ensemble has no label array.")
        true_class = labels_by_seed[0]
        error = (predicted_class != true_class).astype(int)
        output_data[constants.TRUE_CLASS_COLUMN_NAME] = true_class
        output_data[constants.CORRECTLY_CLASSIFIED_COLUMN_NAME] = np.where(
            error == 0, "yes", "no"
        )
        output_data[constants.ERROR_COLUMN_NAME] = error
    output = pd.DataFrame(output_data)

    for class_idx in range(task.n_classes):
        output.insert(
            3 + class_idx,
            constants.CONFIDENCE_LABEL_COLUMN_TEMPLATE.format(class_idx=class_idx),
            mean_probabilities[:, class_idx],
        )

    for member_idx, seed in enumerate(seeds):
        output[f"Seed {seed} probability vector"] = [
            probability_vector_to_string(vector)
            for vector in member_probabilities[member_idx]
        ]
        if "member_predicted_class" in indexed_frames[seed].columns:
            output[f"Seed {seed} predicted class"] = indexed_frames[seed][
                "member_predicted_class"
            ].to_numpy(dtype=int)

    if extra_constant_columns:
        for column, value in extra_constant_columns.items():
            output[column] = value

    return output, pd.DataFrame(exclusions)


def build_de_tables(
    metrics_tables: dict[int, pd.DataFrame],
    tasks: Iterable[ClassificationTaskSpec],
    missing_label: int,
    probability_sum_tolerance: float,
    inference_mode: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build DE UQ tables from deterministic seed metrics.

    Args:
        metrics_tables: Per-seed metrics summary tables.
        tasks: Tasks to process.
        missing_label: Label value used for unavailable annotations.
        probability_sum_tolerance: Allowed probability-sum deviation.
        inference_mode: Whether member summaries are labeled or unlabeled.

    Returns:
        Per-task DE tables and one combined exclusion table.
    """

    task_tables: dict[str, pd.DataFrame] = {}
    all_exclusions: list[pd.DataFrame] = []
    seeds = list(metrics_tables)

    for task in tasks:
        member_frames = {}
        for seed, table in metrics_tables.items():
            frame, exclusions = metrics_table_to_member_frame(
                table=table,
                seed=seed,
                task=task,
                missing_label=missing_label,
                probability_sum_tolerance=probability_sum_tolerance,
                inference_mode=inference_mode,
            )
            member_frames[seed] = frame
            if not exclusions.empty:
                all_exclusions.append(exclusions)

        task_table, alignment_exclusions = build_ensemble_task_table(
            member_frames=member_frames,
            task=task,
            method_name="DE",
            member_count_column="Number of ensemble members",
            member_count_value=len(seeds),
            extra_constant_columns=None,
            probability_sum_tolerance=probability_sum_tolerance,
        )
        task_tables[task.key] = task_table
        if not alignment_exclusions.empty:
            all_exclusions.append(alignment_exclusions)

    exclusion_table = (
        pd.concat(all_exclusions, ignore_index=True)
        if all_exclusions
        else pd.DataFrame(
            columns=["Task", constants.CASE_COLUMN_NAME, "Seed", "Reason", "Detail"]
        )
    )
    return task_tables, exclusion_table


def build_summary_table(
    task_tables: dict[str, pd.DataFrame],
    exclusion_table: pd.DataFrame,
    tasks: Iterable[ClassificationTaskSpec],
) -> pd.DataFrame:
    """Summarize included and excluded cases for all tasks.

    Args:
        task_tables: Per-task UQ outputs.
        exclusion_table: Exclusion rows from the run.
        tasks: Tasks processed by the run.

    Returns:
        Summary dataframe with one row per task.
    """

    rows = []
    for task in tasks:
        task_output = task_tables[task.key]
        labels_available = constants.ERROR_COLUMN_NAME in task_output.columns
        task_exclusions = exclusion_table[
            exclusion_table[constants.TASK_COLUMN_NAME].astype(str) == task.display_name
        ]
        rows.append(
            {
                constants.TASK_COLUMN_NAME: task.display_name,
                "task_key": task.key,
                "n_included": int(len(task_output)),
                "labels_available": labels_available,
                "n_errors": (
                    int(task_output[constants.ERROR_COLUMN_NAME].sum())
                    if labels_available
                    else pd.NA
                ),
                "error_rate": (
                    float(task_output[constants.ERROR_COLUMN_NAME].mean())
                    if labels_available
                    else np.nan
                ),
                "n_exclusion_rows": int(len(task_exclusions)),
                "n_unique_excluded_cases": (
                    int(task_exclusions[constants.CASE_COLUMN_NAME].nunique())
                    if not task_exclusions.empty
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def write_ensemble_outputs(
    task_tables: dict[str, pd.DataFrame],
    exclusion_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    tasks: Iterable[ClassificationTaskSpec],
    output_dir: Path,
    filename_suffix: str,
    method_prefix: str,
    table_extension: str,
) -> dict[str, Path]:
    """Write per-task, combined, exclusion, and summary table outputs.

    Args:
        task_tables: Per-task UQ outputs.
        exclusion_table: Exclusion rows from the run.
        summary_table: Summary rows from the run.
        tasks: Classification tasks represented in task_tables.
        output_dir: Directory where outputs are written.
        filename_suffix: Suffix used in per-task UQ filenames.
        method_prefix: Lowercase method token used for combined filenames.
        table_extension: Configured extension for generated tables.

    Returns:
        Dictionary with all written paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    combined = []
    task_specs = list(tasks)
    for spec in task_specs:
        task_table = task_tables[spec.key]
        path = output_dir / classification_uq_filename(
            spec, filename_suffix, table_extension
        )
        write_output_table(task_table, path, index=False)
        output_paths[f"{spec.key}_csv"] = path
        combined.append(task_table)

    combined_table = pd.concat(combined, ignore_index=True)
    combined_path = output_dir / (
        f"UQ_{method_prefix}_all_tasks_{filename_suffix}{table_extension}"
    )
    write_output_table(combined_table, combined_path, index=False)
    output_paths["combined_csv"] = combined_path

    exclusion_path = output_dir / (
        f"UQ_{method_prefix}_exclusions_{filename_suffix}{table_extension}"
    )
    write_output_table(exclusion_table, exclusion_path, index=False)
    output_paths["exclusions_csv"] = exclusion_path

    summary_path = output_dir / (
        f"UQ_{method_prefix}_summary_{filename_suffix}{table_extension}"
    )
    write_output_table(summary_table, summary_path, index=False)
    output_paths["summary_csv"] = summary_path
    return output_paths


def run_de(
    ensemble_root: Path,
    seeds: Iterable[int],
    output_dir: Path,
    tasks: Iterable[str],
    dropout_rate: float,
    seed_folder_template: str,
    predictions_relative_path: str,
    inference_mode: str,
    probability_map_relative_path: str,
    missing_label: int,
    probability_sum_tolerance: float,
    include_segmentation: bool,
    entropy_epsilon: float,
    table_extension: str,
) -> EnsembleUQOutputs:
    """Run deterministic deep ensemble aggregation.

    Args:
        ensemble_root: Root containing independently trained seed folders.
        seeds: Ensemble-member seeds to include.
        output_dir: Destination directory.
        tasks: Explicit task keys to process.
        dropout_rate: Dropout rate token used in segmentation output filenames.
        seed_folder_template: Folder-name template containing the seed field.
        predictions_relative_path: Relative path from a seed folder to the
            prediction summary selected by inference_mode.
        inference_mode: Whether the selected summaries are labeled or unlabeled.
        probability_map_relative_path: Relative path template from a seed
            folder to each case probability map. Must contain the case field.
        missing_label: Label value used for unavailable annotations.
        probability_sum_tolerance: Allowed probability-sum deviation.
        include_segmentation: Whether to aggregate saved segmentation
            probability maps.
        entropy_epsilon: Numerical-stability constant for entropy.
        table_extension: Configured extension for generated tables.

    Returns:
        Object containing outputs, summaries, and written paths.
    """

    validate_required_value(ensemble_root, "ensemble_root")
    seed_values = validate_required_sequence(seeds, "seeds")
    validate_required_value(output_dir, "output_dir")
    task_values = validate_required_sequence(tasks, "tasks")
    validate_required_value(dropout_rate, "dropout_rate")
    validate_required_value(seed_folder_template, "seed_folder_template")
    validate_required_value(predictions_relative_path, "predictions_relative_path")
    validate_required_value(inference_mode, "inference_mode")
    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )
    validate_required_value(
        probability_map_relative_path, "probability_map_relative_path"
    )
    validate_required_value(missing_label, "missing_label")
    validate_required_value(probability_sum_tolerance, "probability_sum_tolerance")
    validate_required_value(include_segmentation, "include_segmentation")
    validate_required_value(entropy_epsilon, "entropy_epsilon")
    validate_required_value(table_extension, "table_extension")
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
        method=constants.UQ_METHOD_DE,
    )
    task_specs = normalize_tasks(task_values)
    output_dir = Path(output_dir)

    metrics_tables, summary_source_paths = read_seed_prediction_tables(
        ensemble_root=ensemble_root,
        seeds=seed_list,
        seed_folder_template=seed_folder_template,
        predictions_relative_path=predictions_relative_path,
        inference_mode=inference_mode,
    )
    task_tables, exclusion_table = build_de_tables(
        metrics_tables=metrics_tables,
        tasks=task_specs,
        missing_label=missing_label,
        probability_sum_tolerance=probability_sum_tolerance,
        inference_mode=inference_mode,
    )
    summary_table = build_summary_table(
        task_tables=task_tables,
        exclusion_table=exclusion_table,
        tasks=task_specs,
    )
    suffix = f"{constants.UQ_METHOD_DE}_{len(seed_list)}m"
    output_paths = write_ensemble_outputs(
        task_tables=task_tables,
        exclusion_table=exclusion_table,
        summary_table=summary_table,
        tasks=task_specs,
        output_dir=output_dir,
        filename_suffix=suffix,
        method_prefix=constants.UQ_METHOD_DE,
        table_extension=table_extension,
    )
    if include_segmentation:
        segmentation_suffix = f"do{dropout_token(dropout_rate)}_{len(seed_list)}m"
        segmentation_table, segmentation_maps = build_de_segmentation_table(
            ensemble_root=ensemble_root,
            seeds=seed_list,
            metrics_tables=metrics_tables,
            seed_folder_template=seed_folder_template,
            probability_map_relative_path=probability_map_relative_path,
            probability_sum_tolerance=probability_sum_tolerance,
            entropy_epsilon=entropy_epsilon,
        )
        task_tables["seg"] = segmentation_table
        output_paths.update(
            write_de_segmentation_outputs(
                segmentation_table=segmentation_table,
                segmentation_maps=segmentation_maps,
                output_dir=output_dir,
                filename_suffix=segmentation_suffix,
                table_extension=table_extension,
            )
        )

    manifest_path = output_dir / f"UQ_{constants.UQ_METHOD_DE}_manifest_{suffix}.json"
    manifest = {
        "schema_version": constants.UQ_SCHEMA_VERSION,
        "method": constants.UQ_METHOD_DE,
        "inference_mode": inference_mode,
        "tasks": [task.key for task in task_specs],
        "labels_available_by_task": {
            task.key: bool(constants.ERROR_COLUMN_NAME in task_tables[task.key].columns)
            for task in task_specs
        },
        "include_segmentation": bool(include_segmentation),
        "output_format": {"tables": table_extension},
        "seeds": seed_list,
        "source_summary_files": {
            str(seed): str(path) for seed, path in summary_source_paths.items()
        },
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
