import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from prognosais.IO import constants
from prognosais.IO.output import read_output_table, write_output_table


def make_ball_structure(radius_voxels: int) -> np.ndarray:
    """Create a 3D ball-shaped structuring element.

    Args:
        radius_voxels: Radius of the ball in voxels.

    Returns:
        A boolean NumPy array containing the ball-shaped structuring element.
        Radius values smaller than one return a single-voxel structure.
    """
    radius_voxels = int(radius_voxels)
    if radius_voxels < 1:
        return np.ones((1, 1, 1), dtype=bool)

    x, y, z = np.ogrid[
        -radius_voxels : radius_voxels + 1,
        -radius_voxels : radius_voxels + 1,
        -radius_voxels : radius_voxels + 1,
    ]
    return x * x + y * y + z * z <= radius_voxels * radius_voxels


def mean_uncertainty_in_roi(uncertainty_map: np.ndarray, roi_mask: np.ndarray) -> float:
    """Compute mean uncertainty inside a binary region of interest.

    Args:
        uncertainty_map: Voxelwise uncertainty values.
        roi_mask: Binary region of interest with the same shape as
            uncertainty_map.

    Returns:
        Mean uncertainty inside the region of interest. Returns np.nan when the
        region is empty.
    """
    roi_mask = roi_mask.astype(bool)
    if np.count_nonzero(roi_mask) == 0:
        return np.nan
    return float(np.mean(uncertainty_map[roi_mask]))


def boundary_distance_weights(
    pred_tumor: np.ndarray, brain_mask: np.ndarray, sigma_voxels: float
) -> np.ndarray:
    """Compute Gaussian weights around the predicted tumor boundary.

    Args:
        pred_tumor: Binary predicted tumor mask.
        brain_mask: Binary brain mask restricting where weights are valid.
        sigma_voxels: Gaussian sigma in voxels.

    Returns:
        A floating-point weight map. Weights are zero outside the brain mask.
    """
    pred_tumor = pred_tumor.astype(bool)
    brain_mask = brain_mask.astype(bool)

    if sigma_voxels <= 0:
        raise ValueError("boundary_sigma_voxels must be greater than zero.")

    if not np.any(pred_tumor) or not np.any(brain_mask):
        return np.zeros_like(pred_tumor, dtype=float)

    boundary_structure = make_ball_structure(1)
    dilated = binary_dilation(pred_tumor, structure=boundary_structure)
    eroded = binary_erosion(pred_tumor, structure=boundary_structure)
    boundary = np.logical_xor(dilated, eroded)

    if not np.any(boundary):
        boundary = pred_tumor

    distance = distance_transform_edt(~boundary)
    weights = np.exp(-(distance**2) / (2 * sigma_voxels**2))
    weights[~brain_mask] = 0.0
    return weights


def weighted_mean_uncertainty(
    uncertainty_map: np.ndarray, weights: np.ndarray
) -> float:
    """Compute weighted mean uncertainty.

    Args:
        uncertainty_map: Voxelwise uncertainty values.
        weights: Voxelwise non-negative weights with the same shape as
            uncertainty_map.

    Returns:
        Weighted mean uncertainty. Returns np.nan when all weights are zero.
    """
    denominator = float(np.sum(weights))
    if denominator == 0:
        return np.nan
    return float(np.sum(uncertainty_map * weights) / denominator)


def compute_segmentation_uncertainty_aggregations(
    pred_tumor: np.ndarray,
    mask_brain: np.ndarray,
    predictive_entropy: np.ndarray,
    expected_entropy: np.ndarray,
    mutual_information: np.ndarray,
    dilation_radius_voxels: int,
    boundary_sigma_voxels: float,
) -> dict[str, np.ndarray]:
    """Compute case-level segmentation uncertainty aggregations.

    The tumor aggregation uses the predicted tumor mask. The dilated tumor
    aggregation dilates the predicted tumor and intersects it with the brain
    mask. The boundary-weighted aggregation weights brain voxels by distance to
    the predicted tumor boundary.

    Args:
        pred_tumor: Predicted tumor masks, shaped (N, X, Y, Z).
        mask_brain: Brain masks, shaped (N, X, Y, Z).
        predictive_entropy: Predictive entropy maps, shaped (N, X, Y, Z).
        expected_entropy: Expected entropy maps, shaped (N, X, Y, Z).
        mutual_information: Mutual information maps, shaped (N, X, Y, Z).
        dilation_radius_voxels: Predicted-tumor dilation radius in voxels.
        boundary_sigma_voxels: Boundary weighting Gaussian sigma in voxels.

    Returns:
        Dictionary mapping aggregation column names to one value per case.
    """
    expected_shape = pred_tumor.shape
    if pred_tumor.ndim != 4:
        raise ValueError(
            f"pred_tumor must have shape (N, X, Y, Z), not {expected_shape}."
        )
    named_arrays = {
        "mask_brain": mask_brain,
        constants.PREDICTIVE_ENTROPY_COLUMN_NAME: predictive_entropy,
        constants.EXPECTED_ENTROPY_COLUMN_NAME: expected_entropy,
        constants.MUTUAL_INFORMATION_COLUMN_NAME: mutual_information,
    }
    for name, array in named_arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {expected_shape}."
            )
    if dilation_radius_voxels < 0:
        raise ValueError("dilation_radius_voxels cannot be negative.")

    n_cases = pred_tumor.shape[0]
    dilation_structure = make_ball_structure(dilation_radius_voxels)
    maps = {
        "MPE": predictive_entropy,
        "MEE": expected_entropy,
        "MMI": mutual_information,
    }
    aggregations = {f"{prefix}-tumor": np.full(n_cases, np.nan) for prefix in maps}
    aggregations.update(
        {f"{prefix}-brain": np.full(n_cases, np.nan) for prefix in maps}
    )
    aggregations.update(
        {
            f"{prefix}-dilated-tumor-r{dilation_radius_voxels}": np.full(
                n_cases, np.nan
            )
            for prefix in maps
        }
    )
    aggregations.update(
        {
            f"{prefix}-boundary-weighted-sigma{boundary_sigma_voxels}": np.full(
                n_cases, np.nan
            )
            for prefix in maps
        }
    )

    for case_idx in range(n_cases):
        pred_mask = pred_tumor[case_idx].astype(bool)
        brain_mask = mask_brain[case_idx].astype(bool)
        dilated_mask = (
            binary_dilation(pred_mask, structure=dilation_structure) & brain_mask
        )
        boundary_weights = boundary_distance_weights(
            pred_mask, brain_mask, sigma_voxels=boundary_sigma_voxels
        )

        for prefix, uncertainty_maps in maps.items():
            uncertainty_map = uncertainty_maps[case_idx]
            aggregations[f"{prefix}-tumor"][case_idx] = mean_uncertainty_in_roi(
                uncertainty_map, pred_mask
            )
            aggregations[f"{prefix}-brain"][case_idx] = mean_uncertainty_in_roi(
                uncertainty_map, brain_mask
            )
            aggregations[f"{prefix}-dilated-tumor-r{dilation_radius_voxels}"][
                case_idx
            ] = mean_uncertainty_in_roi(uncertainty_map, dilated_mask)
            aggregations[f"{prefix}-boundary-weighted-sigma{boundary_sigma_voxels}"][
                case_idx
            ] = weighted_mean_uncertainty(uncertainty_map, boundary_weights)

    return aggregations


def dropout_token(dropout_rate: float) -> str:
    """Convert a dropout rate into the repository filename token.

    Args:
        dropout_rate: Dropout rate, for example 0.25.

    Returns:
        String token used in filenames, for example "025".
    """
    return str(dropout_rate).replace(".", "")


def build_run_token(
    method: str,
    dropout_rate: float,
    n_sample: int | None,
    num_models: int | None,
) -> str:
    """Build the filename token for one UQ run.

    Args:
        method: UQ method name, such as mcd, de, or mcd_de.
        dropout_rate: Dropout rate used by the run.
        n_sample: Number of MC samples, when relevant.
        num_models: Number of models in the ensemble, when relevant.

    Returns:
        Token used in filenames and temporary-directory markers.
    """
    if method not in constants.SUPPORTED_UQ_METHODS:
        raise ValueError(
            f"Unsupported UQ method '{method}'. Supported methods: "
            + ", ".join(constants.SUPPORTED_UQ_METHODS)
        )
    if method in constants.UQ_METHODS_REQUIRING_NUM_MODELS and num_models is None:
        raise ValueError(f"num_models is required for UQ method '{method}'.")
    if method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES and n_sample is None:
        raise ValueError(f"n_sample is required for UQ method '{method}'.")

    token = f"do{dropout_token(dropout_rate)}"
    if method in constants.UQ_METHODS_REQUIRING_NUM_MODELS:
        token += f"_{num_models}m"
    if method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES:
        token += f"_{n_sample}s"
    return token


def resolve_uq_method_dir(
    experiments_dir: str | Path, method: str, dropout_rate: float
) -> Path:
    """Resolve the UQ method directory for one method-specific experiment.

    Args:
        experiments_dir: Root directory containing dropout_* experiments.
        method: UQ method name, such as mcd, de, or mcd_de.
        dropout_rate: Dropout rate used to build the experiment directory name.

    Returns:
        Path to the saved UQ method directory for the requested run.
    """
    token = dropout_token(dropout_rate)
    if method not in constants.UQ_METHOD_RESULTS_DIR_NAMES:
        raise ValueError(
            f"Unsupported UQ method {method}. Supported methods are: "
            + ", ".join(constants.SUPPORTED_UQ_METHODS)
        )
    return (
        Path(experiments_dir)
        / f"{constants.DROPOUT_EXPERIMENT_DIR_PREFIX}{token}"
        / constants.RESULTS_DIR_NAME
        / constants.UQ_RESULTS_DIR_NAME
        / constants.UQ_METHOD_RESULTS_DIR_NAMES[method]
    )


def resolve_uq_seg_pickle(
    uq_method_dir: str | Path,
    method: str,
    dropout_rate: float,
    n_sample: int | None,
    num_models: int | None,
) -> Path:
    """Find the saved segmentation UQ pickle for one saved UQ run.

    Args:
        uq_method_dir: Path to the saved-UQ method directory.
        method: UQ method name, such as mcd, de, or mcd_de.
        dropout_rate: Dropout rate used in the saved filename.
        n_sample: Number of MC samples used in the saved filename.
        num_models: Number of models used in the saved filename.

    Returns:
        Path to the resolved segmentation UQ pickle.

    Raises:
        FileNotFoundError: If the temp-dir pointer or UQ pickle cannot be found.
    """
    token = build_run_token(
        method=method,
        dropout_rate=dropout_rate,
        n_sample=n_sample,
        num_models=num_models,
    )
    filename = f"{constants.SEGMENTATION_UQ_PREFIX}_{token}.pkl.gz"
    direct_pickle = Path(uq_method_dir) / filename
    if direct_pickle.is_file():
        return direct_pickle

    tmp_dir_file = Path(uq_method_dir) / f"tmp_dir_{token}.txt"
    if not tmp_dir_file.exists():
        raise FileNotFoundError(f"Missing temporary directory pointer: {tmp_dir_file}")

    pointer_value = tmp_dir_file.read_text(encoding="utf-8").strip()
    if not pointer_value:
        raise ValueError(f"Temporary directory pointer is empty: {tmp_dir_file}")
    temp_dir_original = Path(pointer_value)
    temp_dir_full_path = Path(uq_method_dir) / temp_dir_original.name

    candidates = [
        temp_dir_full_path / filename,
        temp_dir_original / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find saved segmentation UQ pickle. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def load_uq_seg_pickle(uq_seg_data_dir: str | Path) -> dict:
    """Load a saved segmentation UQ pickle.

    Args:
        uq_seg_data_dir: Path to a gzip-compressed pickle or joblib file.

    Returns:
        Deserialized segmentation UQ dictionary.
    """
    loaded_data = joblib.load(uq_seg_data_dir)
    if not isinstance(loaded_data, dict):
        raise TypeError(
            f"Segmentation UQ artifact must contain a dictionary, not "
            f"{type(loaded_data).__name__}: {uq_seg_data_dir}"
        )
    return loaded_data


def get_brain_masks_for_all_patients(
    input_dir: str | Path,
    patient_ids: list[str],
    brain_mask_relative_path: str,
) -> np.ndarray:
    """Load brain masks for cases in the saved UQ order.

    Args:
        input_dir: Dataset directory containing patient folders.
        patient_ids: Case identifiers from the segmentation UQ pickle.
        brain_mask_relative_path: Brain-mask path relative to each patient
            directory.

    Returns:
        NumPy array of brain masks with shape (N, X, Y, Z).
    """
    brain_masks = []

    for patient_id in patient_ids:
        mask_path = Path(input_dir) / patient_id / brain_mask_relative_path
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing brain mask for case {patient_id}: {mask_path}"
            )
        mask_sitk = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask_sitk), (2, 1, 0))
        brain_masks.append(mask_arr)

    return np.array(brain_masks)


def get_ground_truth_for_all_patients(
    input_dir: str | Path,
    patient_ids: list[str],
    ground_truth_mask_relative_path: str,
) -> np.ndarray:
    """Load tumor ground-truth masks for cases in the saved UQ order.

    Args:
        input_dir: Dataset directory containing patient folders.
        patient_ids: Case identifiers from the segmentation UQ pickle.
        ground_truth_mask_relative_path: Ground-truth mask path relative to each
            patient directory.

    Returns:
        NumPy array of tumor masks with shape (N, X, Y, Z).
    """
    tumor_masks = []

    for patient_id in patient_ids:
        mask_path = Path(input_dir) / patient_id / ground_truth_mask_relative_path
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing ground-truth mask for case {patient_id}: {mask_path}"
            )
        mask_sitk = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask_sitk), (2, 1, 0))
        tumor_masks.append(mask_arr)

    return np.array(tumor_masks)


def export_segmentation_masks(
    input_dir: str | Path,
    y_pred: np.ndarray,
    reference_mask_relative_path: str,
    out_dir: str | Path,
    patient_ids: list[str],
    y_true: np.ndarray | None = None,
) -> None:
    """Export predicted masks and optional ground-truth masks as NIfTI files.

    Args:
        input_dir: Dataset directory containing patient folders.
        y_pred: Predicted masks, shaped (N, X, Y, Z).
        reference_mask_relative_path: Reference-image path relative to each
            patient directory.
        out_dir: Directory where masks are written.
        patient_ids: Explicit case order for mask export.
        y_true: Optional ground-truth masks, shaped (N, X, Y, Z).

    Raises:
        ValueError: If the provided arrays are not aligned to patient_ids.
    """
    if len(patient_ids) != y_pred.shape[0]:
        raise ValueError(
            "Predicted-mask export inputs are not aligned: "
            f"{len(patient_ids)} case IDs and {y_pred.shape[0]} predictions."
        )
    if y_true is not None and len(patient_ids) != y_true.shape[0]:
        raise ValueError(
            "Ground-truth export inputs are not aligned: "
            f"{len(patient_ids)} case IDs and {y_true.shape[0]} ground-truth masks."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, patient_id in enumerate(patient_ids):
        ref_img = sitk.ReadImage(
            str(Path(input_dir) / patient_id / reference_mask_relative_path)
        )
        pred_img = sitk.GetImageFromArray(
            np.swapaxes(y_pred[i], 0, -1).astype(np.uint8)
        )
        pred_img.CopyInformation(ref_img)
        sitk.WriteImage(
            pred_img,
            str(out_dir / f"{constants.SEGMENTATION_PRED_PREFIX}{patient_id}.nii.gz"),
        )

        if y_true is not None:
            gt_img = sitk.GetImageFromArray(
                np.swapaxes(y_true[i], 0, -1).astype(np.uint8)
            )
            gt_img.CopyInformation(ref_img)
            sitk.WriteImage(
                gt_img,
                str(out_dir / f"{constants.SEGMENTATION_GT_PREFIX}{patient_id}.nii.gz"),
            )


def compute_dice_per_patient(
    input_dir: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_mask_relative_path: str,
    out_dir: str | Path | None,
    patient_ids: list[str],
) -> np.ndarray:
    """Compute Dice coefficient per patient.

    Args:
        input_dir: Dataset directory containing patient folders.
        y_true: Ground-truth masks, shaped (N, X, Y, Z).
        y_pred: Predicted masks, shaped (N, X, Y, Z).
        reference_mask_relative_path: Reference-image path relative to each
            patient directory.
        out_dir: Optional directory for exported predicted masks and GT masks
            when available.
        patient_ids: Explicit case order for the Dice loop.

    Returns:
        One Dice coefficient per patient.
    """
    if len(patient_ids) != y_true.shape[0] or len(patient_ids) != y_pred.shape[0]:
        raise ValueError(
            "Dice inputs are not aligned: "
            f"{len(patient_ids)} case IDs, "
            f"{y_true.shape[0]} ground-truth masks, "
            f"{y_pred.shape[0]} predictions."
        )

    dice_scores = np.zeros(y_true.shape[0])

    for i, patient_id in enumerate(patient_ids):
        ref_img = sitk.ReadImage(
            str(Path(input_dir) / patient_id / reference_mask_relative_path)
        )
        gt_img = sitk.GetImageFromArray(np.swapaxes(y_true[i], 0, -1).astype(np.uint8))
        pred_img = sitk.GetImageFromArray(
            np.swapaxes(y_pred[i], 0, -1).astype(np.uint8)
        )

        gt_img.CopyInformation(ref_img)
        pred_img.CopyInformation(ref_img)

        dice_filter = sitk.LabelOverlapMeasuresImageFilter()
        dice_filter.Execute(gt_img, pred_img)
        dice_scores[i] = dice_filter.GetDiceCoefficient()

    if out_dir is not None:
        export_segmentation_masks(
            input_dir=input_dir,
            y_pred=y_pred,
            reference_mask_relative_path=reference_mask_relative_path,
            out_dir=out_dir,
            patient_ids=patient_ids,
            y_true=y_true,
        )

    return dice_scores


def export_uncertainty_maps(
    input_dir: str | Path,
    cases_id: list[str],
    uncertainty_maps: np.ndarray,
    output_dir: str | Path,
    map_type: str,
    reference_mask_relative_path: str,
) -> None:
    """Export uncertainty maps as NIfTI files.

    Args:
        input_dir: Dataset directory containing patient folders.
        cases_id: Case identifiers in the same order as uncertainty_maps.
        uncertainty_maps: Voxelwise uncertainty maps, shaped (N, X, Y, Z).
        output_dir: Directory where the uncertainty maps will be written.
        map_type: Prefix for saved map filenames, for example PUM.
        reference_mask_relative_path: Reference-image path relative to each
            patient directory.
    """
    uq_maps_dir = Path(output_dir)
    uq_maps_dir.mkdir(parents=True, exist_ok=True)

    for patient_idx, case_id in enumerate(cases_id):
        uncertainty_map = uncertainty_maps[patient_idx]
        uncertainty_map_sitk = sitk.GetImageFromArray(
            np.swapaxes(uncertainty_map, 0, -1)
        )
        ref_img = sitk.ReadImage(
            str(Path(input_dir) / case_id / reference_mask_relative_path)
        )
        uncertainty_map_sitk.CopyInformation(ref_img)
        sitk.WriteImage(
            uncertainty_map_sitk,
            str(uq_maps_dir / f"{map_type}_{case_id}.nii.gz"),
        )


def validate_segmentation_uq_data(
    uq_seg_data: dict,
    source_path: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and unpack one saved segmentation UQ dictionary.

    Args:
        uq_seg_data: Deserialized UQ segmentation dictionary.
        source_path: Source path used in validation messages.

    Returns:
        Case identifiers, predicted masks, predictive entropy, expected
        entropy, and mutual-information maps in the saved case order.

    Raises:
        KeyError: If a required field is absent.
        ValueError: If case identifiers or array shapes are inconsistent.
    """
    required_keys = (
        constants.SEGMENTATION_CASE_COLUMN,
        constants.SEGMENTATION_PREDICTED_CLASS_COLUMN,
        constants.PREDICTIVE_ENTROPY_COLUMN_NAME,
        constants.EXPECTED_ENTROPY_COLUMN_NAME,
        constants.MUTUAL_INFORMATION_COLUMN_NAME,
    )
    missing_keys = [key for key in required_keys if key not in uq_seg_data]
    if missing_keys:
        raise KeyError(
            f"Segmentation UQ file {source_path} is missing required fields: "
            + ", ".join(missing_keys)
        )

    cases_id = [str(case_id) for case_id in uq_seg_data[required_keys[0]]]
    if not cases_id:
        raise ValueError(f"Segmentation UQ file {source_path} contains no cases.")
    duplicated = sorted(
        {case_id for case_id in cases_id if cases_id.count(case_id) > 1}
    )
    if duplicated:
        raise ValueError(
            f"Segmentation UQ file {source_path} contains duplicated cases: "
            + ", ".join(duplicated[:10])
        )

    arrays = tuple(np.asarray(uq_seg_data[key]) for key in required_keys[1:])
    expected_shape = arrays[0].shape
    if len(expected_shape) != 4:
        raise ValueError(
            f"Predicted segmentation in {source_path} must have shape "
            f"(N, X, Y, Z), not {expected_shape}."
        )
    if expected_shape[0] != len(cases_id):
        raise ValueError(
            f"Segmentation UQ file {source_path} contains {len(cases_id)} case "
            f"IDs but {expected_shape[0]} predicted masks."
        )
    if not np.all(np.isfinite(arrays[0])):
        raise ValueError(
            f"Predicted segmentation in {source_path} contains non-finite values."
        )
    rounded_predictions = np.rint(arrays[0])
    if not np.array_equal(arrays[0], rounded_predictions):
        raise ValueError(
            f"Predicted segmentation in {source_path} contains non-integer "
            "class values."
        )
    invalid_classes = np.setdiff1d(
        np.unique(rounded_predictions),
        np.arange(constants.NUM_SEGMENTATION_CLASSES),
    )
    if invalid_classes.size:
        raise ValueError(
            f"Predicted segmentation in {source_path} contains unsupported "
            f"class values: {invalid_classes.tolist()}."
        )
    for key, array in zip(required_keys[2:], arrays[1:]):
        if array.shape != expected_shape:
            raise ValueError(
                f"Field {key} in {source_path} has shape {array.shape}; expected "
                f"{expected_shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"Field {key} in {source_path} contains non-finite values."
            )
    return (cases_id, *arrays)


def align_segmentation_table_to_cases(
    table: pd.DataFrame,
    cases_id: list[str],
    source_path: str | Path,
) -> pd.DataFrame:
    """Validate and align a segmentation summary table to UQ-map case order.

    Args:
        table: Segmentation summary loaded from a configured table format.
        cases_id: Target case order from the full-map UQ pickle.
        source_path: Table path used in validation messages.

    Returns:
        Copy of the table reordered to match cases_id.

    Raises:
        ValueError: If the case column is absent, duplicated, or does not match
            the UQ pickle exactly.
    """
    if constants.CASE_COLUMN_NAME not in table.columns:
        raise ValueError(
            f"Source UQ table {source_path} has no "
            f"{constants.CASE_COLUMN_NAME} column."
        )
    table = table.copy()
    table[constants.CASE_COLUMN_NAME] = table[constants.CASE_COLUMN_NAME].astype(str)
    duplicated = table.loc[
        table[constants.CASE_COLUMN_NAME].duplicated(), constants.CASE_COLUMN_NAME
    ].tolist()
    if duplicated:
        raise ValueError(
            f"Source UQ table {source_path} contains duplicated cases: "
            + ", ".join(duplicated[:10])
        )
    table_cases = set(table[constants.CASE_COLUMN_NAME])
    target_cases = set(cases_id)
    if table_cases != target_cases:
        raise ValueError(
            f"Source UQ table {source_path} does not match its map pickle. "
            f"Missing from table: {sorted(target_cases - table_cases)[:10]}; "
            f"extra in table: {sorted(table_cases - target_cases)[:10]}."
        )
    return table.set_index(constants.CASE_COLUMN_NAME).loc[cases_id].reset_index()


def compute_muq_for_setting(
    input_dir: str | Path,
    uq_method_dir: str | Path,
    method: str,
    dropout_rate: float,
    n_sample: int | None,
    num_models: int | None,
    ground_truth_mask_relative_path: str | None,
    brain_mask_relative_path: str,
    export_maps: bool,
    export_masks: bool,
    dilation_radius_voxels: int,
    boundary_sigma_voxels: float,
    table_extension: str,
    inference_mode: str,
) -> Path:
    """Compute a mask-based segmentation uncertainty table for one UQ setting.

    Args:
        input_dir: Dataset directory containing patient folders.
        uq_method_dir: Path to the saved-UQ method directory.
        method: UQ method name, such as mcd, de, or mcd_de.
        dropout_rate: Dropout rate used in the saved run filenames.
        n_sample: Number of samples used in the saved run filenames.
        num_models: Number of models used in the saved run filenames.
        ground_truth_mask_relative_path: Ground-truth mask path relative to each
            patient directory, or None when ground truth is unavailable.
        brain_mask_relative_path: Brain-mask path relative to each patient
            directory.
        export_maps: Whether to export PUM/AUM/EUM NIfTI maps.
        export_masks: Whether to export predicted masks, plus GT masks when
            ground truth is available.
        dilation_radius_voxels: Predicted-tumor dilation radius in voxels.
        boundary_sigma_voxels: Boundary weighting Gaussian sigma in voxels.
        table_extension: Configured extension for source and generated tables.
        inference_mode: Whether ground-truth masks are required or forbidden.

    Returns:
        Path to the written MUQ table.
    """
    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )
    ground_truth_available = bool(
        ground_truth_mask_relative_path and ground_truth_mask_relative_path.strip()
    )
    labels_available = inference_mode == constants.INFERENCE_MODE_LABELED
    if labels_available and not ground_truth_available:
        raise ValueError(
            "Labeled UQ aggregation requires --ground-truth-mask-relative-path."
        )
    if not labels_available and ground_truth_available:
        raise ValueError(
            "Unlabeled UQ aggregation must not receive "
            "--ground-truth-mask-relative-path."
        )

    token = build_run_token(
        method=method,
        dropout_rate=dropout_rate,
        n_sample=n_sample,
        num_models=num_models,
    )
    uq_method_dir = Path(uq_method_dir)
    uq_seg_data_dir = resolve_uq_seg_pickle(
        uq_method_dir,
        method=method,
        dropout_rate=dropout_rate,
        n_sample=n_sample,
        num_models=num_models,
    )

    print(f"Reading {uq_seg_data_dir}", flush=True)
    uq_seg_data = load_uq_seg_pickle(uq_seg_data_dir)
    source_labels_available = constants.TRUE_CLASS_COLUMN_NAME in uq_seg_data
    if (
        method in constants.UQ_METHODS_WITH_EMBEDDED_SEGMENTATION_LABELS
        and source_labels_available != labels_available
    ):
        raise ValueError(
            f"Saved segmentation UQ artifact {uq_seg_data_dir} is "
            f"{'labeled' if source_labels_available else 'unlabeled'}, but "
            f"aggregation is configured as {inference_mode}."
        )
    (
        cases_id,
        pred_tumor,
        predictive_entropy,
        expected_entropy,
        mutual_information,
    ) = validate_segmentation_uq_data(
        uq_seg_data=uq_seg_data,
        source_path=uq_seg_data_dir,
    )

    mask_brain = get_brain_masks_for_all_patients(
        input_dir=input_dir,
        patient_ids=cases_id,
        brain_mask_relative_path=brain_mask_relative_path,
    )
    if mask_brain.shape != pred_tumor.shape:
        raise ValueError(
            f"Brain-mask array has shape {mask_brain.shape}, while the saved "
            f"predictions have shape {pred_tumor.shape}."
        )

    masks_dir = uq_method_dir / f"{constants.SEGMENTATION_MASKS_DIR_PREFIX}{token}"
    uq_maps_dir = uq_method_dir / f"{constants.SEGMENTATION_UQ_MAPS_DIR_PREFIX}{token}"
    dice_score_per_patient: np.ndarray | None = None
    if ground_truth_available:
        if ground_truth_mask_relative_path is None:
            raise RuntimeError(
                "Internal error: ground truth was marked available without a path."
            )
        mask_tumor = get_ground_truth_for_all_patients(
            input_dir=input_dir,
            patient_ids=cases_id,
            ground_truth_mask_relative_path=ground_truth_mask_relative_path,
        )
        if mask_tumor.shape != pred_tumor.shape:
            raise ValueError(
                f"Ground-truth array has shape {mask_tumor.shape}, while the "
                f"saved predictions have shape {pred_tumor.shape}."
            )
        dice_output_dir = masks_dir if export_masks else None
        dice_score_per_patient = compute_dice_per_patient(
            input_dir=input_dir,
            y_true=mask_tumor,
            y_pred=pred_tumor,
            reference_mask_relative_path=ground_truth_mask_relative_path,
            out_dir=dice_output_dir,
            patient_ids=cases_id,
        )

    if export_masks and not ground_truth_available:
        export_segmentation_masks(
            input_dir=input_dir,
            y_pred=pred_tumor,
            reference_mask_relative_path=brain_mask_relative_path,
            out_dir=masks_dir,
            patient_ids=cases_id,
        )

    if export_maps:
        export_uncertainty_maps(
            input_dir,
            cases_id,
            predictive_entropy,
            uq_maps_dir,
            map_type="PUM",
            reference_mask_relative_path=brain_mask_relative_path,
        )
        export_uncertainty_maps(
            input_dir,
            cases_id,
            expected_entropy,
            uq_maps_dir,
            map_type="AUM",
            reference_mask_relative_path=brain_mask_relative_path,
        )
        export_uncertainty_maps(
            input_dir,
            cases_id,
            mutual_information,
            uq_maps_dir,
            map_type="EUM",
            reference_mask_relative_path=brain_mask_relative_path,
        )

    uncertainty_aggregations = compute_segmentation_uncertainty_aggregations(
        pred_tumor=pred_tumor,
        mask_brain=mask_brain,
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        dilation_radius_voxels=dilation_radius_voxels,
        boundary_sigma_voxels=boundary_sigma_voxels,
    )

    uq_csv_path = uq_method_dir / (
        f"{constants.SEGMENTATION_UQ_PREFIX}_{token}{table_extension}"
    )
    if not uq_csv_path.exists():
        raise FileNotFoundError(f"Missing source UQ table: {uq_csv_path}")

    seg_info_csv = align_segmentation_table_to_cases(
        table=read_output_table(uq_csv_path, index_col=None),
        cases_id=cases_id,
        source_path=uq_csv_path,
    )
    for column, values in uncertainty_aggregations.items():
        seg_info_csv[column] = values
    if dice_score_per_patient is not None:
        seg_info_csv[constants.SEGMENTATION_DICE_SCORE_COLUMN] = dice_score_per_patient

    muq_csv_path = uq_method_dir / (
        f"{constants.SEGMENTATION_MUQ_PREFIX}_{token}{table_extension}"
    )
    write_output_table(seg_info_csv, muq_csv_path, index=False)
    print(f"Wrote MUQ: {muq_csv_path}", flush=True)
    return muq_csv_path


def compute_muq_for_dropout_grid(
    input_dir: str | Path,
    experiments_dir: str | Path,
    method: str,
    dropout_rates: list[float],
    n_samples: list[int] | None,
    num_models: int | None,
    ground_truth_mask_relative_path: str | None,
    brain_mask_relative_path: str,
    export_maps: bool,
    export_masks: bool,
    dilation_radius_voxels: int,
    boundary_sigma_voxels: float,
    table_extension: str,
    inference_mode: str,
) -> list[Path]:
    """Compute mask-based segmentation uncertainty tables for a dropout grid.

    Args:
        input_dir: Dataset directory containing patient folders.
        experiments_dir: Root directory containing dropout_* experiments.
        method: UQ method name, such as mcd, de, or mcd_de.
        dropout_rates: Dropout rates to process.
        n_samples: MC sample counts to process.
        num_models: Number of models to use for ensemble-based methods.
        ground_truth_mask_relative_path: Ground-truth mask path relative to each
            patient directory, or None when ground truth is unavailable.
        brain_mask_relative_path: Brain-mask path relative to each patient
            directory.
        export_maps: Whether to export PUM/AUM/EUM NIfTI maps.
        export_masks: Whether to export predicted masks, plus GT masks when
            ground truth is available.
        dilation_radius_voxels: Predicted-tumor dilation radius in voxels.
        boundary_sigma_voxels: Boundary weighting Gaussian sigma in voxels.
        table_extension: Configured extension for source and generated tables.
        inference_mode: Whether ground-truth masks are required or forbidden.

    Returns:
        List of written MUQ table paths.

    Raises:
        ValueError: If a required method-specific parameter is absent.
    """
    if not dropout_rates:
        raise ValueError("dropout_rates must contain at least one value.")
    if method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES and not n_samples:
        raise ValueError(f"n_samples is required for UQ method '{method}'.")
    if method in constants.UQ_METHODS_REQUIRING_NUM_MODELS and num_models is None:
        raise ValueError(f"num_models is required for UQ method '{method}'.")

    written = []
    for dropout_rate in dropout_rates:
        uq_method_dir = resolve_uq_method_dir(experiments_dir, method, dropout_rate)
        if not uq_method_dir.exists():
            raise FileNotFoundError(
                f"Missing saved-UQ method directory: {uq_method_dir}"
            )

        if method == constants.UQ_METHOD_DE:
            written.append(
                compute_muq_for_setting(
                    input_dir=input_dir,
                    uq_method_dir=uq_method_dir,
                    method=method,
                    dropout_rate=dropout_rate,
                    n_sample=None,
                    num_models=num_models,
                    ground_truth_mask_relative_path=ground_truth_mask_relative_path,
                    brain_mask_relative_path=brain_mask_relative_path,
                    export_maps=export_maps,
                    export_masks=export_masks,
                    dilation_radius_voxels=dilation_radius_voxels,
                    boundary_sigma_voxels=boundary_sigma_voxels,
                    table_extension=table_extension,
                    inference_mode=inference_mode,
                )
            )
            continue

        if n_samples is None:
            raise RuntimeError(
                "Internal error: MC sample counts passed validation but are absent."
            )
        for n_sample in n_samples:
            written.append(
                compute_muq_for_setting(
                    input_dir=input_dir,
                    uq_method_dir=uq_method_dir,
                    method=method,
                    dropout_rate=dropout_rate,
                    n_sample=n_sample,
                    num_models=num_models,
                    ground_truth_mask_relative_path=ground_truth_mask_relative_path,
                    brain_mask_relative_path=brain_mask_relative_path,
                    export_maps=export_maps,
                    export_masks=export_masks,
                    dilation_radius_voxels=dilation_radius_voxels,
                    boundary_sigma_voxels=boundary_sigma_voxels,
                    table_extension=table_extension,
                    inference_mode=inference_mode,
                )
            )

    return written


def parse_dropout_rate(value: str) -> float:
    """Parse and validate one dropout rate CLI value.

    Args:
        value: Raw CLI value.

    Returns:
        Dropout rate as a float.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed as a valid
            dropout rate.
    """
    try:
        dropout_rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dropout rates must be floats") from exc
    if not 0 <= dropout_rate < 1:
        raise argparse.ArgumentTypeError("dropout rates must satisfy 0 <= rate < 1")
    return dropout_rate


def parse_positive_int(value: str) -> int:
    """Parse and validate one positive integer CLI value.

    Args:
        value: Raw CLI value.

    Returns:
        Positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample counts must be integers") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive")
    return parsed


def parse_positive_float(value: str) -> float:
    """Parse and validate one positive float CLI value.

    Args:
        value: Raw CLI value.

    Returns:
        Positive floating-point value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive float.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "boundary sigma must be a floating-point value"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("boundary sigma must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute mask-based segmentation uncertainty aggregations from saved "
            "UQ segmentation outputs."
        )
    )
    parser.add_argument(
        "--table-extension",
        choices=constants.SUPPORTED_TABLE_OUTPUT_FORMATS,
        required=True,
        help="Configured extension for source and generated tables.",
    )
    parser.add_argument(
        "--inference-mode",
        required=True,
        choices=constants.SUPPORTED_INFERENCE_MODES,
        help="Whether ground-truth masks are required or forbidden.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing case subdirectories.",
    )
    parser.add_argument(
        "--ground-truth-mask-relative-path",
        help=(
            "Ground-truth mask path relative to each case directory. Omit this "
            "for unlabeled inference; Dice and GT-mask export will then be skipped, "
            "while predicted-mask export remains available."
        ),
    )
    parser.add_argument(
        "--brain-mask-relative-path",
        required=True,
        help="Brain-mask path relative to each case directory.",
    )
    parser.add_argument(
        "--export-maps",
        action="store_true",
        help="Export PUM/AUM/EUM NIfTI maps.",
    )
    parser.add_argument(
        "--export-masks",
        action="store_true",
        help=(
            "Export predicted NIfTI masks, and GT masks as well when ground "
            "truth is available."
        ),
    )
    parser.add_argument(
        "--dilation-radius-voxels",
        required=True,
        type=parse_positive_int,
        help="Predicted-tumor dilation radius in voxels.",
    )
    parser.add_argument(
        "--boundary-sigma-voxels",
        required=True,
        type=parse_positive_float,
        help="Boundary weighting Gaussian sigma in voxels.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--uq_method_dir",
        help="Single saved-UQ method directory to process.",
    )
    mode_group.add_argument(
        "--experiments-dir",
        help="Root directory containing dropout_*/results/UQ/<method>.",
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=constants.SUPPORTED_UQ_METHODS,
        help="UQ method used to interpret the saved directory layout.",
    )

    parser.add_argument(
        "--dropout-rate",
        type=parse_dropout_rate,
        help="Dropout rate for single-run mode with --uq_method_dir.",
    )
    parser.add_argument(
        "--n-samples",
        type=parse_positive_int,
        help="Sample count for single-run mode with --uq_method_dir.",
    )
    parser.add_argument(
        "--num-models",
        type=parse_positive_int,
        help="Number of models for de or mcd_de runs.",
    )
    parser.add_argument(
        "--dropout-rates",
        nargs="+",
        type=parse_dropout_rate,
        help="Dropout rates for grid mode with --experiments-dir.",
    )
    parser.add_argument(
        "--n-samples-list",
        nargs="+",
        type=parse_positive_int,
        help="MC sample counts for grid mode with --experiments-dir.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate mutually dependent CLI arguments.

    Args:
        args: Parsed CLI namespace.
        parser: Parser used to report argument errors.
    """
    ground_truth_available = bool(
        args.ground_truth_mask_relative_path
        and args.ground_truth_mask_relative_path.strip()
    )
    if (
        args.inference_mode == constants.INFERENCE_MODE_LABELED
        and not ground_truth_available
    ):
        parser.error(
            "--ground-truth-mask-relative-path is required when "
            "--inference-mode=labeled"
        )
    if (
        args.inference_mode == constants.INFERENCE_MODE_UNLABELED
        and ground_truth_available
    ):
        parser.error(
            "--ground-truth-mask-relative-path must be omitted when "
            "--inference-mode=unlabeled"
        )

    if args.uq_method_dir:
        missing = []
        if args.dropout_rate is None:
            missing.append("--dropout-rate")
        if (
            args.method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES
            and args.n_samples is None
        ):
            missing.append("--n-samples")
        if (
            args.method in constants.UQ_METHODS_REQUIRING_NUM_MODELS
            and args.num_models is None
        ):
            missing.append("--num-models")
        if args.method == constants.UQ_METHOD_MCD and args.num_models is not None:
            parser.error("--num-models is only valid with de or mcd_de")
        if args.method == constants.UQ_METHOD_DE and args.n_samples is not None:
            parser.error("--n-samples is only valid with mcd or mcd_de")
        if args.dropout_rates is not None:
            parser.error("--dropout-rates is only valid with --experiments-dir")
        if args.n_samples_list is not None:
            parser.error("--n-samples-list is only valid with --experiments-dir")
        if missing:
            parser.error(
                "single-run mode with --uq_method_dir requires " + " and ".join(missing)
            )
    else:
        missing = []
        if args.dropout_rates is None:
            missing.append("--dropout-rates")
        if (
            args.method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES
            and args.n_samples_list is None
        ):
            missing.append("--n-samples-list")
        if (
            args.method in constants.UQ_METHODS_REQUIRING_NUM_MODELS
            and args.num_models is None
        ):
            missing.append("--num-models")
        if args.method == constants.UQ_METHOD_MCD and args.num_models is not None:
            parser.error("--num-models is only valid with de or mcd_de")
        if args.method == constants.UQ_METHOD_DE and args.n_samples_list is not None:
            parser.error("--n-samples-list is only valid with mcd or mcd_de")
        if args.dropout_rate is not None:
            parser.error("--dropout-rate is only valid with --uq_method_dir")
        if args.n_samples is not None:
            parser.error("--n-samples is only valid with --uq_method_dir")
        if missing:
            parser.error(
                "grid mode with --experiments-dir requires " + " and ".join(missing)
            )


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments.

    Returns:
        Validated CLI namespace.
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def main() -> None:
    """Run mask-based segmentation uncertainty aggregation from CLI arguments."""
    args = parse_args()
    if args.uq_method_dir:
        written = [
            compute_muq_for_setting(
                input_dir=args.input_dir,
                uq_method_dir=args.uq_method_dir,
                method=args.method,
                dropout_rate=args.dropout_rate,
                n_sample=args.n_samples,
                num_models=args.num_models,
                ground_truth_mask_relative_path=args.ground_truth_mask_relative_path,
                brain_mask_relative_path=args.brain_mask_relative_path,
                export_maps=args.export_maps,
                export_masks=args.export_masks,
                dilation_radius_voxels=args.dilation_radius_voxels,
                boundary_sigma_voxels=args.boundary_sigma_voxels,
                table_extension=args.table_extension,
                inference_mode=args.inference_mode,
            )
        ]
    else:
        written = compute_muq_for_dropout_grid(
            input_dir=args.input_dir,
            experiments_dir=args.experiments_dir,
            method=args.method,
            dropout_rates=args.dropout_rates,
            n_samples=args.n_samples_list,
            num_models=args.num_models,
            ground_truth_mask_relative_path=args.ground_truth_mask_relative_path,
            brain_mask_relative_path=args.brain_mask_relative_path,
            export_maps=args.export_maps,
            export_masks=args.export_masks,
            dilation_radius_voxels=args.dilation_radius_voxels,
            boundary_sigma_voxels=args.boundary_sigma_voxels,
            table_extension=args.table_extension,
            inference_mode=args.inference_mode,
        )

    print(f"Finished. Wrote {len(written)} MUQ table files.", flush=True)


if __name__ == "__main__":
    main()
