"""Run the paper branch directly, on a workstation or inside an allocated job."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from prognosais.IO import constants
from prognosais.IO.config import Config


def model_does_segmentation(config: Config) -> bool:
    """Return whether the configured CSNet run emits segmentation logits.

    Keep this small predicate local so command construction and ``--dry-run``
    do not import the Torch/MONAI inference stack.
    """

    return (
        config.model_architecture == constants.CSNET_MODEL_TYPE
        and not config.classification_only
    )


def build_command(
    config: Config,
    config_path: Path,
    mode: str,
    method: str | None,
    fold: int | None,
    scratch_dir: Path | None,
    run_id: str | None,
) -> list[str]:
    """Build an argument list for an existing computation entry point.

    Args:
        config: Configuration used for the computation.
        config_path: Absolute path to its frozen YAML file.
        mode: Training, inference, uncertainty estimation, or aggregation.
        method: Uncertainty method, required for UQ and aggregation.
        fold: Zero-based fold index, or None for a single model.
        scratch_dir: Writable temporary storage for MCD tensors.
        run_id: MLflow run identifier for training.

    Returns:
        Subprocess arguments using the current Python interpreter.

    Raises:
        ValueError: If method-specific parameters are missing or inconsistent.
    """
    root = Path(__file__).resolve().parents[2]
    if mode in ("train", "inference") or (mode == "uq" and method == "mcd"):
        module = {
            "train": "prognosais.model.development.train",
            "inference": "prognosais.model.development.inference",
            "uq": "prognosais.UQ.run_mcd",
        }[mode]
        command = [sys.executable, "-u", "-m", module, "-c", str(config_path)]
        if mode == "train":
            if not run_id:
                raise ValueError("Training requires an MLflow run identifier.")
            command.extend(["--run_id", run_id])
        if fold is not None:
            command.extend(["--current_fold", str(fold)])
        if mode == "uq":
            if scratch_dir is None or not scratch_dir.is_dir():
                raise ValueError(
                    "MCD requires --scratch-dir pointing to an existing directory."
                )
            command.extend(["--temp-root", str(scratch_dir)])
        return command

    from prognosais.pipeline.run_UQ_aggregation import method_to_results_subdir

    results = Path(config.test_results_dir)
    if fold is not None:
        results = results / f"fold_{fold}"
    uq_dir = (
        results
        / constants.RESULTS_DIR_NAME
        / constants.UQ_RESULTS_DIR_NAME
        / method_to_results_subdir(method)
    )
    if mode == "aggregate":
        from prognosais.pipeline.run_UQ_aggregation import get_method_parameters

        if not model_does_segmentation(config):
            raise ValueError(
                "Mask-based UQ aggregation requires CSNet segmentation output. "
                "Set model.classification_only to false."
            )

        params = get_method_parameters(config, method)
        inference_mode = config.data_test_inference_mode
        command = [
            sys.executable,
            "-u",
            "-m",
            "prognosais.UQ.UQ_aggregation",
            "--input-dir",
            config.data_test_dir,
            "--inference-mode",
            inference_mode,
            "--brain-mask-relative-path",
            config.mask_based_seg_unc_brain_mask_relative_path,
            "--uq_method_dir",
            str(uq_dir),
            "--method",
            method,
            "--dropout-rate",
            str(params["dropout_rate"]),
            "--dilation-radius-voxels",
            str(config.mask_based_seg_unc_dilation_radius_voxels),
            "--boundary-sigma-voxels",
            str(config.mask_based_seg_unc_boundary_sigma_voxels),
            "--table-extension",
            config.output_tables_format,
        ]
        if inference_mode == constants.INFERENCE_MODE_LABELED:
            command.extend(
                [
                    "--ground-truth-mask-relative-path",
                    config.mask_based_seg_unc_ground_truth_mask_relative_path,
                ]
            )
        for key, flag in (("n_samples", "--n-samples"), ("num_models", "--num-models")):
            if params[key] is not None:
                command.extend([flag, str(params[key])])
        if config.mask_based_seg_unc_export_maps:
            command.append("--export-maps")
        if config.mask_based_seg_unc_export_masks:
            command.append("--export-masks")
        return command

    from prognosais.pipeline.run_de import build_de_command, validate_ensemble_config
    from prognosais.pipeline.run_mcd_de import build_mcd_de_command

    ensemble_root, template, seeds = validate_ensemble_config(config, method)
    inference_mode = config.data_test_inference_mode
    prediction_summary_stem = (
        constants.PREDICTIONS_SUMMARY_LABELED_STEM
        if inference_mode == constants.INFERENCE_MODE_LABELED
        else constants.PREDICTIONS_SUMMARY_UNLABELED_STEM
    )
    predictions_relative_path = (
        f"{constants.RESULTS_DIR_NAME}/{constants.METRICS_DIR_NAME}/"
        f"{prediction_summary_stem}{config.output_tables_format}"
    )
    kwargs = dict(
        root_dir=root,
        ensemble_root=ensemble_root,
        seeds=seeds,
        seed_folder_template=template,
        output_dir=uq_dir,
        dropout_rate=config.uq_method_dropout_rate(method),
        tasks=list(config.classification_tasks),
        probability_sum_tolerance=constants.PROBABILITY_SUM_TOLERANCE,
        entropy_epsilon=constants.ENTROPY_EPSILON,
        include_segmentation=model_does_segmentation(config),
        table_extension=config.output_tables_format,
        inference_mode=inference_mode,
    )
    if method == "de":
        command_text = build_de_command(
            **kwargs,
            predictions_relative_path=predictions_relative_path,
            probability_map_relative_path=constants.PROBABILITY_MAP_RELATIVE_PATH,
            missing_label=config.data_test_missing_value,
        )
    elif method == "mcd_de":
        command_text = build_mcd_de_command(
            **kwargs,
            n_samples=config.uq_method_num_samples(method),
            mc_folder=constants.MC_DROPOUT_RELATIVE_DIR,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    command = shlex.split(command_text)
    command[0] = sys.executable
    return command


def freeze_config(config_path: Path, frozen_dir: Path) -> Path:
    """Create a distinct configuration snapshot before starting computation.

    Args:
        config_path: YAML file selected by the user.
        frozen_dir: Directory where run-specific copies are retained.

    Returns:
        Path to the complete snapshot used by child processes.
    """
    frozen_dir.mkdir(parents=True, exist_ok=True)
    fd, frozen_name = tempfile.mkstemp(
        prefix=f"{config_path.stem}_", suffix=".yml", dir=frozen_dir
    )
    os.close(fd)
    frozen = Path(frozen_name)
    shutil.copyfile(config_path, frozen)
    return frozen


def main() -> None:
    """Freeze the requested config and execute each requested fold sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "inference", "uq", "aggregate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("mcd", "de", "mcd_de"))
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.mode in ("uq", "aggregate")) != (args.method is not None):
        parser.error("--method is required only for uq and aggregate.")
    path = args.config.expanduser().resolve(strict=True)
    config = Config(path)
    kfold = config.train_kfold if args.mode == "train" else config.test_kfold
    count = config.train_num_folds if args.mode == "train" else config.test_num_folds
    if args.fold is not None and (not kfold or not 0 <= args.fold < count):
        parser.error("--fold requires kfold: true and an index below num_folds.")
    if args.method in ("de", "mcd_de") and kfold:
        parser.error(
            "Ensemble aggregation uses seed folders; set model.test.kfold to false."
        )
    if args.mode == "aggregate" and not model_does_segmentation(config):
        parser.error(
            "Mask-based aggregation requires CSNet segmentation output; set "
            "model.classification_only to false."
        )
    folds = (
        [args.fold]
        if args.fold is not None
        else list(range(count)) if kfold else [None]
    )
    scratch_dir = (
        args.scratch_dir.expanduser().resolve(strict=True) if args.scratch_dir else None
    )
    root = Path(__file__).resolve().parents[2]
    # A dry run must neither create an MLflow run nor freeze a config.
    for fold in folds:
        print(
            shlex.join(
                build_command(
                    config,
                    path,
                    args.mode,
                    args.method,
                    fold,
                    scratch_dir,
                    "CREATED_AT_EXECUTION",
                )
            ),
            flush=True,
        )
    if args.dry_run:
        return
    frozen_dir = root / "prognosais/configs/frozen"
    frozen = freeze_config(path, frozen_dir)
    config = Config(frozen)
    print(f"Frozen config: {frozen}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["MPLBACKEND"] = "Agg"
    for fold in folds:
        run_id = None
        client = None
        if args.mode == "train":
            import mlflow
            from mlflow.tracking import MlflowClient

            experiment = mlflow.set_experiment(config.experiment_name)
            client = MlflowClient()
            run_id = client.create_run(
                experiment.experiment_id, tags={"mlflow.runName": config.run_name}
            ).info.run_id
        command = build_command(
            config, frozen, args.mode, args.method, fold, scratch_dir, run_id
        )
        print(json.dumps({"command": command, "fold": fold}), flush=True)
        try:
            subprocess.run(command, cwd=root, env=environment, check=True)
        except BaseException:
            if client is not None:
                client.set_terminated(run_id, status="FAILED")
            raise
        else:
            if client is not None:
                client.set_terminated(run_id, status="FINISHED")


if __name__ == "__main__":
    main()
