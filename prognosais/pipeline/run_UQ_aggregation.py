"""Submit mask-based UQ aggregation jobs for saved uncertainty outputs."""

import argparse
import os
import shlex
from pathlib import Path

from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.model.development.inference_utils import model_does_segmentation


def method_to_results_subdir(method: str) -> str:
    """Map a UQ method name to its results subdirectory name.

    Args:
        method: UQ method name such as mcd, de, or mcd_de.

    Returns:
        str: Directory name used under results/UQ for the chosen method.

    Raises:
        ValueError: If the method is not supported.
    """
    if method not in constants.UQ_METHOD_RESULTS_DIR_NAMES:
        raise ValueError(
            f"Unsupported UQ method {method}. Supported methods are: "
            + ", ".join(constants.SUPPORTED_UQ_METHODS)
        )
    return constants.UQ_METHOD_RESULTS_DIR_NAMES[method]


def get_method_parameters(
    config: configIO.Config, method: str
) -> dict[str, int | float | None]:
    """Collect method-specific aggregation parameters from the config.

    Args:
        config: Parsed PrognosAIs configuration object.
        method: UQ method name such as mcd, de, or mcd_de.

    Returns:
        dict[str, int | float | None]: Parameters required for the selected
        method.
    """
    if method == constants.UQ_METHOD_MCD:
        return {
            "dropout_rate": config.mc_dropout_rate,
            "n_samples": config.mc_dropout_samples,
            "num_models": None,
        }
    if method == constants.UQ_METHOD_DE:
        return {
            "dropout_rate": config.uq_method_dropout_rate(constants.UQ_METHOD_DE),
            "n_samples": None,
            "num_models": config.uq_method_num_models(constants.UQ_METHOD_DE),
        }
    if method == constants.UQ_METHOD_MCD_DE:
        return {
            "dropout_rate": config.uq_method_dropout_rate(constants.UQ_METHOD_MCD_DE),
            "n_samples": config.uq_method_num_samples(constants.UQ_METHOD_MCD_DE),
            "num_models": config.uq_method_num_models(constants.UQ_METHOD_MCD_DE),
        }
    raise ValueError(
        f"Unsupported UQ method {method}. Supported methods are: "
        + ", ".join(constants.SUPPORTED_UQ_METHODS)
    )


def build_run_token(
    method: str,
    dropout_rate: float,
    n_samples: int | None,
    num_models: int | None,
) -> str:
    """Build the filename token for one UQ aggregation run.

    Args:
        method: UQ method name.
        dropout_rate: Dropout rate used by the run.
        n_samples: Number of MC samples, when relevant.
        num_models: Number of ensemble models, when relevant.

    Returns:
        str: Token used in filenames and job names.

    Raises:
        ValueError: If method is unsupported or a method-specific count is
            absent.
    """
    if method not in constants.SUPPORTED_UQ_METHODS:
        raise ValueError(
            f"Unsupported UQ method '{method}'. Supported methods: "
            + ", ".join(constants.SUPPORTED_UQ_METHODS)
        )
    if method in constants.UQ_METHODS_REQUIRING_NUM_MODELS and num_models is None:
        raise ValueError(f"num_models is required for UQ method '{method}'.")
    if method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES and n_samples is None:
        raise ValueError(f"n_samples is required for UQ method '{method}'.")

    token = f"do{dropout_token(dropout_rate)}"
    if method in constants.UQ_METHODS_REQUIRING_NUM_MODELS:
        token += f"_{num_models}m"
    if method in constants.UQ_METHODS_REQUIRING_MC_SAMPLES:
        token += f"_{n_samples}s"
    return token


def dropout_token(dropout_rate: float) -> str:
    """Convert a dropout rate into the repository filename token.

    Args:
        dropout_rate: Dropout rate, for example 0.25.

    Returns:
        str: Token used in filenames, for example "025".
    """
    return str(dropout_rate).replace(".", "")


def build_results_dir(config: configIO.Config, current_fold: int | None) -> Path:
    """Build the results directory for a test run or k-fold test fold.

    Args:
        config: Parsed PrognosAIs configuration object.
        current_fold: Fold index, or None for a non-k-fold run.

    Returns:
        Path: Path to the corresponding results directory.
    """
    if current_fold is None:
        return Path(config.test_results_dir) / constants.RESULTS_DIR_NAME
    return (
        Path(config.test_results_dir)
        / f"fold_{current_fold}"
        / constants.RESULTS_DIR_NAME
    )


def build_uncertainty_dir(results_dir: Path, uncertainty_subdir_name: str) -> Path:
    """Build the UQ output directory for a results directory.

    Args:
        results_dir: Path to a results directory.
        uncertainty_subdir_name: Name of the uncertainty-method subdirectory.

    Returns:
        Path: Path to the results/UQ directory containing the saved UQ
        outputs.
    """
    return Path(results_dir) / constants.UQ_RESULTS_DIR_NAME / uncertainty_subdir_name


def build_mask_aggregation_command(
    root_dir: Path,
    config_name: str,
    config: configIO.Config,
    uncertainty_dir: Path,
    method: str,
    run_params: dict[str, int | float | None],
) -> str:
    """Build the shell command for UQ mask aggregation.

    Args:
        root_dir: Repository root directory.
        config_name: Configuration file name or path passed to the pipeline.
        config: Parsed PrognosAIs configuration object.
        uncertainty_dir: Path to the UQ directory to aggregate.
        method: UQ method name.
        run_params: Dictionary with method-specific parameters.

    Returns:
        str: Shell command string that runs UQ_aggregation.py.
    """
    inference_mode = config.data_test_inference_mode
    command_parts = [
        "python",
        "-u",
        str(root_dir / "prognosais" / "UQ" / "UQ_aggregation.py"),
        "--input-dir",
        config.data_test_dir,
        "--inference-mode",
        inference_mode,
        "--brain-mask-relative-path",
        config.mask_based_seg_unc_brain_mask_relative_path,
        "--uq_method_dir",
        str(uncertainty_dir),
        "--method",
        method,
        "--dropout-rate",
        str(run_params["dropout_rate"]),
        "--dilation-radius-voxels",
        str(config.mask_based_seg_unc_dilation_radius_voxels),
        "--boundary-sigma-voxels",
        str(config.mask_based_seg_unc_boundary_sigma_voxels),
        "--table-extension",
        config.output_tables_format,
    ]

    if inference_mode == constants.INFERENCE_MODE_LABELED:
        command_parts.extend(
            [
                "--ground-truth-mask-relative-path",
                config.mask_based_seg_unc_ground_truth_mask_relative_path,
            ]
        )
    if run_params["n_samples"] is not None:
        command_parts.extend(["--n-samples", str(run_params["n_samples"])])
    if run_params["num_models"] is not None:
        command_parts.extend(["--num-models", str(run_params["num_models"])])

    if config.mask_based_seg_unc_export_maps:
        command_parts.append("--export-maps")
    if config.mask_based_seg_unc_export_masks:
        command_parts.append("--export-masks")

    quoted_command = " ".join(shlex.quote(part) for part in command_parts)
    return (
        f"echo Running UQ mask aggregation with {shlex.quote(config_name)} "
        f"&& {quoted_command}"
    )


def build_cpu_command(root_dir: Path, cpu_header: str, aggregation_command: str) -> str:
    """Build the full CPU-node shell command for a submitted job.

    Args:
        root_dir: Repository root directory.
        cpu_header: Cluster-specific module-loading command from the config.
        aggregation_command: Python command that performs the aggregation.

    Returns:
        str: Shell command string including environment activation.
    """
    cpu_header = cpu_header.strip().replace("\n", "")
    venv_command = "source " + str(root_dir / ".venv" / "bin" / "activate") + " && "
    return cpu_header + venv_command + aggregation_command


def configure_job(
    job: slurmpie.Job,
    config: configIO.Config,
    uncertainty_dir: Path,
    method: str,
    run_token: str,
    current_fold: int | None,
) -> None:
    """Configure a slurmpie job for UQ mask aggregation.

    Args:
        job: slurmpie.Job instance to configure.
        config: Parsed PrognosAIs configuration object.
        uncertainty_dir: Path to the UQ directory.
        method: UQ method name.
        run_token: Filename token representing the configured run.
        current_fold: Fold index, or None for a non-k-fold run.
    """
    log_dir_out = (
        uncertainty_dir
        / constants.LOGFILES_DIR_NAME
        / constants.OUTPUT_LOGFILES_DIR_NAME
    )
    log_dir_err = (
        uncertainty_dir
        / constants.LOGFILES_DIR_NAME
        / constants.ERROR_LOGFILES_DIR_NAME
    )
    log_dir_out.mkdir(parents=True, exist_ok=True)
    log_dir_err.mkdir(parents=True, exist_ok=True)

    job.tasks = 1
    job.nodes = 1
    job.mail_address = config.slurm_email
    job.mail_type = "END,FAIL"
    job.partition = config.cpu_node_partition
    job.cpus_per_task = config.cpu_node_cpus_per_task
    job.time = config.slurm_test_time
    job.memory_size = config.cpu_node_memory_size(
        constants.UQ_AGGREGATION_MEMORY_CONFIG_KEY
    )

    if current_fold is None:
        job.name = (
            f"uq_agg_{method}_{os.path.basename(config.test_results_dir)}_"
            f"{run_token}"
        )
        job.output_file = f"{log_dir_out}/out_uqagg_{method}_{run_token}_%j.log"
        job.error_file = f"{log_dir_err}/err_uqagg_{method}_{run_token}_%j.log"
    else:
        job.name = (
            f"uq_agg_{method}_{os.path.basename(config.test_results_dir)}_"
            f"fold{current_fold}_{run_token}"
        )
        job.output_file = (
            f"{log_dir_out}/out_uqagg_{method}_{run_token}_f{current_fold}_%j.log"
        )
        job.error_file = (
            f"{log_dir_err}/err_uqagg_{method}_{run_token}_f{current_fold}_%j.log"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run mask-based aggregation for UQ outputs."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Name of the configuration file",
        metavar="configuration file",
        dest="config",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=constants.SUPPORTED_UQ_METHODS,
        help="UQ method used to locate the saved results directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Submit UQ mask aggregation jobs for the configured test run."""
    args = parse_args()
    this_script_dir = Path(__file__).parent.resolve()
    root_dir = this_script_dir.parent.parent.resolve()
    config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
    print(
        f"This job and script {os.path.abspath(__file__)} was run with "
        f"the following config file: {config_dir}",
        flush=True,
    )

    config = configIO.Config(config_dir)
    if not model_does_segmentation(config):
        raise ValueError(
            "UQ mask aggregation requires segmentation output. The configured "
            f"model architecture '{config.model_architecture}' with "
            f"classification_only={config.classification_only} only produces "
            "classification predictions, so there are no voxelwise maps to "
            "aggregate."
        )
    uncertainty_subdir_name = method_to_results_subdir(args.method)
    run_params = get_method_parameters(config, args.method)
    run_token = build_run_token(
        args.method,
        run_params["dropout_rate"],
        n_samples=run_params["n_samples"],
        num_models=run_params["num_models"],
    )

    if config.test_kfold:
        folds = range(config.test_num_folds)
    else:
        folds = [None]

    for current_fold in folds:
        results_dir = build_results_dir(config, current_fold)
        uncertainty_dir = build_uncertainty_dir(results_dir, uncertainty_subdir_name)
        uncertainty_dir.mkdir(parents=True, exist_ok=True)

        aggregation_command = build_mask_aggregation_command(
            root_dir=root_dir,
            config_name=args.config,
            config=config,
            uncertainty_dir=uncertainty_dir,
            method=args.method,
            run_params=run_params,
        )
        job_command = build_cpu_command(
            root_dir=root_dir,
            cpu_header=config.cpu_node_job_header,
            aggregation_command=aggregation_command,
        )
        job = slurmpie.Job(job_command, script_is_file=False)
        configure_job(
            job=job,
            config=config,
            uncertainty_dir=uncertainty_dir,
            method=args.method,
            run_token=run_token,
            current_fold=current_fold,
        )
        job.submit()


if __name__ == "__main__":
    main()
