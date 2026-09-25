"""Submit deterministic deep ensemble aggregation as a cluster job."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.model.development.inference_utils import model_does_segmentation
from prognosais.UQ.de import (
    validate_required_sequence,
    validate_required_value,
    validate_seed_folders_exist,
)


def validate_ensemble_config(
    config: configIO.Config, method: str
) -> tuple[Path, str, list[int]]:
    """Validate and return the configured ensemble location parameters.

    Args:
        config: Parsed PrognosAIs configuration.
        method: UQ method key to validate.

    Returns:
        Tuple containing the ensemble root directory, seed folder template,
        and explicit seed list.

    Raises:
        ValueError: If num_models does not match the number of configured seeds.
    """

    ensemble_root_config = config.uq_method_ensemble_root_dir(method)
    validate_required_value(ensemble_root_config, f"{method}.ensemble_root_dir")
    ensemble_root = Path(ensemble_root_config)
    seed_folder_template = config.uq_method_seed_folder_template(method)
    validate_required_value(seed_folder_template, f"{method}.seed_folder_template")
    seeds = [
        int(seed)
        for seed in validate_required_sequence(
            config.uq_method_seeds(method), f"{method}.seeds"
        )
    ]
    num_models = config.uq_method_num_models(method)
    if num_models != len(seeds):
        raise ValueError(
            f"UQ method '{method}' defines num_models={num_models}, but "
            f"{len(seeds)} seeds were configured: {seeds}."
        )
    validate_seed_folders_exist(
        ensemble_root=ensemble_root,
        seeds=seeds,
        seed_folder_template=seed_folder_template,
        method=method,
    )
    return ensemble_root, seed_folder_template, seeds


def build_de_command(
    root_dir: Path,
    ensemble_root: Path,
    seeds: list[int],
    seed_folder_template: str,
    output_dir: Path,
    dropout_rate: float,
    tasks: list[str],
    predictions_relative_path: str,
    inference_mode: str,
    probability_map_relative_path: str,
    missing_label: int,
    probability_sum_tolerance: float,
    entropy_epsilon: float,
    include_segmentation: bool,
    table_extension: str,
) -> str:
    """Build the shell command that performs DE aggregation.

    Args:
        root_dir: Repository root directory.
        ensemble_root: Directory containing seed folders.
        seeds: Ensemble-member seeds.
        seed_folder_template: Template used to locate each seed folder.
        output_dir: Directory where DE outputs should be written.
        dropout_rate: Dropout-rate token used in segmentation output filenames.
        tasks: Classification tasks to aggregate.
        predictions_relative_path: Relative path from a seed folder to the
            prediction summary selected by inference_mode.
        inference_mode: Whether the selected summaries are labeled or unlabeled.
        probability_map_relative_path: Relative probability-map path template.
        missing_label: Label value indicating unavailable annotations.
        probability_sum_tolerance: Allowed probability-sum deviation.
        entropy_epsilon: Numerical-stability epsilon used for entropy.
        include_segmentation: Whether to aggregate segmentation probability
            maps.
        table_extension: Configured extension for generated tables.

    Returns:
        Shell command string.
    """

    validate_required_value(root_dir, "root_dir")
    validate_required_value(ensemble_root, "ensemble_root")
    validate_required_sequence(seeds, "seeds")
    validate_required_value(seed_folder_template, "seed_folder_template")
    validate_required_value(output_dir, "output_dir")
    validate_required_value(dropout_rate, "dropout_rate")
    validate_required_sequence(tasks, "tasks")
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
    validate_required_value(entropy_epsilon, "entropy_epsilon")
    validate_required_value(include_segmentation, "include_segmentation")
    validate_required_value(table_extension, "table_extension")

    command_parts = [
        "python",
        "-u",
        str(root_dir / "prognosais" / "UQ" / "run_de.py"),
        "--ensemble-root",
        str(ensemble_root),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--seed-folder-template",
        seed_folder_template,
        "--output-dir",
        str(output_dir),
        "--dropout-rate",
        str(dropout_rate),
        "--tasks",
        *tasks,
        "--predictions-relative-path",
        predictions_relative_path,
        "--inference-mode",
        inference_mode,
        "--probability-map-relative-path",
        probability_map_relative_path,
        "--missing-label",
        str(missing_label),
        "--probability-sum-tolerance",
        str(probability_sum_tolerance),
        "--entropy-epsilon",
        str(entropy_epsilon),
        "--table-extension",
        table_extension,
        "--include-segmentation" if include_segmentation else "--skip-segmentation",
    ]
    return " ".join(shlex.quote(part) for part in command_parts)


def build_cpu_command(root_dir: Path, cpu_header: str, command: str) -> str:
    """Build a cluster command with environment activation.

    Args:
        root_dir: Repository root directory.
        cpu_header: Cluster-specific module-loading command.
        command: Python command to run after activation.

    Returns:
        Full shell command submitted to Slurm.
    """

    cpu_header = cpu_header.strip().replace("\n", "")
    venv_command = "source " + str(root_dir / ".venv" / "bin" / "activate") + " && "
    return cpu_header + venv_command + command


def configure_job(
    job: slurmpie.Job,
    config: configIO.Config,
    output_dir: Path,
    seeds: list[int],
) -> None:
    """Configure a DE aggregation Slurm job.

    Args:
        job: slurmpie job object to configure.
        config: Parsed PrognosAIs configuration.
        output_dir: Directory where DE outputs are written.
        seeds: Ensemble-member seeds.
    """

    log_dir_out = (
        output_dir / constants.LOGFILES_DIR_NAME / constants.OUTPUT_LOGFILES_DIR_NAME
    )
    log_dir_err = (
        output_dir / constants.LOGFILES_DIR_NAME / constants.ERROR_LOGFILES_DIR_NAME
    )
    log_dir_out.mkdir(parents=True, exist_ok=True)
    log_dir_err.mkdir(parents=True, exist_ok=True)

    current_seed = config.environment_seed
    job.name = (
        f"de_{os.path.basename(config.test_results_dir)}_{len(seeds)}m_"
        f"{current_seed}"
    )
    job.tasks = 1
    job.nodes = 1
    job.mail_address = config.slurm_email
    job.mail_type = "END,FAIL"
    job.partition = config.cpu_node_partition
    job.cpus_per_task = config.cpu_node_cpus_per_task
    job.time = config.slurm_test_time
    job.memory_size = config.cpu_node_memory_size(constants.UQ_METHOD_DE)
    job.output_file = f"{log_dir_out}/out_de_{len(seeds)}m_{current_seed}_%j.log"
    job.error_file = f"{log_dir_err}/err_de_{len(seeds)}m_{current_seed}_%j.log"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="Submit deep ensemble aggregation.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Name of the configuration file",
        metavar="configuration file",
        dest="config",
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    """Submit the configured DE aggregation job."""

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
    ensemble_root, seed_folder_template, seeds = validate_ensemble_config(
        config=config, method=constants.UQ_METHOD_DE
    )
    dropout_rate = config.uq_method_dropout_rate(constants.UQ_METHOD_DE)
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
    output_dir = (
        Path(config.test_results_dir)
        / constants.RESULTS_DIR_NAME
        / constants.UQ_RESULTS_DIR_NAME
        / constants.UQ_METHOD_RESULTS_DIR_NAMES[constants.UQ_METHOD_DE]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_de_command(
        root_dir=root_dir,
        ensemble_root=ensemble_root,
        seeds=seeds,
        seed_folder_template=seed_folder_template,
        output_dir=output_dir,
        dropout_rate=dropout_rate,
        tasks=list(config.classification_tasks),
        predictions_relative_path=predictions_relative_path,
        inference_mode=inference_mode,
        probability_map_relative_path=constants.PROBABILITY_MAP_RELATIVE_PATH,
        missing_label=config.data_test_missing_value,
        probability_sum_tolerance=constants.PROBABILITY_SUM_TOLERANCE,
        entropy_epsilon=constants.ENTROPY_EPSILON,
        include_segmentation=model_does_segmentation(config),
        table_extension=config.output_tables_format,
    )
    job_command = build_cpu_command(
        root_dir=root_dir,
        cpu_header=config.cpu_node_job_header,
        command=command,
    )
    job = slurmpie.Job(job_command, script_is_file=False)
    configure_job(job=job, config=config, output_dir=output_dir, seeds=seeds)
    job.submit()


if __name__ == "__main__":
    main()
