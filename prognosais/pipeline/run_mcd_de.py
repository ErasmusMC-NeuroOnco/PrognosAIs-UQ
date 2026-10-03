"""Submit Monte Carlo deep ensemble aggregation as a cluster job."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.model.development.inference_utils import model_does_segmentation
from prognosais.pipeline.run_de import build_cpu_command, validate_ensemble_config
from prognosais.UQ.de import validate_required_sequence, validate_required_value


def dropout_token(dropout_rate: float) -> str:
    """Convert a dropout rate into the repository filename token.

    Args:
        dropout_rate: Dropout rate used by seed-level MCD.

    Returns:
        Filename token, for example 025 for 0.25.
    """

    return str(dropout_rate).replace(".", "")


def build_mcd_de_command(
    root_dir: Path,
    ensemble_root: Path,
    seeds: list[int],
    seed_folder_template: str,
    output_dir: Path,
    dropout_rate: float,
    n_samples: int,
    tasks: list[str],
    mc_folder: str,
    probability_sum_tolerance: float,
    entropy_epsilon: float,
    include_segmentation: bool,
    table_extension: str,
    inference_mode: str,
) -> str:
    """Build the shell command that performs MCDE aggregation.

    Args:
        root_dir: Repository root directory.
        ensemble_root: Directory containing seed folders.
        seeds: Ensemble-member seeds.
        seed_folder_template: Template used to locate each seed folder.
        output_dir: Directory where MCDE outputs should be written.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MC dropout samples per seed.
        tasks: Classification tasks to aggregate.
        mc_folder: Relative or absolute seed-level MC dropout folder.
        probability_sum_tolerance: Allowed probability-sum deviation.
        entropy_epsilon: Numerical-stability epsilon used for entropy.
        include_segmentation: Whether to aggregate seed-level MCD
            segmentation mean-probability pickles.
        table_extension: Configured extension for generated tables.
        inference_mode: Whether source MCD outputs are labeled or unlabeled.

    Returns:
        Shell command string.
    """

    validate_required_value(root_dir, "root_dir")
    validate_required_value(ensemble_root, "ensemble_root")
    validate_required_sequence(seeds, "seeds")
    validate_required_value(seed_folder_template, "seed_folder_template")
    validate_required_value(output_dir, "output_dir")
    validate_required_value(dropout_rate, "dropout_rate")
    validate_required_value(n_samples, "n_samples")
    validate_required_sequence(tasks, "tasks")
    validate_required_value(mc_folder, "mc_folder")
    validate_required_value(probability_sum_tolerance, "probability_sum_tolerance")
    validate_required_value(entropy_epsilon, "entropy_epsilon")
    validate_required_value(include_segmentation, "include_segmentation")
    validate_required_value(table_extension, "table_extension")
    validate_required_value(inference_mode, "inference_mode")
    if inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference mode '{inference_mode}'. Supported values: "
            + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
        )

    command_parts = [
        "python",
        "-u",
        str(root_dir / "prognosais" / "UQ" / "run_mcd_de.py"),
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
        "--n-samples",
        str(n_samples),
        "--tasks",
        *tasks,
        "--mc-folder",
        mc_folder,
        "--probability-sum-tolerance",
        str(probability_sum_tolerance),
        "--entropy-epsilon",
        str(entropy_epsilon),
        "--table-extension",
        table_extension,
        "--inference-mode",
        inference_mode,
        "--include-segmentation" if include_segmentation else "--skip-segmentation",
    ]
    return " ".join(shlex.quote(part) for part in command_parts)


def configure_job(
    job: slurmpie.Job,
    config: configIO.Config,
    output_dir: Path,
    seeds: list[int],
    dropout_rate: float,
    n_samples: int,
) -> None:
    """Configure an MCDE aggregation Slurm job.

    Args:
        job: slurmpie job object to configure.
        config: Parsed PrognosAIs configuration.
        output_dir: Directory where MCDE outputs are written.
        seeds: Ensemble-member seeds.
        dropout_rate: Dropout rate used by seed-level MCD.
        n_samples: Number of MC dropout samples per seed.
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
    run_token = f"do{dropout_token(dropout_rate)}_{len(seeds)}m_{n_samples}s"
    job.name = (
        f"mcd_de_{os.path.basename(config.test_results_dir)}_{run_token}_"
        f"{current_seed}"
    )
    job.tasks = 1
    job.nodes = 1
    job.mail_address = config.slurm_email
    job.mail_type = "END,FAIL"
    job.partition = config.cpu_node_partition
    job.memory_size = config.cpu_node_memory_size(constants.UQ_METHOD_MCD_DE)
    job.cpus_per_task = config.cpu_node_cpus_per_task
    job.time = config.slurm_test_time
    job.output_file = f"{log_dir_out}/out_mcd_de_{run_token}_{current_seed}_%j.log"
    job.error_file = f"{log_dir_err}/err_mcd_de_{run_token}_{current_seed}_%j.log"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description="Submit Monte Carlo deep ensemble aggregation."
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
    return parser.parse_args()


def main() -> None:
    """Submit the configured MCDE aggregation job."""

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
        config=config, method=constants.UQ_METHOD_MCD_DE
    )
    dropout_rate = config.uq_method_dropout_rate(constants.UQ_METHOD_MCD_DE)
    n_samples = config.uq_method_num_samples(constants.UQ_METHOD_MCD_DE)
    inference_mode = config.data_test_inference_mode
    output_dir = (
        Path(config.test_results_dir)
        / constants.RESULTS_DIR_NAME
        / constants.UQ_RESULTS_DIR_NAME
        / constants.UQ_METHOD_RESULTS_DIR_NAMES[constants.UQ_METHOD_MCD_DE]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_mcd_de_command(
        root_dir=root_dir,
        ensemble_root=ensemble_root,
        seeds=seeds,
        seed_folder_template=seed_folder_template,
        output_dir=output_dir,
        dropout_rate=dropout_rate,
        n_samples=n_samples,
        tasks=list(config.classification_tasks),
        mc_folder=constants.MC_DROPOUT_RELATIVE_DIR,
        probability_sum_tolerance=constants.PROBABILITY_SUM_TOLERANCE,
        entropy_epsilon=constants.ENTROPY_EPSILON,
        include_segmentation=model_does_segmentation(config),
        table_extension=config.output_tables_format,
        inference_mode=inference_mode,
    )
    job_command = build_cpu_command(
        root_dir=root_dir,
        cpu_header=config.cpu_node_job_header,
        command=command,
    )
    job = slurmpie.Job(job_command, script_is_file=False)
    configure_job(
        job=job,
        config=config,
        output_dir=output_dir,
        seeds=seeds,
        dropout_rate=dropout_rate,
        n_samples=n_samples,
    )
    job.submit()


if __name__ == "__main__":
    main()
