import argparse
import os
from pathlib import Path

from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants

parser = argparse.ArgumentParser(description="Perform inference given a trained model")
parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Name of the configuration file",
    metavar="configuration file",
    dest="config",
    type=str,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
root_dir = this_script_dir.parent.parent.resolve()
config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {os.path.abspath(__file__)} was run with the following config file: {config_dir}",
    flush=True,
)

config = configIO.Config(config_dir)

if config.test_kfold:
    num_folds = config.test_num_folds

cpu_header = config.cpu_node_job_header
gpu_header = config.gpu_node_job_header

gpu_header = gpu_header.strip().replace("\n", "")
cpu_header = cpu_header.strip().replace("\n", "")

venv_command = "source " + str(root_dir.joinpath(".venv", "bin", "activate")) + " && "

gpu_header += venv_command
cpu_header += venv_command

log_dir = Path(config.test_results_dir)
log_dir_err = log_dir.joinpath(
    constants.LOGFILES_DIR_NAME, constants.ERROR_LOGFILES_DIR_NAME
)
log_dir_out = log_dir.joinpath(
    constants.LOGFILES_DIR_NAME, constants.OUTPUT_LOGFILES_DIR_NAME
)

inference_command = (
    gpu_header
    + "python "
    + str(root_dir.joinpath("prognosais", "model", "development", "inference.py"))
    + f" -c {args.config} "
)

kfold_command = (
    cpu_header
    + "python "
    + str(root_dir.joinpath("prognosais", "model", "development", "metrics_kfold.py"))
    + f" -c {args.config} "
)

if __name__ == "__main__":

    if config.test_kfold:

        pipeline_inference = slurmpie.Pipeline()

        inference_command += " -cf $SLURM_ARRAY_TASK_ID"
        job_inference = slurmpie.Job(inference_command, script_is_file=False)

        job_inference.array = f"0-{config.test_num_folds-1}%{config.test_num_folds}"
        job_inference.output_file = log_dir.joinpath(
            log_dir_out, "out_inference_fold_$SLURM_ARRAY_TASK_ID_%A-%a.log"
        )
        job_inference.error_file = log_dir.joinpath(
            log_dir_err, "err_inference_fold_$SLURM_ARRAY_TASK_ID_%A-%a.log"
        )

        job_inference.name = "inference_fold_$SLURM_ARRAY_TASK_ID"
        job_inference.tasks = 1
        job_inference.nodes = 1
        job_inference.mail_address = config.slurm_email
        job_inference.mail_type = "END,FAIL,ARRAY_TASKS"
        job_inference.partition = config.gpu_node_partition
        job_inference.cpus_per_task = 4
        job_inference.gres = config.gpu_node_gres
        job_inference.time = config.slurm_test_time
        pipeline_inference.add(job_inference)

        if config.data_test_inference_mode == constants.INFERENCE_MODE_LABELED:
            job_metrics = slurmpie.Job(kfold_command, script_is_file=False)
            job_metrics.output_file = log_dir.joinpath(
                log_dir_out, "out_metrics_%j.log"
            )
            job_metrics.error_file = log_dir.joinpath(log_dir_err, "err_metrics_%j.log")
            job_metrics.name = "metrics_kfold"
            job_metrics.tasks = 1
            job_metrics.nodes = 1
            job_metrics.mail_address = config.slurm_email
            job_metrics.mail_type = "END,FAIL"
            job_metrics.partition = config.gpu_node_partition
            job_metrics.gres = config.gpu_node_gres
            job_metrics.cpus_per_task = 4
            job_metrics.time = config.slurm_test_time
            pipeline_inference.add({"afterok": [job_metrics]}, parent_job=job_inference)
        else:
            print(
                "Skipping k-fold metrics summary because test inference mode "
                "is unlabeled.",
                flush=True,
            )
        pipeline_inference.submit()

    else:
        job_inference = slurmpie.Job(inference_command, script_is_file=False)
        job_inference.output_file = Path(log_dir_out).joinpath("out_inference_%j.log")
        job_inference.error_file = Path(log_dir_err).joinpath("err_inference_%j.log")
        job_inference.name = "inference"
        job_inference.tasks = 1
        job_inference.nodes = 1
        job_inference.mail_address = config.slurm_email
        job_inference.mail_type = "END,FAIL"
        job_inference.partition = config.gpu_node_partition
        job_inference.cpus_per_task = 4
        job_inference.gres = config.gpu_node_gres
        job_inference.time = config.slurm_test_time
        job_inference.submit()
