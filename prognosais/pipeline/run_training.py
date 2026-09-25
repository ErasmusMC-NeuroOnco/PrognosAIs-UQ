import argparse
import os
import shlex
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants

parser = argparse.ArgumentParser(description="Train a model")
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

all_experiments = Path(config.experiments_dir)
experiment_dir = all_experiments.joinpath(config.experiment_name)
run_dir = experiment_dir.joinpath(experiment_dir, config.run_name)
info_dir = run_dir.joinpath(run_dir, constants.INFORMATION_DIR_NAME)

os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(info_dir, exist_ok=True)

shutil.copy(config_dir, info_dir.joinpath(info_dir, "config.yml"))

log_dir = run_dir.joinpath(run_dir, constants.LOGFILES_DIR_NAME)
out_log_dir = run_dir.joinpath(log_dir, constants.OUTPUT_LOGFILES_DIR_NAME)
err_log_dir = run_dir.joinpath(log_dir, constants.ERROR_LOGFILES_DIR_NAME)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(out_log_dir, exist_ok=True)
os.makedirs(err_log_dir, exist_ok=True)

results_dir = run_dir.joinpath(run_dir, constants.RESULTS_DIR_NAME)

gpu_header = config.gpu_node_job_header
gpu_header = gpu_header.strip().replace("\n", "")

venv_command = ("source " + str(root_dir.joinpath(".venv", "bin", "activate"))) + " && "

gpu_header += venv_command

if __name__ == "__main__":

    client = MlflowClient()

    if not client.get_experiment_by_name(config.experiment_name):
        experiment_id = mlflow.create_experiment(config.experiment_name)
    else:
        print("Existing experiment..")
        existing_experiment = mlflow.set_experiment(config.experiment_name)
        experiment_id = existing_experiment.experiment_id

    with mlflow.start_run(
        run_name=config.run_name, experiment_id=experiment_id
    ) as parent_run:
        run_id = parent_run.info.run_id

    os.environ.pop("MLFLOW_RUN_ID", None)

    training_command = (
        gpu_header
        + "MLFLOW_TRACKING_URI="
        + shlex.quote(mlflow.get_tracking_uri())
        + " "
        + "python "
        + str(root_dir.joinpath("prognosais", "model", "development", "train.py"))
        + f" -r_id {run_id} "
        + f" -c {args.config} "
    )

    if config.train_kfold:
        training_command += "-cf $SLURM_ARRAY_TASK_ID"
        job = slurmpie.Job(training_command, script_is_file=False)
        job.array = f"0-{config.train_num_folds-1}%{config.train_num_folds}"
        job.output_file = log_dir.joinpath(out_log_dir, "out_train_%A-%a.log")
        job.error_file = log_dir.joinpath(err_log_dir, "err_train_%A-%a.log")
        job.mail_type = "END,FAIL,ARRAY_TASKS"

    else:
        job = slurmpie.Job(training_command, script_is_file=False)
        job.output_file = log_dir.joinpath(out_log_dir, "out_train_%j.log")
        job.error_file = log_dir.joinpath(err_log_dir, "err_train_%j.log")
        job.mail_type = "END,FAIL"

    job.name = config.run_name
    job.tasks = 1
    job.nodes = 1
    job.mail_address = config.slurm_email
    job.partition = config.gpu_node_partition
    job.cpus_per_task = 4
    job.gres = config.gpu_node_gres
    job.time = config.slurm_train_time
    job.memory_size = "15GB"
    job.submit()
