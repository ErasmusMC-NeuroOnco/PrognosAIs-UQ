import argparse
import os
from pathlib import Path

from slurmpie import slurmpie

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants

parser = argparse.ArgumentParser(
    description="Quantify uncertainty given a model and a method"
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

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
root_dir = this_script_dir.parent.parent.resolve()
config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {os.path.abspath(__file__)} was run with the following config file: {config_dir}",
    flush=True,
)

config = configIO.Config(config_dir)
inference_mode = config.data_test_inference_mode
print(f"MCD inference mode: {inference_mode}", flush=True)
mc_samples = config.mc_dropout_samples
dropout_rate = config.mc_dropout_rate
dropout_rate_str = str(dropout_rate).replace(".", "")

if config.test_kfold:
    num_folds = config.test_num_folds

cpu_header = config.cpu_node_job_header
gpu_header = config.gpu_node_job_header

gpu_header = gpu_header.strip().replace("\n", "")
cpu_header = cpu_header.strip().replace("\n", "")

venv_command = "source " + str(root_dir.joinpath(".venv", "bin", "activate")) + " && "

gpu_header += venv_command
cpu_header += venv_command

UQ_command = (
    gpu_header
    + "python -u "
    + str(root_dir.joinpath("prognosais", "UQ", "run_mcd.py"))
    + f" -c {args.config} "
)

if __name__ == "__main__":

    if config.test_kfold:

        for fold in range(config.test_num_folds):

            fold_uq_command = UQ_command + f" -cf {fold}"
            job_UQ = slurmpie.Job(fold_uq_command, script_is_file=False)

            results_dir = os.path.join(
                config.test_results_dir, f"fold_{fold}", constants.RESULTS_DIR_NAME
            )
            os.makedirs(results_dir, exist_ok=True)
            mcd_dir = os.path.join(
                results_dir,
                constants.UQ_RESULTS_DIR_NAME,
                constants.MC_DROPOUT_DIR_NAME,
            )
            os.makedirs(mcd_dir, exist_ok=True)
            log_dir_out = os.path.join(
                mcd_dir, constants.LOGFILES_DIR_NAME, constants.OUTPUT_LOGFILES_DIR_NAME
            )
            os.makedirs(log_dir_out, exist_ok=True)
            log_dir_err = os.path.join(
                mcd_dir, constants.LOGFILES_DIR_NAME, constants.ERROR_LOGFILES_DIR_NAME
            )
            os.makedirs(log_dir_err, exist_ok=True)

            job_UQ.output_file = (
                f"{log_dir_out}/out_do{dropout_rate_str}_{mc_samples}s_f{fold}.log"
            )
            job_UQ.error_file = (
                f"{log_dir_err}/err_do{dropout_rate_str}_{mc_samples}s_f{fold}.log"
            )

            job_UQ.name = f"mcd_{os.path.basename(config.test_results_dir)}_fold{fold}_{mc_samples}_samples"
            job_UQ.tasks = 1
            job_UQ.nodes = 1
            job_UQ.mail_address = config.slurm_email
            job_UQ.mail_type = "END,FAIL,ARRAY_TASKS"
            job_UQ.partition = config.gpu_node_partition
            job_UQ.gres = config.gpu_node_gres
            job_UQ.time = config.slurm_test_time
            job_UQ.submit()
    else:

        job_UQ = slurmpie.Job(UQ_command, script_is_file=False)

        results_dir = os.path.join(config.test_results_dir, constants.RESULTS_DIR_NAME)
        os.makedirs(results_dir, exist_ok=True)
        mcd_dir = os.path.join(
            results_dir, constants.UQ_RESULTS_DIR_NAME, constants.MC_DROPOUT_DIR_NAME
        )
        os.makedirs(mcd_dir, exist_ok=True)
        log_dir_out = os.path.join(
            mcd_dir, constants.LOGFILES_DIR_NAME, constants.OUTPUT_LOGFILES_DIR_NAME
        )
        os.makedirs(log_dir_out, exist_ok=True)
        log_dir_err = os.path.join(
            mcd_dir, constants.LOGFILES_DIR_NAME, constants.ERROR_LOGFILES_DIR_NAME
        )
        os.makedirs(log_dir_err, exist_ok=True)
        job_UQ.name = (
            f"mcd_{os.path.basename(config.test_results_dir)}_{mc_samples}_samples"
        )
        job_UQ.tasks = 1
        job_UQ.nodes = 1
        job_UQ.mail_address = config.slurm_email
        job_UQ.partition = config.gpu_node_partition
        job_UQ.time = config.slurm_test_time
        job_UQ.gres = config.gpu_node_gres
        job_UQ.memory_size = config.gpu_node_memory_size(constants.UQ_METHOD_MCD)
        job_UQ.output_file = (
            f"{log_dir_out}/out_do{dropout_rate_str}_{mc_samples}s_%j.log"
        )
        job_UQ.error_file = (
            f"{log_dir_err}/err_do{dropout_rate_str}_{mc_samples}s_%j.log"
        )
        job_UQ.mail_type = "END,FAIL"
        job_UQ.submit()
