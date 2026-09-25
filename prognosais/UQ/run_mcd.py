import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.data.utils import pad_list_data_collate

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.IO.dataset import DataGenerator
from prognosais.IO.utils import seed_worker, set_random_seed
from prognosais.model.development.inference_utils import (
    build_inference_transforms,
    build_model_from_config,
    model_does_segmentation,
)
from prognosais.UQ.mcd import MCDropout
from prognosais.UQ.task_utils import build_task_specs

parser = argparse.ArgumentParser(description="Run inference using a trained model")

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
    "--temp-root",
    type=Path,
    help="Existing writable directory for temporary voxelwise MCD tensors.",
)

parser.add_argument(
    "-cf",
    "--current_fold",
    required=False,
    help="Current fold in which inference is being performed",
    dest="current_fold",
    type=str,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
config_dir = this_script_dir.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {os.path.abspath(__file__)} was run with the following config file: {config_dir}",
    flush=True,
)

config = configIO.Config(config_dir)

env_seed = config.environment_seed
torch_generator = set_random_seed(env_seed)

if args.current_fold is None:
    model_file_dir = config.test_model_dir
    results_dir = os.path.join(config.test_results_dir, constants.RESULTS_DIR_NAME)
    os.makedirs(results_dir, exist_ok=True)

    if model_file_dir is None or not str(model_file_dir).strip():
        model_type = config.test_model_type
        if model_type is None:
            raise ValueError(
                "Since you did not provide a model file, model type is mandatory to retrieve the model file automatically."
            )
        else:
            if model_type == "best":
                model_file_dir = os.path.join(
                    results_dir, constants.MODELS_DIR_NAME, "best_model.pt"
                )
            else:
                model_file_dir = os.path.join(
                    results_dir, constants.MODELS_DIR_NAME, "last_model.pt"
                )
else:
    results_dir = os.path.join(
        config.test_results_dir, f"fold_{args.current_fold}", constants.RESULTS_DIR_NAME
    )
    os.makedirs(results_dir, exist_ok=True)

    model_type = config.test_model_type
    if model_type is None:
        raise ValueError(
            "For k-fold cross-validation, model type is mandatory to retrieve the model files automatically."
        )
    else:
        if model_type == "best":
            model_file_dir = os.path.join(
                results_dir, constants.MODELS_DIR_NAME, "best_model.pt"
            )
            print(f"Model file dir was {model_file_dir}")
        else:
            model_file_dir = os.path.join(
                results_dir, constants.MODELS_DIR_NAME, "last_model.pt"
            )

dropout_rate = config.mc_dropout_rate

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model file dir: {model_file_dir}")

if not Path(model_file_dir).is_file():
    raise FileNotFoundError(
        f"The configured MCD model checkpoint does not exist: {model_file_dir}"
    )

model_info = torch.load(model_file_dir, map_location=device)
model = build_model_from_config(
    config=config,
    dropout_rate=dropout_rate,
)
try:
    model.load_state_dict(model_info["model_state_dict"])
except RuntimeError as error:
    raise RuntimeError(
        "The checkpoint is incompatible with the configured architecture, "
        "modalities, or classification tasks. Run MCD with the frozen "
        "configuration that belongs to this checkpoint."
    ) from error
model.to(device)
batch_size = config.test_batch_size
inference_mode = config.data_test_inference_mode
labels_file_dir = config.data_test_labels_file_dir
subset = config.data_test_subset
missing_value = config.data_test_missing_value
data_dir = config.data_test_dir
data_type = config.data_test_type
mc_samples = config.mc_dropout_samples

print("Modules printing", flush=True)
for name, module in model.named_modules():
    if module.__class__.__name__.startswith("Dropout"):
        print(f"Dropout layer: {name}, Dropout rate: {module.p}", flush=True)

does_segmentation = model_does_segmentation(config)
labels_available = inference_mode == constants.INFERENCE_MODE_LABELED
require_mask = does_segmentation and labels_available
task_specs = build_task_specs(config.classification_tasks)

test_dict = DataGenerator(
    data_dir=data_dir,
    file_extension=constants.DATA_NIFTI_EXTENSION,
    labels_file_dir=labels_file_dir,
    data_type=data_type,
    train=False,
    subset=subset,
    missing_value=missing_value,
    seed=env_seed,
    modalities=config.data_modalities,
    mask_file_name=config.data_mask_file_name,
    data_folders=config.data_folders,
    labels_config=config.labels,
    case_id_column=config.data_label_case_id_column,
    require_mask=require_mask,
)

test_transforms = build_inference_transforms(
    config=config,
    require_mask=require_mask,
)

test_dataset = Dataset(test_dict.data, transform=test_transforms)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=pad_list_data_collate,
    generator=torch_generator,
    worker_init_fn=seed_worker,
)

if __name__ == "__main__":

    dropout_rate_str = str(dropout_rate).replace(".", "")

    if args.temp_root is None:
        parser.error("--temp-root is required for MCD temporary storage.")
    temp_root = args.temp_root.expanduser().resolve(strict=True)

    if not temp_root.is_dir():
        raise NotADirectoryError(
            f"MCD temporary-storage directory does not exist: {temp_root}"
        )

    fold_token = f"{args.current_fold}_" if args.current_fold is not None else ""
    temp_dir = tempfile.mkdtemp(
        prefix=(
            f"UQ_{Path(config.test_results_dir).name}_{dropout_rate_str}_"
            f"{mc_samples}_{fold_token}"
        ),
        dir=temp_root,
    )

    temp_pointer_path = Path(mcd_dir, f"tmp_dir_do{dropout_rate_str}_{mc_samples}s.txt")
    temp_pointer_path.write_text(temp_dir, encoding="utf-8")

    mc_dropout = MCDropout(
        model=model,
        device=device,
        data_loader=test_data_loader,
        dropout_rate=dropout_rate,
        mc_samples=mc_samples,
        mcd_dir=mcd_dir,
        temp_dir=temp_dir,
        task_specs=task_specs,
        does_segmentation=does_segmentation,
        inference_mode=inference_mode,
        labels_available=labels_available,
        mask_available=require_mask,
        image_extension=config.output_images_format,
        table_extension=config.output_tables_format,
    )
    mc_dropout.output_paths["temp_pointer"] = temp_pointer_path
    mc_dropout.run()
    if labels_available:
        mc_dropout.violin_plot()

    # Move the temporary directory to the mc_dropout directory

    original_temp_dir = mc_dropout.temp_dir
    relocated_temp_dir = Path(shutil.move(original_temp_dir, mc_dropout.mcd_dir))
    mc_dropout.relocate_temp_outputs(relocated_temp_dir)
    temp_pointer_path.write_text(str(relocated_temp_dir), encoding="utf-8")
    mc_dropout.write_manifest()
    print(f"Moved {original_temp_dir} to {relocated_temp_dir}")
