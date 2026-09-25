import argparse
import os
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import AsDiscreted, Compose

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.IO.dataset import DataGenerator
from prognosais.IO.utils import seed_worker, set_random_seed
from prognosais.model.architectures.Evaluator import Evaluator
from prognosais.model.development.inference_utils import (
    build_inference_transforms,
    build_model_from_config,
    model_does_segmentation,
)
from prognosais.model.development.metrics import ClassificationMetrics

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
    "-cf",
    "--current_fold",
    required=False,
    help="Current fold in which inference is being performed",
    dest="current_fold",
    type=str,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
root_dir = this_script_dir.parent.parent.resolve()
config_dir = this_script_dir.parent.parent.resolve().joinpath("configs", args.config)
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
    if not model_file_dir or not str(model_file_dir).strip():
        model_type = config.test_model_type
        if not model_type or not str(model_type).strip():
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
    print(f"The model loaded is: {model_file_dir}")
else:
    results_dir = os.path.join(
        config.test_results_dir, f"fold_{args.current_fold}", constants.RESULTS_DIR_NAME
    )
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
        else:
            model_file_dir = os.path.join(
                results_dir, constants.MODELS_DIR_NAME, "last_model.pt"
            )

metrics_dir = os.path.join(results_dir, constants.METRICS_DIR_NAME)
os.makedirs(metrics_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_info = torch.load(model_file_dir, map_location=device)
model = build_model_from_config(
    config=config,
    dropout_rate=config.train_dropout_rate,
)

model.load_state_dict(model_info["model_state_dict"])
model.to(device)
batch_size = config.test_batch_size
subset = config.data_test_subset
missing_value = config.data_test_missing_value
data_dir = config.data_test_dir
data_type = config.data_test_type
test_kfold = config.test_kfold

inference_mode = config.data_test_inference_mode
labels_file_dir = config.data_test_labels_file_dir

does_segmentation = model_does_segmentation(config)

require_mask = does_segmentation and inference_mode == constants.INFERENCE_MODE_LABELED

test_dict = DataGenerator(
    data_dir=data_dir,
    file_extension=constants.DATA_NIFTI_EXTENSION,
    labels_file_dir=labels_file_dir,
    data_type=data_type,
    train=False,
    subset=subset,
    missing_value=missing_value,
    ids_to_exclude=None,
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
    generator=torch_generator,
    worker_init_fn=seed_worker,
)

if does_segmentation:
    post_transform_keys = ["pred_seg"]

    if require_mask:
        post_transform_keys.append("mask")

    post_transforms = Compose(
        [
            AsDiscreted(
                keys=post_transform_keys,
                argmax=True,
            ),
        ]
    )
else:
    post_transforms = Compose([])

if __name__ == "__main__":

    model_evaluator = Evaluator(
        model=model,
        test_loader=test_data_loader,
        device=device,
        post_transforms=post_transforms,
        results_dir=results_dir,
        table_extension=config.output_tables_format,
    )

    model_evaluator.evaluate(
        save_predictions=config.test_save_predictions,
        save_prob_map=config.test_save_prob_map,
    )

    # A segmentation-metrics file from an earlier run is removed to avoid confusion with
    # new metrics generated in a new run under the same results directory.
    segmentation_metrics_path = Path(metrics_dir) / (
        f"{constants.SEGMENTATION_METRICS_STEM}{config.output_tables_format}"
    )
    segmentation_metrics_path.unlink(missing_ok=True)

    # Compute classification metrics if MRI files, segmentations and labels are available

    if inference_mode == constants.INFERENCE_MODE_LABELED:

        classification_metrics = ClassificationMetrics(
            config_file_dir=config_dir, results_dir=results_dir
        )

        if test_kfold:
            classification_metrics.generate_roc(fold_name=str(args.current_fold))
            classification_metrics.generate_pr(fold_name=str(args.current_fold))
            classification_metrics.generate_cf(fold_name=str(args.current_fold))
            classification_metrics.calculate_tp_tn_fp_fn()
            classification_metrics.calculate_specificity()
            classification_metrics.generate_classification_report()
            classification_metrics.barplots(fold_name=str(args.current_fold))
            if require_mask:
                classification_metrics.get_bp_dice_hd()
        else:
            classification_metrics.generate_roc(fold_name=None)
            classification_metrics.generate_pr(fold_name=None)
            classification_metrics.generate_cf(fold_name=None)
            classification_metrics.calculate_tp_tn_fp_fn()
            classification_metrics.calculate_specificity()
            classification_metrics.generate_classification_report()
            classification_metrics.barplots(fold_name=None)
            if require_mask:
                classification_metrics.get_bp_dice_hd()
