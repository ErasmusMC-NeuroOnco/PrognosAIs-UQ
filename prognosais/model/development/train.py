import argparse
import os
from pathlib import Path

import numpy as np
import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
import torch
from monai.data import Dataset
from monai.transforms import (
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotated,
    RandShiftIntensityd,
)
from prognosais.IO.dataset import DataGenerator, KFoldGenerator
from prognosais.IO.transforms import RandCropandZerod
from prognosais.IO.utils import seed_worker, set_random_seed
from prognosais.model.architectures.CSNet import CSNet
from prognosais.model.architectures.Trainer import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Run a training of the prognosais pipeline"
)
parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Name of the configuration file",
    dest="config",
    type=str,
)

parser.add_argument(
    "-r_id",
    "--run_id",
    required=True,
    help="ID of the MLflow run",
    dest="run_id",
    type=str,
)

parser.add_argument(
    "-cf",
    "--current_fold",
    required=False,
    help="Current fold for which to run the experiment",
    dest="current_fold",
    type=int,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
config_dir = this_script_dir.parent.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {os.path.abspath(__file__)} was run with the following config file: {config_dir}",
    flush=True,
)
config = configIO.Config(config_dir)

env_seed = config.environment_seed
torch_generator = set_random_seed(env_seed)

# Data variables
train_dir = config.data_train_dir
data_type = config.data_train_type
subset = config.data_train_subset
missing_value = config.data_train_missing_value
modalities = config.data_modalities
if not modalities:
    raise ValueError(
        "No conventional MRI modalities are enabled under 'data.modalities'."
    )

classification_tasks = config.classification_tasks
num_classes_per_task = {
    task_name: config.get_label_num_classes(task_name)
    for task_name in classification_tasks
}

print("=" * 80)
print("Active conventional MRI modalities for this training run:")
for modality_key, modality_name in modalities.items():
    print(f"  - {modality_key}: {modality_name}")

print("\nActive classification tasks:")
for task_name in classification_tasks:
    print(
        f"  - {task_name}: "
        f"{config.get_label_column(task_name)} "
        f"({num_classes_per_task[task_name]} classes)"
    )

print("=" * 80, flush=True)


# Training variables
num_epochs = config.train_epochs
batch_size = config.train_batch_size
dropout_rate = config.train_dropout_rate
learning_rate = config.train_optimizer_lr
weight_decay = config.train_optimizer_weight_decay
delta = config.train_early_stopping_delta
patience = config.train_early_stopping_patience
augmentation_probability = config.data_train_augmentation_probability
augmentation_factor = config.data_train_augmentation_factor
architecture = config.model_architecture
does_segmentation = (
    architecture == constants.CSNET_MODEL_TYPE
    and not config.classification_only
)

print(
    f"Training with dropout rate: {dropout_rate} and augmentation factor: {augmentation_factor}",
    flush=True,
)

if architecture != constants.CSNET_MODEL_TYPE:
    raise ValueError(
        f"Unsupported model architecture '{architecture}'. "
        f"This repository supports {constants.CSNET_MODEL_TYPE} only."
    )

model = CSNet(
    dropout_rate=dropout_rate,
    in_channels=len(modalities),
    classification_tasks=num_classes_per_task,
    classification_only=config.classification_only,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=config.train_optimizer_scheduler_reduction_factor,
    patience=config.train_optimizer_scheduler_patience,
    threshold=config.train_optimizer_scheduler_threshold,
    min_lr=config.train_optimizer_scheduler_minimum_lr,
    eps=0.0,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory variables
experiment_name = config.experiment_name
run_name = config.run_name
experiments_dir = config.experiments_dir
fold_name = None

prognosais_data = DataGenerator(
    data_dir=train_dir,
    file_extension=constants.DATA_NIFTI_EXTENSION,
    labels_file_dir=config.data_train_labels_file_dir,
    data_type=data_type,
    train=True,
    subset=subset,
    missing_value=missing_value,
    ids_to_exclude=None,
    seed=env_seed,
    modalities=modalities,
    mask_file_name=config.data_mask_file_name,
    data_folders=config.data_folders,
    labels_config=config.labels,
    case_id_column=config.data_label_case_id_column,
    require_mask=does_segmentation,
)

classification_weights = prognosais_data.class_weights
classification_loss_weights = prognosais_data.classification_loss_weights
weights_loss_seg = prognosais_data.weight_loss_seg

if set(classification_tasks) != set(classification_weights):
    raise ValueError(
        "Mismatch between configured classification tasks and class weights. "
        f"Configured tasks: {sorted(classification_tasks)}; "
        f"weight tasks: {sorted(classification_weights)}"
    )

if set(classification_tasks) != set(classification_loss_weights):
    raise ValueError(
        "Mismatch between configured classification tasks and classification "
        "loss weights. "
        f"Configured tasks: {sorted(classification_tasks)}; "
        f"loss-weight tasks: {sorted(classification_loss_weights)}"
    )

if config.train_kfold:
    if args.current_fold is None:
        parser.error(
            "--current_fold is required because kfold training was set to true."
        )
    kf = KFoldGenerator(num_folds=config.train_num_folds, seed=env_seed)
    idx = kf.get_folds(prognosais_data.data)
    fold_name = f"fold_{args.current_fold}"
    train_idx, val_idx = idx[args.current_fold]
    train_data = np.array(prognosais_data.data)[train_idx].tolist()
    train_data = train_data * augmentation_factor
    val_data = np.array(prognosais_data.data)[val_idx].tolist()
else:
    train_data, val_data = train_test_split(
        prognosais_data.data, test_size=0.2, random_state=env_seed
    )
    train_data = train_data * augmentation_factor

data_keys = ["structural"]
if does_segmentation:
    data_keys.insert(0, "mask")

rotation_modes = tuple(
    "nearest" if key == "mask" else "bilinear"
    for key in data_keys
)

train_transform_list = [
    LoadImaged(keys=data_keys),
    EnsureChannelFirstd(keys=data_keys),
    EnsureTyped(keys=data_keys),

    # Geometric transforms are applied identically to all image groups
    # and the segmentation mask.
    RandRotated(
        keys=data_keys,
        range_x=np.pi / 6,
        range_y=np.pi / 6,
        range_z=np.pi / 6,
        prob=augmentation_probability,
        mode=rotation_modes,
        padding_mode="border",
    ).set_random_state(env_seed),

    RandCropandZerod(
        keys=data_keys,
        max_crop=20,
        prob=augmentation_probability,
        channel_wise=False,
    ).set_random_state(env_seed),

    RandFlipd(
        keys=data_keys,
        prob=augmentation_probability,
        spatial_axis=0,
    ).set_random_state(env_seed),

    RandFlipd(
        keys=data_keys,
        prob=augmentation_probability,
        spatial_axis=1,
    ).set_random_state(env_seed),

    RandFlipd(
        keys=data_keys,
        prob=augmentation_probability,
        spatial_axis=2,
    ).set_random_state(env_seed),
]

val_transform_list = [
    LoadImaged(keys=data_keys),
    EnsureChannelFirstd(keys=data_keys),
    EnsureTyped(keys=data_keys),
]

# Only apply mask transforms if needed
if does_segmentation:
    train_transform_list.append(
        AsDiscreted(keys=["mask"], to_onehot=2)
    )

    val_transform_list.append(
        AsDiscreted(keys=["mask"], to_onehot=2)
    )

# Conventional structural-MRI intensity preprocessing and augmentation.
train_transform_list.extend(
    [
        NormalizeIntensityd(
            keys=["structural"],
            nonzero=True,
            channel_wise=True,
        ),
        RandShiftIntensityd(
            keys=["structural"],
            offsets=np.random.uniform(0, 0.2),
            prob=augmentation_probability,
        ).set_random_state(env_seed),
        RandAdjustContrastd(
            keys=["structural"],
            prob=augmentation_probability,
            gamma=(0.85, 1.15),
        ).set_random_state(env_seed),
    ]
)

val_transform_list.append(
    NormalizeIntensityd(
        keys=["structural"],
        nonzero=True,
        channel_wise=True,
    )
)

train_transform_list.append(
    ConcatItemsd(
        keys=["structural"],
        name="image",
        dim=0,
    )
)

val_transform_list.append(
    ConcatItemsd(
        keys=["structural"],
        name="image",
        dim=0,
    )
)

train_transforms = Compose(train_transform_list)
val_transforms = Compose(val_transform_list)

train_dataset = Dataset(data=train_data, transform=train_transforms)
val_dataset = Dataset(data=val_data, transform=val_transforms)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch_generator,
    worker_init_fn=seed_worker,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    generator=torch_generator,
    worker_init_fn=seed_worker,
)

if __name__ == "__main__":

    model_trainer = Trainer(
        model=model,
        modalities=modalities,
        num_epochs=num_epochs,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        weights_loss_seg=weights_loss_seg,
        classification_weights=classification_weights,
        classification_loss_weights=classification_loss_weights,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config_dir=config_dir,
    )

    model_trainer.train(
        experiment_name=experiment_name,
        run_name=run_name,
        experiments_dir=experiments_dir,
        patience=patience,
        delta=delta,
        fold_name=fold_name,
        run_id=args.run_id,
    )
