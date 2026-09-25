"""Shared model and transform construction for inference workflows."""

from __future__ import annotations

import torch.nn as nn
from monai.transforms import (
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
)

from prognosais.IO import constants
from prognosais.IO.config import Config
from prognosais.model.architectures.CSNet import CSNet


def get_classification_task_sizes(config: Config) -> dict[str, int]:
    """Return enabled classification tasks and their output dimensions.

    Args:
        config: Parsed project configuration.

    Returns:
        Mapping from internal task key to number of classes, preserving the
        configured task order.
    """

    return {
        task_name: config.get_label_num_classes(task_name)
        for task_name in config.classification_tasks
    }


def model_does_segmentation(config: Config) -> bool:
    """Determine whether the configured model emits segmentation logits.

    Args:
        config: Parsed project configuration.

    Returns:
        True only for a CSNet model whose classification-only mode is disabled.
    """

    return (
        config.model_architecture == constants.CSNET_MODEL_TYPE
        and not config.classification_only
    )


def build_model_from_config(config: Config, dropout_rate: float) -> nn.Module:
    """Construct the configured inference model with matching task heads.

    Args:
        config: Parsed project configuration describing architecture,
            modalities, and classification tasks.
        dropout_rate: Dropout probability to install in the model. MCD can use
            an inference-time rate, while deterministic inference normally
            passes the training rate.

    Returns:
        Uninitialized CSNet model matching the configuration.

    Raises:
        ValueError: If no modalities are enabled or the configured architecture
            is unsupported.
    """

    modalities = config.data_modalities
    if not modalities:
        raise ValueError("At least one imaging modality must be enabled.")

    task_sizes = get_classification_task_sizes(config)
    architecture = config.model_architecture
    if architecture == constants.CSNET_MODEL_TYPE:
        return CSNet(
            dropout_rate=dropout_rate,
            in_channels=len(modalities),
            classification_tasks=task_sizes,
            classification_only=config.classification_only,
        )
    raise ValueError(
        f"Unsupported model architecture '{architecture}'. Supported values: "
        + ", ".join(constants.MODEL_TYPES)
    )


def build_inference_transforms(config: Config, require_mask: bool) -> Compose:
    """Build transforms for the configured conventional MRI modalities.

    The modalities are loaded and normalized together in canonical structural
    order, then concatenated into the model image tensor.

    Args:
        config: Parsed project configuration.
        require_mask: Whether a segmentation mask is present and should be
            loaded and one-hot encoded.

    Returns:
        MONAI transform pipeline producing the image key expected by models.

    Raises:
        ValueError: If the configuration enables no supported modalities.
    """

    if not config.data_modalities:
        raise ValueError("At least one conventional MRI modality is required.")

    data_keys = ["structural"]
    if require_mask:
        data_keys.insert(0, "mask")

    transforms = [
        LoadImaged(keys=data_keys),
        EnsureChannelFirstd(keys=data_keys),
    ]
    transforms.append(
        NormalizeIntensityd(
            keys=["structural"],
            nonzero=True,
            channel_wise=True,
        )
    )
    if require_mask:
        transforms.append(
            AsDiscreted(
                keys=["mask"],
                to_onehot=constants.NUM_SEGMENTATION_CLASSES,
            )
        )
    transforms.append(
        ConcatItemsd(
            keys=["structural"],
            name="image",
            dim=0,
        )
    )
    return Compose(transforms)
