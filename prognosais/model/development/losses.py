import torch
import torch.nn as nn
from monai.losses import DiceLoss


def dice_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the Dice loss for a pair of prediction and target tensors.

    Args:
        predictions: torch.Tensor
            Model output tensor.
        targets: torch.Tensor
            Ground-truth tensor.

    Returns:
        torch.Tensor: Dice loss value.
    """

    criterion = DiceLoss(reduction="mean", softmax=True)

    loss = criterion(predictions, targets)

    return loss


def masked_cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    hard_mining: bool = False,
) -> torch.Tensor:
    """
    Compute masked cross-entropy loss for partially missing labels.

    The loss ignores samples whose target vector is entirely missing. When
    `hard_mining` is enabled, it scales the loss to emphasize misclassified
    examples.

    Args:
        predictions: torch.Tensor
            Model output tensor.
        targets: torch.Tensor
            Ground-truth tensor, one-hot encoded with missing labels encoded as
            all zeros.
        class_weights: torch.Tensor
            Class weights used by cross entropy.
        hard_mining: bool
            If True, apply hard example mining.

    Returns:
        torch.Tensor: Masked cross-entropy loss value.
    """

    # Find cases for which the ground truth is available.
    mask = targets.sum(dim=1) > 0
    predictions_masked = predictions[mask]
    targets_masked = targets[mask]

    # Get the indices of the ground truth classes
    targets_idx_masked = targets_masked.argmax(dim=1)

    # Initialize the cross-entropy loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    # Loss is zero if no ground truth available (all values are missing)
    if targets_masked.numel() == 0:
        loss = (
            predictions.sum() * 0.0
        )  # Zero loss, but maintain computational graph. Harmless since predictions.sum() is a scalar.
    else:
        # Compute the base loss
        loss = criterion(predictions_masked, targets_idx_masked)

        if hard_mining:
            print("Hard mining enabled")
            # Identify misclassified examples
            _, predicted_classes = torch.max(predictions_masked, dim=1)
            misclassified = (predicted_classes != targets_idx_masked).float()

            # Hard example mask: 1 for misclassified, 0 for correct
            hard_example_mask = misclassified

            # Scale the loss: increase the weight of misclassified examples
            weight_hard_example = 1 + hard_example_mask.mean()
            print(weight_hard_example)
            loss = (
                loss * weight_hard_example
            )  # Apply scaling factor to penalize misclassified examples

    return loss
