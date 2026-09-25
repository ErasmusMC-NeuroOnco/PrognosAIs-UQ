import math
import os
from typing import List

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib import pyplot as plt


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    The class tracks the best validation loss, counts epochs without
    improvement, and saves checkpoints for both best and last model states.

    Attributes:
        patience: Number of epochs without improvement before stopping.
        delta: Minimum validation-loss improvement required to reset the
            counter.
        root_path: Directory where checkpoints will be written.
    """

    def __init__(self, patience: int, delta: int, root_path: str):
        """
        Initialize the early stopping helper.

        Args:
            patience: Number of epochs without improvement before stopping.
            delta: Minimum validation-loss improvement required.
            root_path: Directory where checkpoints will be saved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.root_path = root_path

    def __call__(
        self,
        current_train_loss: float,
        current_val_loss: float,
        train_loss_history: List[float],
        train_loss_seg_history: List[float],
        train_loss_cls_history: dict[str, List[float]],
        val_loss_history: List[float],
        val_loss_seg_history: List[float],
        val_loss_cls_history: dict[str, List[float]],
        model,
        optimizer,
        epoch: int,
    ):
        """
        Evaluate whether training should stop early.

        Args:
            current_train_loss: Current training loss.
            current_val_loss: Current validation loss.
            train_loss_history: Training-loss history.
            train_loss_seg_history: Segmentation-loss history.
            train_loss_cls_history: loss history of classification tasks.
            val_loss_history: Validation-loss history.
            val_loss_seg_history: Validation segmentation-loss history.
            val_loss_cls_history: Validation loss history of classification tasks.
            model: Model being trained.
            optimizer: Optimizer being used.
            epoch: Current epoch index.
        """
        score = -current_val_loss

        if self.best_score is None:
            self.best_score = score
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {current_val_loss:.6f}). Saving model..."
            )
            self.save_checkpoint(
                train_loss=current_train_loss,
                val_loss=current_val_loss,
                train_loss_history=train_loss_history,
                train_loss_seg_history=train_loss_seg_history,
                train_loss_cls_history=train_loss_cls_history,
                val_loss_history=val_loss_history,
                val_loss_seg_history=val_loss_seg_history,
                val_loss_cls_history=val_loss_cls_history,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                file_name=os.path.join(self.root_path, "best_model.pt"),
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {current_val_loss:.6f}). Saving model..."
            )
            self.save_checkpoint(
                train_loss=current_train_loss,
                val_loss=current_val_loss,
                train_loss_history=train_loss_history,
                train_loss_seg_history=train_loss_seg_history,
                train_loss_cls_history=train_loss_cls_history,
                val_loss_history=val_loss_history,
                val_loss_seg_history=val_loss_seg_history,
                val_loss_cls_history=val_loss_cls_history,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                file_name=os.path.join(self.root_path, "best_model.pt"),
            )
            self.counter = 0

    def save_checkpoint(
        self,
        train_loss: float,
        val_loss: float,
        train_loss_history: List[float],
        train_loss_seg_history: List[float],
        train_loss_cls_history: dict[str, List[float]],
        val_loss_history: List[float],
        val_loss_seg_history: List[float],
        val_loss_cls_history: dict[str, List[float]],
        model,
        optimizer,
        epoch: int,
        file_name: str,
    ):
        """
        Save a model checkpoint to disk.

        Args:
            train_loss: Current training loss.
            val_loss: Current validation loss.
            train_loss_history: Training-loss history.
            train_loss_seg_history: Segmentation-loss history.
            train_loss_cls_history: classification task-loss history.
            val_loss_history: Validation-loss history.
            val_loss_seg_history: Validation segmentation-loss history.
            val_loss_cls_history: Validation classification task-loss history.
            model: Model object.
            optimizer: Optimizer object.
            epoch: Current epoch index.
            file_name: Output checkpoint file name.
        """

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_loss_history": train_loss_history,
                "train_loss_seg_history": train_loss_seg_history,
                "train_loss_cls_history": train_loss_cls_history,
                "val_loss": val_loss,
                "val_loss_history": val_loss_history,
                "val_loss_seg_history": val_loss_seg_history,
                "val_loss_cls_history": val_loss_cls_history,
            },
            os.path.join(self.root_path, file_name),
        )
        self.val_loss_min = val_loss


def merge_activation_maps(
    activation_maps: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Merge activation maps from multiple batches into one dictionary.

    Args:
        activation_maps: List of per-batch activation dictionaries. Each
            dictionary maps layer name to a tensor of shape
            [batch_size, channels, x, y, z].

    Returns:
        dict[str, torch.Tensor]: Layer name to concatenated activations.
    """
    merged = {}

    # Get all layer names from the first batch
    layer_names = activation_maps[0].keys()

    for layer in layer_names:
        # Collect all batches for this layer
        layer_batches = [batch[layer] for batch in activation_maps]

        # Concatenate along the batch dimension (dim=0)
        merged[layer] = torch.cat(layer_batches, dim=0)

    return merged


def get_scan_slice_all_views(
    scan: str | np.ndarray, library: str, slice_number: int
) -> dict[str, np.ndarray]:
    """
    Extract axial, sagittal, and coronal slices from a 3D image.

    Args:
        scan: Path to the image file or a 3D NumPy array.
        library: Loading backend. Must be either nibabel or SimpleITK.
        slice_number: Slice index to extract in each plane.

    Returns:
        dict[str, np.ndarray]: Dictionary with axial, sagittal, and coronal
        slices.

    Raises:
        AssertionError: If library is not supported.
        ValueError: If slice_number is out of bounds for the image dimensions.
    """

    assert library in [
        "nibabel",
        "SimpleITK",
    ], "Library must be either 'nibabel' or 'SimpleITK'"

    if isinstance(scan, str):
        if library == "nibabel":
            im_arr = nib.load(scan).get_fdata()
        else:
            im_arr = sitk.GetArrayFromImage(sitk.ReadImage(scan))
    elif isinstance(scan, np.ndarray):
        if scan.ndim != 3:
            raise ValueError("Input numpy array must be 3D.")
        im_arr = scan
    else:
        raise TypeError("scan must be either a file path (str) or a 3D numpy array.")

    im_arr = np.flip(im_arr, axis=(0, 1, 2))

    scan_views = {}

    if (
        slice_number >= im_arr.shape[0]
        or slice_number >= im_arr.shape[1]
        or slice_number >= im_arr.shape[2]
    ):
        raise ValueError(
            f"Slice number {slice_number} out of bounds for image with shape {im_arr.shape}"
        )

    if library == "nibabel":
        axial_slice = im_arr[:, :, im_arr.shape[2] - 1 - slice_number].T
        sagittal_slice = im_arr[im_arr.shape[0] - 1 - slice_number, :, :].T
        coronal_slice = im_arr[:, im_arr.shape[1] - 1 - slice_number, :].T
    else:
        axial_slice = im_arr[im_arr.shape[0] - 1 - slice_number, :, :]
        sagittal_slice = im_arr[:, :, im_arr.shape[2] - 1 - slice_number]
        coronal_slice = im_arr[:, im_arr.shape[1] - 1 - slice_number, :]

    scan_views["axial"] = axial_slice
    scan_views["sagittal"] = sagittal_slice
    scan_views["coronal"] = coronal_slice

    return scan_views


def plot_mri_modalities_all_views(
    t1_dict: dict[str, np.ndarray],
    t1ce_dict: dict[str, np.ndarray],
    t2_dict: dict[str, np.ndarray],
    flair_dict: dict[str, np.ndarray],
    out_im_dir: str,
    title: str = None,
    cmap: str = "gray",
) -> None:
    """
    Plot the three orthogonal views for each structural modality.

    Args:
        t1_dict: Axial, sagittal, and coronal slices for T1.
        t1ce_dict: Axial, sagittal, and coronal slices for T1CE.
        t2_dict: Axial, sagittal, and coronal slices for T2.
        flair_dict: Axial, sagittal, and coronal slices for FLAIR.
        out_im_dir: Output image path.
        title: Optional figure title.
        cmap: Colormap used for image display.

    Returns:
        None. Displays and saves the plot.
    """

    modality_names = ["T1", "T1CE", "T2", "FLAIR"]

    # Make sure the keys are present
    required_keys = ["axial", "sagittal", "coronal"]
    for key in required_keys:
        if (
            key not in t1_dict
            or key not in t1ce_dict
            or key not in t2_dict
            or key not in flair_dict
        ):
            raise ValueError(
                f"Missing key '{key}' in one of the modality dictionaries."
            )

    modality_dicts = [t1_dict, t1ce_dict, t2_dict, flair_dict]

    n_rows = len(modality_dicts)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig.patch.set_facecolor("black")  # background color

    # If only one row, make sure axes is 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, d in enumerate(modality_dicts):
        for j, view in enumerate(required_keys):
            ax = axes[i, j]
            img = d[view]

            ax.imshow(img, cmap="gray")
            if i == 0:
                ax.set_title(view.capitalize(), color="white", fontsize=12)
            # ax.axis("off")
            if j == 0:
                ax.set_ylabel(modality_names[i], color="white", fontsize=12)
            ax.set_facecolor("black")

    # Add supertitle
    fig.suptitle(title, color="white", fontsize=16, weight="bold")

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_im_dir, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def map_slice_index(
    slice_num: int, original_dim: int, downsampled_dim: int, mode: str = "round"
) -> int:
    """
    Map a slice index from one resolution to another.

    Args:
        slice_num: Index in the original dimension.
        original_dim: Size of the original dimension.
        downsampled_dim: Size of the target dimension.
        mode: Mapping strategy, either round or floor.

    Returns:
        int: Mapped slice index in the target dimension.
    """

    if original_dim <= 1:
        return 0
    assert mode in ["round", "floor"], "mode must be 'round' or 'floor'"
    if mode == "floor":
        j = int(math.floor(slice_num * downsampled_dim / original_dim))
    else:  # "round" using endpoints mapping
        scale = (downsampled_dim - 1) / (original_dim - 1)
        j = int(round(slice_num * scale))
    # clamp to valid range
    return max(0, min(downsampled_dim - 1, j))


def prepare_layer_for_visualization(layer_tensor: torch.Tensor) -> np.ndarray:
    """
    Prepare a 5D layer tensor for visualization.

    Args:
        layer_tensor: Tensor of shape [n_patients, n_filters, x, y, z].

    Returns:
        np.ndarray: Array prepared for visualization.

    """

    if layer_tensor.ndim != 5:
        raise ValueError(
            "Input tensor must be 5D with shape (n_patients, n_filters, x, y, z)."
        )

    n_patients, n_filters, x, y, z = layer_tensor.shape
    layer_tensor_out = layer_tensor.clone().numpy()

    for patient in range(n_patients):
        for filt in range(n_filters):
            layer_tensor_out[patient, filt] = np.flip(
                layer_tensor_out[patient, filt], axis=(0, 1, 2)
            )

    return layer_tensor_out


def plot_feature_maps_grid(
    feature_map: np.ndarray,
    patient_index: int,
    slice_index: int,
    view: str = "axial",
    title: str = None,
    cmap: str = "gray",
) -> None:
    """
    Plot all feature maps for one patient at one slice.

    Args:
        feature_map: Array of shape [n_patients, n_filters, x, y, z].
        patient_index: Patient index to visualize.
        slice_index: Slice index to visualize.
        view: One of axial, sagittal, or coronal.
        title: Optional figure title.
        cmap: Colormap for imshow.

    Returns:
        None. Displays the plot.
    """
    assert (
        feature_map.ndim == 5
    ), f"Expected feature map with shape (n_patients, n_filters, X, Y, Z), got {feature_map.shape}"
    n_patients, n_filters, X, Y, Z = feature_map.shape

    assert (
        0 <= patient_index < n_patients
    ), f"patient_index must be in [0, {n_patients-1}]"

    feature_map = feature_map[patient_index]  # shape: (n_filters, X, Y, Z)

    grid_cols = int(math.ceil(math.sqrt(n_filters)))
    grid_rows = int(math.ceil(n_filters / grid_cols))

    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2)
    )
    fig.patch.set_facecolor("black")

    for i in range(grid_rows * grid_cols):

        ax = axes.flat[i]
        if i < n_filters:
            if view == "axial":
                assert (
                    0 <= slice_index < Z
                ), f"slice_index must be in [0, {Z-1}] for axial view"
                img = feature_map[i, :, :, feature_map.shape[3] - 1 - slice_index].T
            elif view == "sagittal":
                assert (
                    0 <= slice_index < X
                ), f"slice_index must be in [0, {X-1}] for sagittal view"
                img = feature_map[i, feature_map.shape[1] - 1 - slice_index, :, :].T
            elif view == "coronal":
                assert (
                    0 <= slice_index < Y
                ), f"slice_index must be in [0, {Y-1}] for coronal view"
                img = feature_map[i, :, feature_map.shape[2] - 1 - slice_index, :].T
            ax.imshow(img, cmap=cmap)
            ax.text(
                5,
                15,
                f"Filter {i+1}",
                color="white",
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )
        ax.axis("off")
        ax.set_facecolor("black")

    if title:
        fig.suptitle(title, fontsize=16, color="white")
    else:
        fig.suptitle(
            f"Patient {patient_index} - {view.capitalize()} view - Slice {slice_index}",
            fontsize=16,
            color="white",
        )

    plt.tight_layout()
    plt.show()
