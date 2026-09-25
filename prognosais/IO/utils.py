import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.utils import set_determinism


def set_random_seed(seed):
    """
    Set all supported random number generators to a fixed seed.

    This helper synchronizes the seed across Python's random module,
    NumPy, PyTorch, and MONAI so that repeated runs can reproduce the same
    sampling behavior as closely as the backend allows. When CUDA is
    available, it also enables deterministic cuDNN behavior and configures the
    workspace setting required by PyTorch for deterministic CUDA execution.

    Args:
        seed: Seed value used for all random number generators.

    Returns:
        torch.Generator: A PyTorch generator seeded with the provided value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed=seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def seed_worker(worker_id):
    """
    Seed a data-loading worker process.

    PyTorch data loader workers inherit a process-specific initial seed. This
    function derives a NumPy and Python random seed from that worker seed
    so that per-worker augmentation and sampling remain deterministic.

    Args:
        worker_id: Integer worker identifier provided by the data loader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reorient_and_visualize_itk_snap_view(
    im_dir: str, library: str, slice_number: int
) -> None:
    """
    Reorient a medical image to ITK-SNAP view and visualize orthogonal slices.

    The image is loaded either with NiBabel or SimpleITK, converted to the
    LPI-style orientation used in ITK-SNAP, and displayed in axial, sagittal,
    and coronal views for quick inspection.

    Args:
        im_dir: Path to the image file.
        library: Image-loading backend. Must be either nibabel or
            SimpleITK.
        slice_number: Slice index to visualize in each plane.

    Returns:
        None. Displays a matplotlib figure.

    Raises:
        AssertionError: If library is not one of the supported backends.
    """
    assert library in [
        "nibabel",
        "SimpleITK",
    ], "Library must be either 'nibabel' or 'SimpleITK'"

    if library == "nibabel":
        im_arr = nib.load(im_dir).get_fdata()
    else:
        im_arr = sitk.GetArrayFromImage(sitk.ReadImage(im_dir))

    im_arr = np.flip(im_arr, axis=(0, 1, 2))

    if library == "nibabel":
        print(im_arr.shape)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(im_arr[:, :, im_arr.shape[2] - 1 - slice_number].T, cmap="gray")
        plt.title(f"Axial slice {slice_number}")
        plt.subplot(1, 3, 2)
        plt.imshow(im_arr[im_arr.shape[0] - 1 - slice_number, :, :].T, cmap="gray")
        plt.title(f"Sagittal slice {slice_number}")
        plt.subplot(1, 3, 3)
        plt.imshow(im_arr[:, im_arr.shape[1] - 1 - slice_number, :].T, cmap="gray")
        plt.title(f"Coronal slice {slice_number}")
    else:
        print(im_arr.shape)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(im_arr[im_arr.shape[0] - 1 - slice_number, :, :], cmap="gray")
        plt.title(f"Axial slice {slice_number}")
        plt.subplot(1, 3, 2)
        plt.imshow(im_arr[:, :, im_arr.shape[2] - 1 - slice_number], cmap="gray")
        plt.title(f"Sagital slice {slice_number}")
        plt.subplot(1, 3, 3)
        plt.imshow(im_arr[:, im_arr.shape[1] - 1 - slice_number, :], cmap="gray")
        plt.title(f"Coronal slice {slice_number}")


def reorient_image_to_itk_snap_view(im_dir):
    """
    Reorient a medical image to the ITK-SNAP display convention.

    The image is loaded with SimpleITK, reoriented to LPI, converted to a
    NumPy array, and flipped so that downstream visualization code can index
    it in the same way as the rest of the project.

    Args:
        im_dir: Path to the image file.

    Returns:
        numpy.ndarray: Reoriented image array.
    """
    im_sitk = sitk.ReadImage(im_dir)
    im_sitk_reoriented = sitk.DICOMOrient(im_sitk, "LPI")
    im_sitk = np.moveaxis(im_sitk_reoriented, 0, -1)
    im_sitk = im_sitk[:, :, ::-1]

    return im_sitk


def plot_mri_modalities_slice_sitk(
    t1_path: str,
    t1ce_path: str,
    t2_path: str,
    flair_path: str,
    slice_idx: int,
    title: str = None,
    cmap: str = "gray",
):
    """
    Plot one axial slice from the four structural MRI modalities.

    The scans are reoriented to ITK-SNAP convention before plotting so that
    all modalities are shown in a consistent anatomical orientation.

    Args:
        t1_path: Path to the T1 image file.
        t1ce_path: Path to the T1CE image file.
        t2_path: Path to the T2 image file.
        flair_path: Path to the FLAIR image file.
        slice_idx: Axial slice index to plot.
        title: Optional figure title.
        cmap: Matplotlib colormap used for all modalities.

    Returns:
        None. Displays the plot.
    """

    modality_names = ["T1", "T1CE", "T2", "FLAIR"]
    modality_paths = [t1_path, t1ce_path, t2_path, flair_path]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("black")

    for ax, path, name in zip(axes, modality_paths, modality_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        img = reorient_image_to_itk_snap_view(path)

        if slice_idx >= img.shape[-1]:
            raise ValueError(
                f"Slice index {slice_idx} out of bounds for {name} with shape {img.shape}"
            )

        slice_img = img[:, :, slice_idx]  # Axial view

        ax.imshow(slice_img, cmap=cmap)
        ax.set_title(name, color="white")
        ax.axis("off")
        ax.set_facecolor("black")

    if title:
        fig.suptitle(title, fontsize=16, color="white")

    plt.tight_layout()
    plt.show()
