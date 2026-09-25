import subprocess
from pathlib import Path

import psutil
import SimpleITK as sitk

import prognosais.IO.constants as constants


def get_number_of_threads() -> int:
    """
    Return the number of CPU threads available to the current process.

    Returns:
        int: Number of CPU threads available to the process.
    """
    return len(psutil.Process().cpu_affinity())


def convert_dicoms_to_niftis(patient_dir: Path) -> None:
    """
    Convert all DICOM series for one patient to NIfTI files.

    Args:
        patient_dir: Directory containing one patient's DICOM folder.
    """
    dicom_dir = patient_dir / constants.DICOM_FOLDER
    nifti_dir = patient_dir / constants.NIFTI_FOLDER
    nifti_dir.mkdir(parents=True, exist_ok=True)

    for scan_type in constants.SCAN_TYPES:
        dicom_scan_folder = dicom_dir / scan_type
        nifti_output_path = (nifti_dir / scan_type).with_suffix(
            constants.DATA_NIFTI_EXTENSION
        )
        if nifti_output_path.exists():
            continue
        if not dicom_scan_folder.is_dir():
            raise FileNotFoundError(f"Missing DICOM series: {dicom_scan_folder}")
        subprocess.run(
            [
                "dcm2niix",
                "-9",
                "-z",
                "i",
                "-f",
                scan_type,
                "-o",
                str(nifti_dir),
                str(dicom_scan_folder),
            ],
            check=True,
        )
        if not nifti_output_path.is_file():
            raise FileNotFoundError(
                f"dcm2niix completed but did not produce {nifti_output_path}"
            )


def crop_to_mask(image: sitk.Image, mask: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    """
    Crop an image to the bounding box of a mask.

    Args:
        image: Input image to crop.
        mask: Mask defining the crop region.

    Returns:
        tuple[sitk.Image, sitk.Image]: Cropped image and cropped mask.
    """
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)
    bounding_box = label_shape_filter.GetBoundingBox(1)

    bounding_box_index = bounding_box[:3]
    bounding_box_size = bounding_box[3:]

    cropped_image = sitk.RegionOfInterest(image, bounding_box_size, bounding_box_index)
    cropped_mask = sitk.RegionOfInterest(mask, bounding_box_size, bounding_box_index)

    return cropped_image, cropped_mask


def normalize_intensity_with_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    Z-score normalize the image intensity within the masked region.

    Args:
        image: Input image to normalize.
        mask: Mask defining the region used to compute intensity statistics.

    Returns:
        sitk.Image: Normalized image.
    """
    mask_label_filter = sitk.LabelIntensityStatisticsImageFilter()
    mask_label_filter.Execute(mask, image)
    img_mean = mask_label_filter.GetMean(1)
    img_std = mask_label_filter.GetStandardDeviation(1)

    normalized_image = sitk.ShiftScale(image, -img_mean, 1.0 / img_std)
    return normalized_image


def mask_background_to_min(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    Set values outside the mask to the minimum intensity value within the mask.

    Args:
        image: Input image to mask.
        mask: Mask defining the region to keep.

    Returns:
        sitk.Image: Image with background replaced by the masked minimum.
    """
    mask_label_filter = sitk.LabelIntensityStatisticsImageFilter()
    mask_label_filter.Execute(mask, image)
    img_min = mask_label_filter.GetMinimum(1)

    masked_image = sitk.Mask(image, mask, img_min)
    return masked_image


def collapse_mask_labels(mask: sitk.Image) -> sitk.Image:
    """
    Collapse multi-label segmentation into a single label.

    Args:
        mask: Multi-label segmentation image.

    Returns:
        sitk.Image: Segmentation image with collapsed labels.
    """
    mask = sitk.LabelImageToLabelMap(mask)
    mask = sitk.AggregateLabelMap(mask)
    mask = sitk.RelabelLabelMap(mask)
    collapsed_mask = sitk.LabelMapToLabel(mask)
    collapsed_mask = sitk.Cast(collapsed_mask, sitk.sitkUInt8)
    return collapsed_mask
